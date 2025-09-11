# app/main.py

import uuid
import json
import traceback
from datetime import datetime, date
from typing import List, Optional
import requests

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Body, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc, func

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    AsyncIOScheduler = None
    CronTrigger = None

# Import from our new modules
from . import config, models, schemas, services, auth
from .database import engine, get_db, SessionLocal

# Create DB tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# --- FastAPI App Initialization ---
app = FastAPI(title="Psych Support - RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- App Events ---
@app.on_event("startup")
async def start_email_scheduler():
    services.init_embedder()
    services.init_qdrant()
    print("Application startup: RAG services initialized.")

    if not config.ENABLE_EMAIL_SCHEDULER or AsyncIOScheduler is None or CronTrigger is None:
        print("[SCHED] Scheduler disabled or APScheduler not installed.")
        return
    try:
        scheduler = AsyncIOScheduler(timezone=config.LOCAL_TZ)
        def _job_wrapper():
            db = SessionLocal()
            try:
                n = services.send_reminders_to_incomplete_users(db)
                print(f"[SCHED] Reminders sent: {n}")
            finally:
                db.close()
        
        scheduler.add_job(_job_wrapper, CronTrigger(hour=14, minute=0, timezone=config.LOCAL_TZ))
        scheduler.add_job(_job_wrapper, CronTrigger(hour=20, minute=0, timezone=config.LOCAL_TZ))
        scheduler.start()
        print("[SCHED] Email scheduler started (IST 14:00 & 20:00).")
    except Exception as e:
        print("[SCHED] Failed to start:", e)

# ---------------- Routes: Meta ----------------
@app.get("/health")
def health(user=Depends(auth.get_current_user)):
    return {
        "ok": True,
        "qdrant": services.init_qdrant() is not None,
        "embedder": services.init_embedder() is not None,
        "me": {
            "email": "admin" if auth.is_admin_principal(user) else user.email,
            "username": "admin" if auth.is_admin_principal(user) else services.username_from_email(user.email),
            "is_admin": auth.is_admin_principal(user),
        },
        "provider_defaults": {"mistral": config.OLLAMA_MODEL, "groq": config.GROQ_MODEL_DEFAULT if config.GROQ_API_KEY else None},
    }

# ---------------- Routes: Auth ----------------
@app.post("/auth/register")
def register(payload: schemas.RegisterIn, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered. Please log in instead.")
    user = models.User(email=payload.email, password_hash=auth.hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)
    token = auth.create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer", "is_admin": False}

@app.post("/auth/login")
def login(payload: schemas.LoginIn, db: Session = Depends(get_db)):
    if payload.identifier == "admin" and payload.password == "admin":
        token = auth.create_access_token("admin")
        return {"access_token": token, "token_type": "bearer", "is_admin": True}
    user = db.query(models.User).filter(models.User.email == payload.identifier).first()
    if not user or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email/username or password")
    token = auth.create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer", "is_admin": False}

@app.post("/register")
def register_alias(payload: schemas.RegisterIn, db: Session = Depends(get_db)): return register(payload, db)
@app.post("/login")
def login_alias(payload: schemas.LoginIn, db: Session = Depends(get_db)): return login(payload, db)

# ---------------- Routes: Sessions/Chats ----------------
@app.get("/sessions")
def list_sessions(user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): return []
    rows = db.query(models.Conversation).filter(models.Conversation.user_id == user.id).order_by(models.Conversation.created_at.desc()).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat()} for r in rows]

@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admins don't have chat history.")
    msgs = db.query(models.Message).filter(models.Message.conversation_id == session_id, models.Message.user_id == user.id).order_by(models.Message.created_at.asc()).all()
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in msgs]

@app.patch("/sessions/{session_id}")
def update_session_title(session_id: str, payload: schemas.UpdateTitleIn, user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admins cannot rename user sessions.")
    conv = db.query(models.Conversation).filter(models.Conversation.id == session_id, models.Conversation.user_id == user.id).first()
    if not conv: raise HTTPException(status_code=404, detail="Session not found")
    conv.title = payload.title[:255]
    db.commit()
    return {"id": conv.id, "title": conv.title}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str = Path(...), user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admins cannot delete user sessions.")
    conv = db.query(models.Conversation).filter(models.Conversation.id == session_id, models.Conversation.user_id == user.id).first()
    if not conv: raise HTTPException(status_code=404, detail="Session not found")
    db.query(models.Message).filter(models.Message.conversation_id == session_id, models.Message.user_id == user.id).delete()
    db.delete(conv); db.commit()
    return {"ok": True}

# ---------------- Routes: Ingest ----------------
@app.post("/ingest")
def ingest(file: UploadFile = File(...), user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admin cannot ingest documents.")
    if not file.filename or not file.filename.lower().endswith(".txt"): raise HTTPException(status_code=400, detail="Only .txt supported.")
    raw = file.file.read().decode("utf-8", errors="ignore")
    chunks = [raw[i:i+800].strip() for i in range(0, len(raw), 800) if raw[i:i+800].strip()]
    payloads = [{"id": str(uuid.uuid4()), "text": c, "meta": {"uploader": user.email, "filename": file.filename}, "user_id": user.id} for c in chunks]
    db.add(models.Document(id=str(uuid.uuid4()), uploader=user.email, filename=file.filename)); db.commit()
    count = services.upsert_documents_to_qdrant(payloads)
    return {"ok": True, "ingested_chunks": count}

# ---------------- Routes: Chat (Main RAG Endpoint) ----------------
@app.post("/chat", response_model=schemas.ChatOut)
def chat(body: schemas.ChatIn, user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admin cannot chat.")
    try:
        contexts = services.search_qdrant(body.message, limit=config.TOP_K, user_id=user.id)
    except Exception as e:
        print("[RAG] search error:", e, traceback.format_exc()); contexts = []

    prompt, numbered_ctx, ooc = services.build_prompt_with_numbered_context(body.message, contexts)
    max_toks = body.num_predict or 512
    raw_reply = services.call_groq(prompt, num_predict=max_toks) if body.provider == "groq" else services.call_ollama(prompt, num_predict=max_toks)
    
    reply = (raw_reply or "").strip()
    # if ooc and not reply.startswith(config.OOC_PREFIX): reply = f"{config.OOC_PREFIX}\n\n{reply}"
    
    reply_with_marker = services.append_cited_sources_marker(reply, numbered_ctx)
    
    conv_id = body.session_id
    if not conv_id:
        conv = models.Conversation(id=str(uuid.uuid4()), user_id=user.id, title=services.short_title(body.message))
        db.add(conv); db.commit(); conv_id = conv.id
    
    try:
        db.add_all([
            models.Message(conversation_id=conv_id, user_id=user.id, role="user", content=body.message),
            models.Message(conversation_id=conv_id, user_id=user.id, role="assistant", content=reply_with_marker),
        ])
        db.commit()
    except Exception as e:
        print("[DB] save failed:", e, traceback.format_exc())

    _, items = services.split_reply_and_sources_for_api(reply_with_marker)
    return {"reply": reply_with_marker, "sources": items}

@app.post("/chat_stream")
def chat_stream(body: schemas.ChatIn, user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admin cannot chat.")
    
    # The streaming logic is complex and best kept directly in the route handler.
    def gen():
        try:
            contexts = services.search_qdrant(body.message, limit=config.TOP_K, user_id=user.id)
        except Exception as e:
            print("[RAG] search error:", e, traceback.format_exc()); contexts = []
        
        prompt, numbered_ctx, ooc = services.build_prompt_with_numbered_context(body.message, contexts)
        max_toks = body.num_predict or 512
        chunks: List[str] = []

        try:
            # if ooc: yield json.dumps({"type":"delta","text": f"{config.OOC_PREFIX}\n\n"}) + "\n"
            if body.provider == "groq":
                if not services._groq_client:
                    yield json.dumps({"type":"delta","text":"[Groq key missing]\n"}) + "\n"
                else:
                    stream = services._groq_client.chat.completions.create(
                        model=config.GROQ_MODEL_DEFAULT,
                        messages=[{"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":prompt}],
                        max_tokens=max_toks, temperature=0.2, stream=True,
                    )
                    for chunk in stream:
                        d = chunk.choices[0].delta.content or ""
                        if d: chunks.append(d); yield json.dumps({"type":"delta","text": d}) + "\n"
            else:
                 with requests.post(
                    f"{config.OLLAMA_URL}/api/generate",
                    json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": True, "options": {"num_predict": max_toks}},
                    stream=True, timeout=600,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue
                        try:
                            ev = json.loads(line)
                            d = ev.get("response", "")
                            if d: chunks.append(d); yield json.dumps({"type":"delta","text": d}) + "\n"
                        except Exception: continue
        except requests.exceptions.Timeout:
            yield json.dumps({"type":"delta","text":"[Provider timed out]\n"}) + "\n"
        finally:
            body_text = "".join(chunks).strip()
            final_text = f"{config.OOC_PREFIX}\n\n{body_text}" if ooc and not body_text.startswith(config.OOC_PREFIX) else body_text
            final_text_with_marker = services.append_cited_sources_marker(final_text, numbered_ctx)
            
            conv_id = body.session_id
            if not conv_id:
                conv = models.Conversation(id=str(uuid.uuid4()), user_id=user.id, title=services.short_title(body.message))
                db.add(conv); db.commit(); db.refresh(conv); conv_id = conv.id

            try:
                db.add_all([
                    models.Message(conversation_id=conv_id, user_id=user.id, role="user", content=body.message),
                    models.Message(conversation_id=conv_id, user_id=user.id, role="assistant", content=final_text_with_marker),
                ])
                db.commit()
            except Exception as e:
                print("[DB] save failed (stream):", e, traceback.format_exc())
            
            yield json.dumps({"type":"delta","text": final_text_with_marker[len(final_text):]}) + "\n"
            _, sources = services.split_reply_and_sources_for_api(final_text_with_marker)
            yield json.dumps({"type":"sources", "sources": sources}) + "\n"
            yield json.dumps({"type":"done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

# ---------------- Routes: Videos ----------------
@app.get("/videos", response_model=List[schemas.VideoOut])
def list_videos(q: Optional[str] = None, db: Session = Depends(get_db), user=Depends(auth.get_current_user)):
    qry = db.query(models.Video)
    if not auth.is_admin_principal(user):
        qry = qry.filter(models.Video.access.in_(["all", "users"]), models.Video.is_public == 1)
    if q:
        like = f"%{q.strip()}%"
        qry = qry.filter(or_(models.Video.title.ilike(like), models.Video.description.ilike(like), models.Video.tags.ilike(like)))
    rows = qry.order_by(desc(models.Video.created_at)).all()
    return [{**v.__dict__, "tags": services._str_to_tags(v.tags), "is_public": bool(v.is_public)} for v in rows]

# ---------------- Routes: Questionnaire ----------------
@app.post("/questionnaire/start", response_model=schemas.QStartOut)
def questionnaire_start(user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admin cannot start questionnaires.")
    conv = models.Conversation(id=str(uuid.uuid4()), user_id=user.id, title=f"[Questionnaire] {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    db.add(conv); db.commit()
    qs = models.QuestionnaireSession(session_id=conv.id, user_id=user.id, target_questions=config.MIN_QUESTIONS, is_active=1)
    db.add(qs); db.commit()
    first_q = "To get started, how would you describe your overall mood today?"
    db.add(models.Message(conversation_id=conv.id, user_id=user.id, role="assistant", content=first_q))
    db.commit()
    return {"session_id": conv.id, "question": first_q}

@app.post("/questionnaire/answer", response_model=schemas.QAnswerOut)
def questionnaire_answer(body: schemas.QAnswerIn, user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): raise HTTPException(status_code=403, detail="Admin cannot answer questionnaires.")
    db.add(models.Message(conversation_id=body.session_id, user_id=user.id, role="user", content=body.answer)); db.commit()
    qs = db.query(models.QuestionnaireSession).filter(models.QuestionnaireSession.session_id == body.session_id, models.QuestionnaireSession.user_id == user.id).first()
    if not qs or not qs.is_active: raise HTTPException(status_code=400, detail="Questionnaire session not active.")
    
    hist = services.q_history(db, user.id, body.session_id)
    last_assistant_msg = next((m["content"] for m in reversed(hist[:-1]) if m["role"] == "assistant"), None)
    actual_q_count = sum(1 for m in hist if m["role"] == "assistant" and services.is_regular_question_text(m["content"]))

    def finalize_and_save():
        data = services.finalize_score_from_history(hist)
        score, comment, tips = data.get("score", 0.0), data.get("comment", ""), data.get("tips", [])
        summary = f"**Summary**\nScore: {score:.1f}/100\n\n{comment}\n\nTips:\n" + "\n".join([f"- {t}" for t in tips])
        db.add(models.Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=summary))
        db.add(models.QuestionnaireAttempt(user_id=user.id, day=date.today().isoformat(), score=score, comment=comment, tips_json=json.dumps(tips), session_id=body.session_id))
        qs.is_active = 0
        db.commit()
        return {"done": True, "score": score, "comment": comment, "tips": tips}

    if last_assistant_msg and services.is_continue_prompt_text(last_assistant_msg):
        ans = body.answer.strip().lower()
        if any(x in ans for x in ["continue", "yes", "more"]):
            qs.target_questions = (qs.target_questions or config.MIN_QUESTIONS) + config.CONTINUE_STEP
            db.commit()
            q = services.next_question_from_llm(hist)
            db.add(models.Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=q)); db.commit()
            return {"done": False, "question": q}
        elif any(x in ans for x in ["score", "finish", "end"]):
            return finalize_and_save()
    
    if actual_q_count >= (qs.target_questions or config.MIN_QUESTIONS):
        prompt_text = f"You've answered {actual_q_count} questions. Would you like to continue with {config.CONTINUE_STEP} more, or get your score? Reply with 'continue' or 'score'."
        db.add(models.Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=prompt_text)); db.commit()
        return {"done": False, "question": prompt_text}

    q = services.next_question_from_llm(hist)
    db.add(models.Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=q)); db.commit()
    return {"done": False, "question": q}

@app.get("/questionnaire/dashboard", response_model=List[schemas.QDashItem])
def questionnaire_dashboard(user=Depends(auth.get_current_user), db: Session = Depends(get_db)):
    if auth.is_admin_principal(user): return []
    agg_rows = db.query(
        models.QuestionnaireAttempt.day.label("day"),
        func.count(models.QuestionnaireAttempt.id).label("attempts"),
        func.avg(models.QuestionnaireAttempt.score).label("avg_score"),
    ).filter(models.QuestionnaireAttempt.user_id == user.id).group_by("day").order_by(desc("day")).all()
    
    sub_latest = db.query(
        models.QuestionnaireAttempt.day.label("day"),
        func.max(models.QuestionnaireAttempt.created_at).label("max_ts"),
    ).filter(models.QuestionnaireAttempt.user_id == user.id).group_by("day").subquery()
    
    latest_rows = db.query(models.QuestionnaireAttempt.day, models.QuestionnaireAttempt.comment).join(sub_latest, (models.QuestionnaireAttempt.day == sub_latest.c.day) & (models.QuestionnaireAttempt.created_at == sub_latest.c.max_ts)).filter(models.QuestionnaireAttempt.user_id == user.id).all()
    latest_map = {d: c for d, c in latest_rows}
    
    return [{"day": r.day, "attempts": r.attempts, "avg_score": round(r.avg_score or 0.0, 2), "latest_comment": latest_map.get(r.day)} for r in agg_rows]

# ---------------- Routes: Admin ----------------
@app.get("/admin/users")
def admin_list_users(db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    out = []
    for u in users:
        out.append({
            "id": u.id, "email": u.email, "created_at": u.created_at.isoformat(),
            "conversations": db.query(models.Conversation).filter(models.Conversation.user_id == u.id).count(),
            "messages": db.query(models.Message).filter(models.Message.user_id == u.id).count(),
            "questionnaire_attempts": db.query(models.QuestionnaireAttempt).filter(models.QuestionnaireAttempt.user_id == u.id).count(),
            "active_questionnaires": db.query(models.QuestionnaireSession).filter(models.QuestionnaireSession.user_id == u.id, models.QuestionnaireSession.is_active == 1).count()
        })
    return out

@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: int, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    services.delete_user_vectors(user_id)
    db.query(models.Message).filter(models.Message.user_id == user_id).delete()
    db.query(models.Conversation).filter(models.Conversation.user_id == user_id).delete()
    db.query(models.QuestionnaireAttempt).filter(models.QuestionnaireAttempt.user_id == user_id).delete()
    db.query(models.QuestionnaireSession).filter(models.QuestionnaireSession.user_id == user_id).delete()
    db.query(models.User).filter(models.User.id == user_id).delete()
    db.commit()
    return {"ok": True}

@app.patch("/admin/users/{user_id}/password")
def admin_update_user_password(user_id: int, new_pw: str = Body(..., embed=True), db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = auth.hash_password(new_pw)
    db.commit()
    return {"ok": True}

@app.delete("/admin/users/{user_id}/chats")
def admin_clear_user_chats(user_id: int, purge_vectors: bool = False, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    db.query(models.Message).filter(models.Message.user_id == user_id).delete()
    db.query(models.Conversation).filter(models.Conversation.user_id == user_id).delete()
    db.commit()
    if purge_vectors: services.delete_user_vectors(user_id)
    return {"ok": True}

@app.delete("/admin/users/{user_id}/questionnaires")
def admin_clear_user_questionnaires(user_id: int, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    db.query(models.QuestionnaireAttempt).filter(models.QuestionnaireAttempt.user_id == user_id).delete()
    db.query(models.QuestionnaireSession).filter(models.QuestionnaireSession.user_id == user_id).delete()
    db.commit()
    return {"ok": True}

@app.delete("/admin/users/{user_id}/vectors")
def admin_clear_user_vectors(user_id: int, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    services.delete_user_vectors(user_id)
    return {"ok": True}

@app.post("/admin/videos", response_model=schemas.VideoOut)
def admin_add_video(body: schemas.VideoCreate, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    v = models.Video(id=str(uuid.uuid4()), title=body.title.strip(), url=services.normalize_youtube_url(str(body.url)),
                    description=(body.description or "").strip() or None, tags=services._tags_to_str(body.tags),
                    is_public=1 if body.is_public else 0, added_by="admin", access="all")
    db.add(v); db.commit(); db.refresh(v)
    return {**v.__dict__, "tags": services._str_to_tags(v.tags), "is_public": bool(v.is_public)}

@app.patch("/admin/videos/{video_id}", response_model=schemas.VideoOut)
def admin_update_video(video_id: str, body: schemas.VideoUpdate, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    v = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not v: raise HTTPException(status_code=404, detail="Video not found")
    if body.title is not None: v.title = body.title.strip()
    if body.url is not None: v.url = services.normalize_youtube_url(str(body.url))
    if body.description is not None: v.description = (body.description or "").strip() or None
    if body.tags is not None: v.tags = services._tags_to_str(body.tags)
    if body.is_public is not None: v.is_public = 1 if body.is_public else 0
    db.commit(); db.refresh(v)
    return {**v.__dict__, "tags": services._str_to_tags(v.tags), "is_public": bool(v.is_public)}

@app.delete("/admin/videos/{video_id}")
def admin_delete_video(video_id: str, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    v = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not v: raise HTTPException(status_code=404, detail="Video not found")
    db.delete(v); db.commit()
    return {"ok": True}

@app.delete("/admin/vectorstore")
def admin_clear_vector_db(db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    qc = services.init_qdrant()
    if qc:
        try: qc.delete_collection(config.COLLECTION_NAME)
        except Exception: pass
    db.query(models.Document).delete()
    db.commit()
    return {"ok": True, "message": f"Vector collection '{config.COLLECTION_NAME}' deleted and document records cleared."}

@app.post("/admin/emails/test")
def admin_test_email(to: str = Body(..., embed=True), admin=Depends(auth.require_admin)):
    try:
        ok = services.send_email(to_email=to, subject="Psych Support: SMTP Test", html_body="<div>✅ SMTP test from Psych Support.</div>")
        return {"ok": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMTP error: {e!r}")

@app.post("/admin/emails/reminders")
def admin_send_reminders(db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    return {"ok": True, "sent": services.send_reminders_to_incomplete_users(db)}

@app.post("/admin/emails/daily-summaries")
def admin_send_daily_summaries(db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    return {"ok": True, "sent": services.send_daily_summaries_today(db)}

@app.post("/admin/emails/periodic-summaries")
def admin_send_periodic_summaries(db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    return {"ok": True, "sent": services.send_periodic_summaries_all(db)}