# app/services.py

import uuid
import json
import re
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from typing import List, Optional, Dict, Any, Tuple
from datetime import date

import requests
from qdrant_client import QdrantClient, models as qmodels
from sentence_transformers import SentenceTransformer
from groq import Groq
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException

from . import config, models

# --- Globals for expensive-to-load models ---
_qdrant: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None
_groq_client: Optional[Groq] = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None


# ---------------- RAG (Vector DB) Service ----------------
def init_embedder() -> Optional[SentenceTransformer]:
    global _embedder
    if _embedder is not None: return _embedder
    try:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
        print("[RAG] Embedder loaded:", config.EMBED_MODEL)
        return _embedder
    except Exception as e:
        print("[RAG] Embedder load failed:", e)
        _embedder = None
        return None

def init_qdrant() -> Optional[QdrantClient]:
    global _qdrant
    if _qdrant is not None: return _qdrant
    try:
        _qdrant = QdrantClient(url=config.QDRANT_URL)
        try:
            _qdrant.get_collection(config.COLLECTION_NAME)
        except Exception:
            emb = init_embedder()
            if emb is None: raise RuntimeError("Embedder failed, cannot create collection.")
            vec_size = emb.get_sentence_embedding_dimension()
            _qdrant.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=vec_size, distance=qmodels.Distance.COSINE),
            )
            _qdrant.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name="user_id",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
        print("[RAG] Qdrant connected:", config.QDRANT_URL)
        return _qdrant
    except Exception as e:
        print("[RAG] Qdrant init failed:", e)
        _qdrant = None
        return None

def embed_texts(texts: List[str]) -> List[List[float]]:
    emb = init_embedder()
    if emb is None: raise RuntimeError("Embedder not available")
    vectors = emb.encode(texts, normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

def upsert_documents_to_qdrant(payloads: List[dict]) -> int:
    qc = init_qdrant()
    if qc is None: raise RuntimeError("Qdrant not available")
    vectors = embed_texts([p["text"] for p in payloads])
    points = [
        qmodels.PointStruct(
            id=p["id"],
            vector=vectors[i],
            payload={"text": p["text"], "meta": p.get("meta", {}), "user_id": p.get("user_id")},
        )
        for i, p in enumerate(payloads)
    ]
    qc.upsert(collection_name=config.COLLECTION_NAME, points=points)
    return len(points)

def search_qdrant(query: str, limit: int, user_id: Optional[int] = None) -> List[dict]:
    qc = init_qdrant()
    if qc is None: return []
    qvec = embed_texts([query])[0]
    q_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]) if user_id is not None else None
    
    hits = qc.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=qvec,
        limit=limit,
        query_filter=q_filter,
    )
    return [
        {"id": str(h.id), "text": h.payload.get("text", ""), "meta": h.payload.get("meta", {}), "score": h.score or 0.0}
        for h in hits
    ]

def delete_user_vectors(user_id: int) -> None:
    qc = init_qdrant()
    if not qc: return
    try:
        qc.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))])
            ),
            wait=True,
        )
        print(f"[Qdrant] Deleted vectors for user_id={user_id}")
    except Exception as e:
        print("[Qdrant] delete_user_vectors failed:", e)

# ---------------- LLM Service ----------------
def call_ollama(prompt: str, num_predict: int) -> str:
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": num_predict}},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text") or ""
    except requests.exceptions.Timeout:
        return "The local model (Mistral) timed out. For big questions, switch provider to Groq."
    except Exception as e:
        print("[LLM] Ollama call failed:", e)
        return "Local model is not available right now."

def call_groq(prompt: str, num_predict: int) -> str:
    if not _groq_client: return "Groq API key not configured. Add GROQ_API_KEY to .env."
    try:
        chat_completion = _groq_client.chat.completions.create(
            model=config.GROQ_MODEL_DEFAULT,
            messages=[
                {"role": "system", "content": "You are a helpful, compassionate assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=num_predict,
            temperature=0.2,
        )
        content = chat_completion.choices[0].message.content
        return content if content else ""
    except Exception as e:
        print("[LLM] Groq SDK call failed:", e)
        return "Groq API is not available right now."

# ---------------- Email Service ----------------
def _html_to_text(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()

def send_email(to_email: str, subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
    msg = MIMEMultipart("alternative")
    from_name, from_addr = parseaddr(config.SMTP_FROM)
    if not from_addr: from_addr = "no-reply@psychsupport.local"
    msg["From"] = formataddr((from_name or "Psych Support", from_addr))
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(text_body or _html_to_text(html_body), "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    context = ssl.create_default_context()
    try:
        if config.SMTP_SECURE == "ssl":
            server = smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=config.SMTP_TIMEOUT, context=context)
        else:
            server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=config.SMTP_TIMEOUT)
            server.ehlo()
            if config.SMTP_SECURE == "starttls":
                server.starttls(context=context)
                server.ehlo()
        if config.SMTP_USER:
            server.login(config.SMTP_USER, config.SMTP_PASSWORD or "")
        server.sendmail(from_addr, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[MAIL] send_email to={to_email} failed:", repr(e))
        raise

def users_without_any_attempts(db: Session) -> list[models.User]:
    return (
        db.query(models.User)
        .outerjoin(models.QuestionnaireAttempt, models.QuestionnaireAttempt.user_id == models.User.id)
        .group_by(models.User.id)
        .having(func.count(models.QuestionnaireAttempt.id) == 0)
        .all()
    )

def users_with_attempts_on_day(db: Session, day_str: str) -> list[models.User]:
    sub = db.query(models.QuestionnaireAttempt.user_id).filter(models.QuestionnaireAttempt.day == day_str).distinct().subquery()
    return db.query(models.User).filter(models.User.id.in_(sub)).all()

def attempts_for_user_on_day(db: Session, user_id: int, day_str: str) -> list[models.QuestionnaireAttempt]:
    return (
        db.query(models.QuestionnaireAttempt)
        .filter(models.QuestionnaireAttempt.user_id == user_id, models.QuestionnaireAttempt.day == day_str)
        .order_by(models.QuestionnaireAttempt.created_at.asc())
        .all()
    )

def attempts_for_user_all(db: Session, user_id: int) -> list[models.QuestionnaireAttempt]:
    return (
        db.query(models.QuestionnaireAttempt)
        .filter(models.QuestionnaireAttempt.user_id == user_id)
        .order_by(models.QuestionnaireAttempt.created_at.asc())
        .all()
    )

def compose_reminder_html(username: str) -> tuple[str, str]:
    subject = "Gentle reminder: Try today’s quick wellness check-in"
    html = f"""<div style="font-family:system-ui,Segoe UI,Arial,sans-serif"><h2>Hi {username},</h2><p>This is a quick nudge to try our <b>2–3 minute</b> mental wellness questionnaire.</p><ul><li>Track your mood & stress</li><li>Get personalized tips</li></ul><p>You can start it from your dashboard any time.</p><p style="color:#666">If you didn’t sign up for this, you can ignore this mail.</p></div>"""
    return subject, html

def compose_daily_summary_html(username: str, attempts: list[models.QuestionnaireAttempt]) -> tuple[str, str]:
    scores = [a.score for a in attempts if a.score is not None]
    avg = sum(scores) / len(scores) if scores else 0.0
    latest_comment = attempts[-1].comment if attempts and attempts[-1].comment else ""
    subject = "Your daily check-in summary"
    html = f"""<div style="font-family:system-ui,Segoe UI,Arial,sans-serif"><h2>Hi {username}, here’s your today’s summary</h2><p><b>Attempts today:</b> {len(attempts)}<br/><b>Average score:</b> {avg:.1f}/100</p>{"<p><b>Latest note:</b> " + latest_comment + "</p>" if latest_comment else ""}<p>Keep going—small, steady steps make a difference.</p></div>"""
    return subject, html

def compose_periodic_html(username: str, attempts: list[models.QuestionnaireAttempt]) -> tuple[str, str]:
    if not attempts: return ("Your check-in summary", "<div style='font-family:system-ui,Segoe UI,Arial,sans-serif'><p>No attempts yet.</p></div>")
    scores = [a.score for a in attempts if a.score is not None]
    avg_all = sum(scores)/len(scores) if scores else 0.0
    max_a = max(attempts, key=lambda a: (a.score or 0))
    min_a = min(attempts, key=lambda a: (a.score or 0))
    monthly: dict[str, list[models.QuestionnaireAttempt]] = {}
    for a in attempts:
        ym = (a.day or a.created_at.date().isoformat())[:7]
        monthly.setdefault(ym, []).append(a)
    monthly_html = []
    for ym, rows in sorted(monthly.items(), reverse=True):
        s = [r.score for r in rows if r.score is not None]
        avg_m = sum(s)/len(s) if s else 0.0
        scores_str = ", ".join(f"{(r.score or 0):.0f}" for r in rows)
        monthly_html.append(f"""<div style="margin-bottom:10px"><h4 style="margin:0">Month {ym}</h4><div>Attempts: {len(rows)} · Avg: {avg_m:.1f}</div><div>Scores: {scores_str}</div></div>""")
    subject = "Your check-in progress summary"
    html = f"""<div style="font-family:system-ui,Segoe UI,Arial,sans-serif"><h2>Hi {username}, here’s your progress</h2><p><b>Total attempts:</b> {len(attempts)}<br/><b>Average score:</b> {avg_all:.1f}/100</p><p><b>Highest:</b> {max_a.score:.1f} — {max_a.comment or "—"}</p><p><b>Lowest:</b> {min_a.score:.1f} — {min_a.comment or "—"}</p><hr/><h3 style="margin-top:10px">By month</h3>{''.join(monthly_html)}</div>"""
    return subject, html

def send_reminders_to_incomplete_users(db: Session) -> int:
    users = users_without_any_attempts(db)
    sent = 0
    for u in users:
        try:
            subject, html = compose_reminder_html(username_from_email(u.email))
            send_email(u.email, subject, html)
            sent += 1
        except Exception as e:
            print("[MAIL] reminder failed for", u.email, e)
    return sent

def send_daily_summaries_today(db: Session) -> int:
    today = date.today().isoformat()
    users = users_with_attempts_on_day(db, today)
    sent = 0
    for u in users:
        rows = attempts_for_user_on_day(db, u.id, today)
        if not rows: continue
        try:
            subject, html = compose_daily_summary_html(username_from_email(u.email), rows)
            send_email(u.email, subject, html)
            sent += 1
        except Exception as e:
            print("[MAIL] daily summary failed for", u.email, e)
    return sent

def send_periodic_summaries_all(db: Session) -> int:
    users = db.query(models.User).all()
    sent = 0
    for u in users:
        rows = attempts_for_user_all(db, u.id)
        if not rows: continue
        try:
            subject, html = compose_periodic_html(username_from_email(u.email), rows)
            send_email(u.email, subject, html)
            sent += 1
        except Exception as e:
            print("[MAIL] periodic summary failed for", u.email, e)
    return sent

# ---------------- Questionnaire Service ----------------
def is_continue_prompt_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return ("reply with 'continue' or 'score'" in t) or t.startswith("you've answered")

def is_regular_question_text(text: str) -> bool:
    if not text: return False
    t = text.strip()
    if not t.endswith("?"): return False
    return not is_continue_prompt_text(t)

def q_system_prompt() -> str:
    return (
        "You are a licensed-therapist-style assistant creating an adaptive mental-health questionnaire. "
        "Ask concise, empathetic questions. Use the user's previous answers to decide the next question. "
        "Do not give scores until asked. Keep each question short and clear."
    )

def q_history(db: Session, user_id: int, session_id: str) -> List[Dict[str, str]]:
    msgs = db.query(models.Message).filter(models.Message.conversation_id == session_id, models.Message.user_id == user_id).order_by(models.Message.created_at.asc()).all()
    return [{"role": m.role, "content": m.content} for m in msgs]

def _extract_qa_transcript(history_pairs: List[Dict[str, str]]) -> List[str]:
    t = []
    for m in history_pairs:
        t.append(f"Q: {m['content'].strip()}" if m["role"] == "assistant" else f"A: {m['content']}")
    return t

def next_question_from_llm(history_pairs: List[Dict[str, str]]) -> str:
    transcript = "\n".join(_extract_qa_transcript(history_pairs))
    prompt = (
        q_system_prompt() +
        "\n\nTRANSCRIPT:\n" + transcript +
        "\n\nProduce ONLY the next question (no preface, no JSON, no commentary)."
    )
    text = call_groq(prompt, num_predict=120) if _groq_client else call_ollama(prompt, num_predict=120)
    q = (text or "").strip().split("\n")[0].strip()
    if q and not q.endswith("?"): q += "?"
    return q or "Can you tell me more about that?"

def finalize_score_from_history(history_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    transcript = "\n".join(_extract_qa_transcript(history_pairs))
    prompt = (
        "Analyze the conversation transcript and output JSON ONLY with keys:\n"
        "{\"score\": <0-100>, \"comment\": \"...\", \"tips\": [\"...\", \"...\"]}\n"
        "Be concise, empathetic, and actionable.\n\nTRANSCRIPT:\n" + transcript + "\n\nJSON:"
    )
    text = call_groq(prompt, num_predict=400) if _groq_client else call_ollama(prompt, num_predict=400)
    try:
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
            data["score"] = float(data.get("score", 0.0))
            if not isinstance(data.get("tips"), list): data["tips"] = []
            return data
    except Exception: pass
    return {"score": 50.0, "comment": "Neutral mood with mixed stress signals.", "tips": ["Take a short walk", "Practice 5-minute breathing"]}

# ---------------- General Helpers & RAG Formatting ----------------
_YT_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_\-]{6,})",
    r"(?:https?://)?youtu\.be/([A-Za-z0-9_\-]{6,})",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([A-Za-z0-9_\-]{6,})",
]

def short_title(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t: return "New Chat"
    words = t.split()
    s = " ".join(words[:8])
    return s[:80] + ("…" if len(words) > 8 else "")

def username_from_email(email: str) -> str:
    return (email or "").split("@", 1)[0] or "there"

def normalize_youtube_url(url: str) -> str:
    url = (url or "").strip()
    for pat in _YT_PATTERNS:
        m = re.match(pat, url)
        if m:
            vid = m.group(1)
            return f"https://www.youtube.com/watch?v={vid}"
    if "youtube.com" in url or "youtu.be" in url:
        return url
    raise HTTPException(status_code=400, detail="Only YouTube links are supported for now.")

def _tags_to_str(tags: Optional[List[str]]) -> str:
    if not tags: return ""
    return ",".join(t.strip() for t in tags if t and t.strip())

def _str_to_tags(s: Optional[str]) -> List[str]:
    if not s: return []
    return [t.strip() for t in s.split(",") if t.strip()]

def is_out_of_context(contexts: List[dict]) -> bool:
    best_score = max((c.get("score", 0.0) for c in contexts), default=0.0)
    return (best_score < config.OUT_OF_CONTEXT_THRESHOLD) or (not contexts)

def _number_contexts(contexts: List[dict]) -> List[dict]:
    ordered = []
    for i, c in enumerate(contexts[:config.TOP_K], start=1): # 1-based index for prompts
        cc = dict(c)
        cc["index"] = i
        ordered.append(cc)
    return ordered

def build_prompt_with_numbered_context(user_msg: str, contexts: List[dict]) -> tuple[str, List[dict], bool]:
    system_prompt = "You are a compassionate, knowledgeable mental wellness assistant. Prefer concise, practical guidance. When you use information from a provided source, append the citation marker like [1], [2] exactly matching the numbered CONTEXT SOURCES below. Do not invent citation numbers. If a statement is general knowledge, do not cite it."
    numbered = _number_contexts(contexts)
    ooc = is_out_of_context(contexts)

    if ooc:
        prompt = (f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"
                  f"Begin your response with EXACTLY this line followed by a blank line:\n"
                  f"\"{config.OOC_PREFIX}\"\n\nThen provide a clear, supportive answer based on general knowledge only. "
                  "Do not add any [n] citations because no uploaded context is being used.")
        return prompt, numbered, True

    blocks = []
    for c in numbered:
        meta = c.get("meta") or {}
        src = meta.get("filename") or meta.get("uploader") or c.get("id")
        text = (c.get("text") or "")[:800]
        score = float(c.get("score") or 0.0)
        blocks.append(f"[{c['index']}] filename={src} score={score:.4f}\n{text}")
    ctx = "\n\n".join(blocks)

    prompt = (f"SYSTEM:\n{system_prompt}\n\nCONTEXT SOURCES (numbered):\n{ctx}\n\nUSER:\n{user_msg}\n\n"
              "ASSISTANT:\nUse the numbered sources above where appropriate; append [n] after the relevant sentence. "
              "Only cite numbers that exist in the context. If a statement is not supported by the context, don't cite.")
    return prompt, numbered, False

def append_cited_sources_marker(reply_text: str, numbered_contexts: List[dict]) -> str:
    if not reply_text or not numbered_contexts: return reply_text
    pat_num = re.compile(r"\[(\d+)\]")
    seen: List[int] = []
    for m in pat_num.finditer(reply_text):
        i = int(m.group(1))
        if i not in seen: seen.append(i)
    if not seen: return reply_text

    by_index = {c["index"]: c for c in numbered_contexts}
    items = []
    for idx in seen:
        c = by_index.get(idx)
        if not c: continue
        text = (c.get("text") or "").strip().replace("\n", " ")
        excerpt = text[:280] + ("…" if len(text) > 280 else "")
        meta = c.get("meta") or {}
        fname = meta.get("filename") or meta.get("uploader") or c.get("id")
        items.append({"index": idx, "excerpt": excerpt, "filename": fname, "id": c.get("id")})
    if not items: return reply_text
    payload = json.dumps(items, ensure_ascii=False)
    return f"{reply_text}\n\n{config.SOURCES_MARKER_PREFIX}{payload}{config.SOURCES_MARKER_SUFFIX}"

def split_reply_and_sources_for_api(text: str) -> tuple[str, List[dict]]:
    if not text: return text, []
    start = text.rfind(config.SOURCES_MARKER_PREFIX)
    if start == -1: return text, []
    end = text.find(config.SOURCES_MARKER_SUFFIX, start)
    if end == -1: return text, []
    try:
        payload = text[start + len(config.SOURCES_MARKER_PREFIX): end]
        items = json.loads(payload)
        clean = text[:start].rstrip()
        return clean, items if isinstance(items, list) else []
    except Exception:
        return text, []