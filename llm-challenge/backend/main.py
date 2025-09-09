import os
import uuid
import json
from datetime import datetime, timedelta, date
from typing import List, Optional, Literal, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, func, desc
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
import jwt
import requests
import traceback

# RAG libs
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# Groq SDK
from groq import Groq

# ---------------- CONFIG ----------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "240"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./psych_support.db")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "psych_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
OUT_OF_CONTEXT_THRESHOLD = float(os.getenv("OUT_OF_CONTEXT_THRESHOLD", "0.22"))

# Providers
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_groq_client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# Questionnaire rules
MIN_QUESTIONS = 10           # minimum to reach before offering score
CONTINUE_STEP = 3            # increment each time user says "continue"

# ---------------- DB ----------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role = Column(String(16))  # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id = Column(String(36), primary_key=True, index=True)
    uploader = Column(String(255))
    filename = Column(String(255))
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class QuestionnaireAttempt(Base):
    __tablename__ = "q_attempts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    day = Column(String(10), index=True)  # yyyy-mm-dd
    score = Column(Float)
    comment = Column(Text)
    tips_json = Column(Text)  # JSON list of strings
    session_id = Column(String(36))  # link to its conversation (optional)
    created_at = Column(DateTime, default=datetime.utcnow)


# NEW: QuestionnaireSession to manage target question count and active flag
class QuestionnaireSession(Base):
    __tablename__ = "q_sessions"
    session_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    target_questions = Column(Integer, default=MIN_QUESTIONS)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = closed
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- Auth utils ----------------
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()


def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    return pwd_ctx.verify(pw, hashed)


def create_access_token(sub: str, minutes: int = JWT_EXPIRE_MINUTES) -> str:
    exp = datetime.utcnow() + timedelta(minutes=minutes)
    return jwt.encode({"sub": sub, "exp": exp}, SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)
) -> User:
    token = creds.credentials
    email = decode_token(token)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------------- RAG ----------------
_qdrant: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None


def init_embedder() -> Optional[SentenceTransformer]:
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("[RAG] Embedder loaded:", EMBED_MODEL)
        return _embedder
    except Exception as e:
        print("[RAG] Embedder load failed:", e)
        _embedder = None
        return None


def init_qdrant() -> Optional[QdrantClient]:
    global _qdrant
    if _qdrant is not None:
        return _qdrant
    try:
        _qdrant = QdrantClient(url=QDRANT_URL)
        try:
            _qdrant.get_collection(COLLECTION)
        except Exception:
            emb = init_embedder()
            if emb is None:
                raise RuntimeError("Cannot create collection because embedder failed")
            vec_size = emb.get_sentence_embedding_dimension()
            _qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=qmodels.VectorParams(size=vec_size, distance=qmodels.Distance.COSINE),
            )
        print("[RAG] Qdrant connected:", QDRANT_URL)
        return _qdrant
    except Exception as e:
        print("[RAG] Qdrant init failed:", e)
        _qdrant = None
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    emb = init_embedder()
    if emb is None:
        raise RuntimeError("Embedder not available")
    vectors = emb.encode(texts, normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def upsert_documents_to_qdrant(payloads: List[dict]) -> int:
    qc = init_qdrant()
    if qc is None:
        raise RuntimeError("Qdrant not available")
    vectors = embed_texts([p["text"] for p in payloads])
    points = []
    for i, p in enumerate(payloads):
        points.append(
            qmodels.PointStruct(
                id=p["id"],
                vector=vectors[i],
                payload={"text": p["text"], "meta": p.get("meta", {})},
            )
        )
    qc.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search_qdrant(query: str, limit: int = TOP_K) -> List[dict]:
    qc = init_qdrant()
    if qc is None:
        return []
    qvec = embed_texts([query])[0]
    hits = qc.search(collection_name=COLLECTION, query_vector=qvec, limit=limit)
    results = []
    for h in hits:
        results.append({
            "id": str(h.id),
            "text": h.payload.get("text", ""),
            "meta": h.payload.get("meta", {}),
            "score": float(h.score) if h.score is not None else 0.0,
        })
    return results


# ---------------- helpers: sources as excerpts ----------------
SOURCES_MARKER_PREFIX = "<!--SOURCES_JSON:"
SOURCES_MARKER_SUFFIX = "-->"

def build_source_items(contexts: List[dict]) -> List[dict]:
    items = []
    for c in contexts:
        meta = c.get("meta") or {}
        fname = meta.get("filename") or meta.get("uploader") or c.get("id")
        text = (c.get("text") or "").strip().replace("\n", " ")
        excerpt = text[:280] + ("…" if len(text) > 280 else "")
        items.append({"filename": fname, "excerpt": excerpt})
    return items

def append_sources_marker(reply_text: str, contexts: List[dict]) -> str:
    items = build_source_items(contexts)
    if not items:
        return reply_text
    payload = json.dumps(items, ensure_ascii=False)
    return f"{reply_text}\n\n{SOURCES_MARKER_PREFIX}{payload}{SOURCES_MARKER_SUFFIX}"


# ---------------- LLMs ----------------
def call_ollama(prompt: str, num_predict: int = 512) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": num_predict}},
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


def call_groq(prompt: str, num_predict: int = 512) -> str:
    if not _groq_client:
        return "Groq API key not configured. Add GROQ_API_KEY to .env."
    try:
        chat_completion = _groq_client.chat.completions.create(
            model=GROQ_MODEL_DEFAULT,
            messages=[
                {"role": "system", "content": "You are a helpful, compassionate assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=num_predict,
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print("[LLM] Groq SDK call failed:", e)
        return "Groq API is not available right now."


# ---------------- FastAPI ----------------
app = FastAPI(title="Psych Support - RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Schemas ----------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None
    num_predict: Optional[int] = 512
    provider: Literal["mistral", "groq"] = "mistral"

class ChatOut(BaseModel):
    reply: str
    sources: List[dict] = Field(default_factory=list)

class UpdateTitleIn(BaseModel):
    title: str

# Questionnaire
class QStartOut(BaseModel):
    session_id: str
    question: str

class QAnswerIn(BaseModel):
    session_id: str
    answer: str

class QAnswerOut(BaseModel):
    done: bool
    question: Optional[str] = None
    score: Optional[float] = None
    comment: Optional[str] = None
    tips: List[str] = Field(default_factory=list)


# ---------------- helpers: titles, username ----------------
def short_title(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t: return "New Chat"
    words = t.split()
    s = " ".join(words[:8])
    if len(words) > 8: s += "…"
    return s[:80]

def username_from_email(email: str) -> str:
    return (email or "").split("@", 1)[0] or "there"


# ---------------- Routes: meta ----------------
@app.get("/health")
def health(user: User = Depends(get_current_user)):
    return {
        "ok": True,
        "qdrant": init_qdrant() is not None,
        "embedder": init_embedder() is not None,
        "me": {"email": user.email, "username": username_from_email(user.email)},
        "provider_defaults": {"mistral": OLLAMA_MODEL, "groq": GROQ_MODEL_DEFAULT if GROQ_API_KEY else None},
    }


# ---------------- Auth ----------------
@app.post("/auth/register")
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered. Please log in instead.")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

# Aliases
@app.post("/register")
def register_alias(payload: RegisterIn, db: Session = Depends(get_db)):
    return register(payload, db)
@app.post("/login")
def login_alias(payload: LoginIn, db: Session = Depends(get_db)):
    return login(payload, db)


# ---------------- Sessions/Chats ----------------
@app.get("/sessions")
def list_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat()} for r in rows]

@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == session_id, Message.user_id == user.id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in msgs]

@app.patch("/sessions/{session_id}")
def update_session_title(session_id: str, payload: UpdateTitleIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == session_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")
    conv.title = payload.title[:255]
    db.commit()
    return {"id": conv.id, "title": conv.title}


# ---------------- Ingest ----------------
@app.post("/ingest")
def ingest(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt supported.")
    raw = file.file.read().decode("utf-8", errors="ignore")
    chunks = [raw[i:i+800].strip() for i in range(0, len(raw), 800) if raw[i:i+800].strip()]
    payloads = [{"id": str(uuid.uuid4()), "text": c, "meta": {"uploader": user.email, "filename": file.filename}} for c in chunks]
    db.add(Document(id=str(uuid.uuid4()), uploader=user.email, filename=file.filename)); db.commit()
    count = upsert_documents_to_qdrant(payloads)
    return {"ok": True, "ingested_chunks": count}


# ---------------- Chat (no session created until first message) ----------------
def build_prompt(body_message: str, contexts: List[dict]) -> str:
    system_prompt = (
        "You are a compassionate, knowledgeable mental wellness assistant. Give well-structured, practical, "
        "non-clinical advice. If content refers to documented sources, cite them as [source1], [source2], etc. "
        "If user is in crisis, advise contacting local emergency services."
    )
    best_score = max((c["score"] for c in contexts), default=0.0)
    is_out = best_score < OUT_OF_CONTEXT_THRESHOLD

    if is_out or not contexts:
        return (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{body_message}\n\n"
            "ASSISTANT:\nProvide a clear, supportive answer. "
            "Note: This answer is based on general knowledge and not from your uploaded documents."
        )
    MAX_CTX = 2000
    used = 0; blocks = []
    for i, c in enumerate(contexts[:TOP_K], start=1):
        meta = c.get("meta", {})
        src = meta.get("filename") or meta.get("uploader") or c.get("id")
        text = (c["text"] or "")[:800]
        block = f"[source{i}] filename={src} score={c['score']:.4f}\n{text}"
        if used + len(block) > MAX_CTX: break
        blocks.append(block); used += len(block)
    ctx = "\n\n".join(blocks)
    return (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"USER:\n{body_message}\n\n"
        "ASSISTANT:\nUse the context above; cite like [source1]."
    )

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # RAG search
    try:
        contexts = search_qdrant(body.message, limit=TOP_K)
    except Exception as e:
        print("[RAG] search error:", e, traceback.format_exc())
        contexts = []

    prompt = build_prompt(body.message, contexts)
    max_toks = body.num_predict or 512
    reply = call_groq(prompt, num_predict=max_toks) if body.provider == "groq" else call_ollama(prompt, num_predict=max_toks)
    reply_with_marker = append_sources_marker(reply, contexts)

    # Create a conversation ONLY NOW (first message), if missing
    conv_id = body.session_id
    if not conv_id:
        conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title=short_title(body.message))
        db.add(conv); db.commit(); conv_id = conv.id

    # save
    try:
        db.add_all([
            Message(conversation_id=conv_id, user_id=user.id, role="user", content=body.message),
            Message(conversation_id=conv_id, user_id=user.id, role="assistant", content=reply_with_marker),
        ])
        db.commit()
    except Exception as e:
        print("[DB] save failed:", e, traceback.format_exc())

    return {"reply": reply_with_marker, "sources": build_source_items(contexts)}


@app.post("/chat_stream")
def chat_stream(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    def gen():
        # RAG search
        try:
            contexts = search_qdrant(body.message, limit=TOP_K)
        except Exception as e:
            print("[RAG] search error:", e, traceback.format_exc())
            contexts = []

        prompt = build_prompt(body.message, contexts)
        max_toks = body.num_predict or 512

        chunks: List[str] = []
        try:
            if body.provider == "groq":
                if not _groq_client:
                    yield json.dumps({"type":"delta","text":"[Groq key missing]\n"}) + "\n"
                else:
                    stream = _groq_client.chat.completions.create(
                        model=GROQ_MODEL_DEFAULT,
                        messages=[{"role":"system","content":"You are a helpful, compassionate assistant."},{"role":"user","content":prompt}],
                        max_tokens=max_toks, temperature=0.2, stream=True,
                    )
                    for chunk in stream:
                        d = ""
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            d = chunk.choices[0].delta.content
                        if d:
                            chunks.append(d)
                            yield json.dumps({"type":"delta","text": d}) + "\n"
            else:
                with requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "options": {"num_predict": max_toks}},
                    stream=True, timeout=600,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue
                        try: ev = json.loads(line)
                        except Exception: continue
                        d = ev.get("response")
                        if d:
                            chunks.append(d)
                            yield json.dumps({"type":"delta","text": d}) + "\n"
        except requests.exceptions.Timeout:
            yield json.dumps({"type":"delta","text":"[Provider timed out]\n"}) + "\n"
        finally:
            final_text = "".join(chunks)
            final_text_with_marker = append_sources_marker(final_text, contexts)

            # Create conversation only now (first message), if missing
            conv_id = body.session_id
            if not conv_id:
                conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title=short_title(body.message))
                db.add(conv); db.commit(); conv_id = conv.id

            try:
                db.add_all([
                    Message(conversation_id=conv_id, user_id=user.id, role="user", content=body.message),
                    Message(conversation_id=conv_id, user_id=user.id, role="assistant", content=final_text_with_marker),
                ])
                db.commit()
            except Exception as e:
                print("[DB] save failed (stream):", e, traceback.format_exc())

            yield json.dumps({"type":"delta","text": final_text_with_marker[len(final_text):]}) + "\n"
            yield json.dumps({"type":"done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/jsonl")


# ---------------- Questionnaire logic ----------------
Q_MARK = "<!--Q-->"                    # tag assistant questions
CONTINUE_PROMPT_MARK = "<!--CONTINUE_PROMPT-->"  # tag "continue or score?" prompt

def q_system_prompt() -> str:
    return (
        "You are a licensed-therapist-style assistant creating an adaptive mental-health questionnaire. "
        "Ask concise, empathetic questions. Use the user's previous answers to decide the next question. "
        "Do not give scores until asked. Keep each question short and clear."
    )

def q_history(db: Session, user_id: int, session_id: str) -> List[Dict[str, str]]:
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == session_id, Message.user_id == user_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in msgs]

def _extract_qa_transcript(history_pairs: List[Dict[str, str]]) -> List[str]:
    t = []
    for m in history_pairs:
        if m["role"] == "assistant":
            # Strip markers if present
            c = m["content"].replace(Q_MARK, "").replace(CONTINUE_PROMPT_MARK, "").strip()
            t.append(f"Q: {c}")
        else:
            t.append(f"A: {m['content']}")
    return t

def next_question_from_llm(history_pairs: List[Dict[str, str]]) -> str:
    transcript = "\n".join(_extract_qa_transcript(history_pairs))
    prompt = (
        q_system_prompt() +
        "\n\nTRANSCRIPT:\n" + transcript +
        "\n\nProduce ONLY the next question (no preface, no JSON, no commentary)."
    )
    text = call_groq(prompt, num_predict=120) if _groq_client else call_ollama(prompt, num_predict=120)
    # take first line as question
    q = (text or "").strip().split("\n")[0].strip()
    if not q.endswith("?"): q += "?"
    return q

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
            if not isinstance(data.get("tips"), list):
                data["tips"] = []
            return data
    except Exception:
        pass
    return {"score": 50.0, "comment": "Neutral mood with mixed stress signals.", "tips": ["Take a short walk", "Practice 5-minute breathing"]}


@app.post("/questionnaire/start", response_model=QStartOut)
def questionnaire_start(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Create a new conversation
    conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title=f"[Questionnaire] {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    db.add(conv); db.commit()
    # Create q_session row with target = MIN_QUESTIONS
    qs = QuestionnaireSession(session_id=conv.id, user_id=user.id, target_questions=MIN_QUESTIONS, is_active=1)
    db.add(qs); db.commit()
    # First question
    first_q = "To get started, how would you describe your overall mood today?" + Q_MARK
    db.add(Message(conversation_id=conv.id, user_id=user.id, role="assistant", content=first_q))
    db.commit()
    return {"session_id": conv.id, "question": first_q.replace(Q_MARK, "")}


@app.post("/questionnaire/answer", response_model=QAnswerOut)
def questionnaire_answer(body: QAnswerIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Append user answer
    db.add(Message(conversation_id=body.session_id, user_id=user.id, role="user", content=body.answer))
    db.commit()

    # Fetch session settings
    qs = db.query(QuestionnaireSession).filter(
        QuestionnaireSession.session_id == body.session_id,
        QuestionnaireSession.user_id == user.id
    ).first()
    if not qs or not qs.is_active:
        raise HTTPException(status_code=400, detail="Questionnaire session not active.")

    # Load history
    hist = q_history(db, user.id, body.session_id)

    # Determine if the latest answer was to a CONTINUE prompt
    # Find the last assistant message before this user answer
    last_assistant = None
    for m in reversed(hist[:-1]):  # exclude the just-added user message
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    # Count "actual" questions asked so far (assistant messages with Q_MARK)
    actual_q_count = sum(1 for m in hist if m["role"] == "assistant" and Q_MARK in m["content"])

    # If last assistant was the continue prompt, interpret the user's answer:
    if last_assistant and CONTINUE_PROMPT_MARK in last_assistant:
        ans = (body.answer or "").strip().lower()
        wants_continue = any(x in ans for x in ["continue", "yes", "y", "go on", "more", "add"])
        if wants_continue:
            # bump target by 3 and ask next adaptive question
            qs.target_questions = (qs.target_questions or MIN_QUESTIONS) + CONTINUE_STEP
            db.commit()
            q = next_question_from_llm(hist) + Q_MARK
            db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=q))
            db.commit()
            return {"done": False, "question": q.replace(Q_MARK, "")}
        else:
            # finalize now
            data = finalize_score_from_history(hist)
            score = float(data.get("score", 0.0))
            comment = data.get("comment") or ""
            tips = data.get("tips") or []
            summary_text = f"**Summary**\nScore: {score:.1f}/100\n\n{comment}\n\nTips:\n" + "\n".join([f"- {t}" for t in tips])
            db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=summary_text))
            # save attempt
            today = date.today().isoformat()
            db.add(QuestionnaireAttempt(
                user_id=user.id, day=today, score=score, comment=comment, tips_json=json.dumps(tips), session_id=body.session_id
            ))
            qs.is_active = 0
            db.commit()
            return {"done": True, "score": score, "comment": comment, "tips": tips}

    # If we reached at least target questions -> ask for continue or score
    if actual_q_count >= (qs.target_questions or MIN_QUESTIONS):
        prompt = (
            f"You've answered {actual_q_count} questions so far. "
            f"Would you like to **continue** with {CONTINUE_STEP} more questions, or **get your score** now? "
            "Reply with 'continue' or 'score'."
        ) + CONTINUE_PROMPT_MARK
        db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=prompt))
        db.commit()
        return {"done": False, "question": "continue or score?"}

    # Otherwise: ask next adaptive question
    q = next_question_from_llm(hist) + Q_MARK
    db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=q))
    db.commit()
    return {"done": False, "question": q.replace(Q_MARK, "")}


# ---------------- Questionnaire dashboard (fast) ----------------
class QDashItem(BaseModel):
    day: str
    attempts: int
    avg_score: float
    latest_comment: Optional[str] = None

@app.get("/questionnaire/dashboard", response_model=List[QDashItem])
def questionnaire_dashboard(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # AVG + COUNT per day
    agg_rows = (
        db.query(
            QuestionnaireAttempt.day.label("day"),
            func.count(QuestionnaireAttempt.id).label("attempts"),
            func.avg(QuestionnaireAttempt.score).label("avg_score"),
        )
        .filter(QuestionnaireAttempt.user_id == user.id)
        .group_by(QuestionnaireAttempt.day)
        .order_by(desc(QuestionnaireAttempt.day))
        .all()
    )

    # Latest comment per day (using max created_at)
    sub_latest = (
        db.query(
            QuestionnaireAttempt.day.label("day"),
            func.max(QuestionnaireAttempt.created_at).label("max_ts"),
        )
        .filter(QuestionnaireAttempt.user_id == user.id)
        .group_by(QuestionnaireAttempt.day)
        .subquery()
    )

    latest_rows = (
        db.query(QuestionnaireAttempt.day, QuestionnaireAttempt.comment)
        .join(sub_latest, (QuestionnaireAttempt.day == sub_latest.c.day) & (QuestionnaireAttempt.created_at == sub_latest.c.max_ts))
        .filter(QuestionnaireAttempt.user_id == user.id)
        .all()
    )
    latest_map = {d: c for d, c in latest_rows}

    out = []
    for r in agg_rows:
        out.append({
            "day": r.day,
            "attempts": int(r.attempts or 0),
            "avg_score": round(float(r.avg_score or 0.0), 2),
            "latest_comment": latest_map.get(r.day, None),
        })
    return out