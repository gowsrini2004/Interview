import os
import uuid
import json
import re
from datetime import datetime, timedelta, date
from typing import List, Optional, Literal, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Body, Path
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
MIN_QUESTIONS = 7          # minimum to reach before offering score
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


# Admin principal helper
class AdminPrincipal:
    id = -1
    email = "admin"
    is_admin = True


def is_admin_principal(obj: Any) -> bool:
    return getattr(obj, "email", None) == "admin" or getattr(obj, "is_admin", False) is True


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)
):
    token = creds.credentials
    sub = decode_token(token)
    # Hardcoded admin principal
    if sub == "admin":
        return AdminPrincipal()
    # Normal user
    user = db.query(User).filter(User.email == sub).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_admin(user=Depends(get_current_user)):
    if is_admin_principal(user):
        return user
    raise HTTPException(status_code=403, detail="Admin only")


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
        # ✅ Optional but recommended: index user_id for fast filtering
        try:
            _qdrant.create_payload_index(
                collection_name=COLLECTION,
                field_name="user_id",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
        except Exception:
            # index may already exist; ignore
            pass

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
                payload={
                    "text": p["text"],
                    "meta": p.get("meta", {}),
                    "user_id": p.get("user_id")  # ✅ add user_id here
                },
            )
        )
    qc.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search_qdrant(query: str, limit: int = TOP_K, user_id: Optional[int] = None) -> List[dict]:
    qc = init_qdrant()
    if qc is None:
        return []
    qvec = embed_texts([query])[0]

    q_filter = None
    if user_id is not None:
        q_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
        )

    # Depending on qdrant-client version, the kwarg is either `query_filter` or `filter`.
    # Most recent clients use `query_filter=`.
    hits = qc.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=limit,
        query_filter=q_filter,  # <-- if this errors, change to `filter=q_filter`
    )
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
    """(Legacy helper) Not used for rendering anymore; kept for compatibility."""
    items = []
    for c in contexts:
        meta = c.get("meta") or {}
        fname = meta.get("filename") or meta.get("uploader") or c.get("id")
        text = (c.get("text") or "").strip().replace("\n", " ")
        excerpt = text[:280] + ("…" if len(text) > 280 else "")
        items.append({"filename": fname, "excerpt": excerpt})
    return items

# -------------- NEW helpers for numbered citations & OOC prefix --------------
OOC_PREFIX = "🌐 I couldn’t find context in your uploaded sources — this answer is drawn from general knowledge!!"


def is_out_of_context(contexts: List[dict]) -> bool:
    best_score = max((c.get("score", 0.0) for c in contexts), default=0.0)
    return (best_score < OUT_OF_CONTEXT_THRESHOLD) or (not contexts)

def _number_contexts(contexts: List[dict]) -> List[dict]:
    """Return contexts[:TOP_K] with an added 1-based 'index' key."""
    ordered = []
    for i, c in enumerate(contexts[:TOP_K], start=0):
        cc = dict(c)
        cc["index"] = i
        ordered.append(cc)
    return ordered

def build_prompt_with_numbered_context(user_msg: str, contexts: List[dict]) -> tuple[str, List[dict], bool]:
    """
    Returns: (prompt, numbered_contexts, out_of_context_flag)
    - numbered_contexts: contexts[:TOP_K] with .index
    - prompt instructs the model to cite using [n] and to emit OOC_PREFIX if out-of-context
    """
    system_prompt = (
        "You are a compassionate, knowledgeable mental wellness assistant. "
        "Prefer concise, practical guidance. "
        "When you use information from a provided source, append the citation marker like [1], [2] "
        "exactly matching the numbered CONTEXT SOURCES below. "
        "Do not invent citation numbers. If a statement is general knowledge, do not cite it."
    )

    numbered = _number_contexts(contexts)
    ooc = is_out_of_context(contexts)

    if ooc:
        # Explicitly instruct the first line with emoji
        prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{user_msg}\n\n"
            "ASSISTANT:\n"
            f"Begin your response with EXACTLY this line followed by a blank line:\n"
            f"\"{OOC_PREFIX}\"\n\n"
            "Then provide a clear, supportive answer based on general knowledge only. "
            "Do not add any [n] citations because no uploaded context is being used."
        )
        return prompt, numbered, True

    # Build a compact numbered context block
    blocks = []
    for c in numbered:
        meta = c.get("meta") or {}
        src = meta.get("filename") or meta.get("uploader") or c.get("id")
        text = (c.get("text") or "")[:800]
        score = float(c.get("score") or 0.0)
        blocks.append(f"[{c['index']}] filename={src} score={score:.4f}\n{text}")
    ctx = "\n\n".join(blocks)

    prompt = (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXT SOURCES (numbered):\n{ctx}\n\n"
        f"USER:\n{user_msg}\n\n"
        "ASSISTANT:\nUse the numbered sources above where appropriate; append [n] after the relevant sentence. "
        "Only cite numbers that exist in the context. If a statement is not supported by the context, don't cite."
    )
    return prompt, numbered, False


def append_cited_sources_marker(reply_text: str, numbered_contexts: List[dict]) -> str:
    """
    Scan reply for [n] citations and attach ONLY those sources as JSON in the marker.
    Keeps order of first appearance. Each item: {index, excerpt, filename, id}
    """
    if not reply_text or not numbered_contexts:
        return reply_text

    pat_num = re.compile(r"\[(\d+)\]")
    pat_legacy = re.compile(r"\[source(\d+)\]", re.IGNORECASE)

    seen: List[int] = []
    for m in pat_num.finditer(reply_text):
        i = int(m.group(1))
        if i not in seen:
            seen.append(i)
    for m in pat_legacy.finditer(reply_text):
        i = int(m.group(1))
        if i not in seen:
            seen.append(i)

    if not seen:
        return reply_text

    by_index = {c["index"]: c for c in numbered_contexts}
    items = []
    for idx in seen:
        c = by_index.get(idx)
        if not c:
            continue
        text = (c.get("text") or "").strip().replace("\n", " ")
        excerpt = text[:280] + ("…" if len(text) > 280 else "")
        meta = c.get("meta") or {}
        fname = meta.get("filename") or meta.get("uploader") or c.get("id")
        items.append({
            "index": idx,
            "excerpt": excerpt,
            "filename": fname,
            "id": c.get("id")
        })

    if not items:
        return reply_text

    payload = json.dumps(items, ensure_ascii=False)
    return f"{reply_text}\n\n{SOURCES_MARKER_PREFIX}{payload}{SOURCES_MARKER_SUFFIX}"


def split_reply_and_sources_for_api(text: str) -> tuple[str, List[dict]]:
    """Split assistant reply and extract JSON source items from marker."""
    if not text:
        return text, []
    start = text.rfind(SOURCES_MARKER_PREFIX)
    if start == -1:
        return text, []
    end = text.find(SOURCES_MARKER_SUFFIX, start)
    if end == -1:
        return text, []
    try:
        payload = text[start + len(SOURCES_MARKER_PREFIX): end]
        items = json.loads(payload)
        clean = text[:start].rstrip()
        return clean, items if isinstance(items, list) else []
    except Exception:
        return text, []


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
    allow_origins=CORS_ALLOWED_ORIGINS,   # ["*"]
    allow_credentials=False,              # <- change this if using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Schemas ----------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str

# CHANGED: allow username OR email for login
class LoginIn(BaseModel):
    identifier: str  # email OR "admin"
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
def delete_user_vectors(user_id: int) -> None:
    qc = init_qdrant()
    if not qc:
        return
    try:
        qc.delete(
            collection_name=COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
                )
            ),
            wait=True,
        )
        print(f"[Qdrant] Deleted vectors for user_id={user_id}")
    except Exception as e:
        print("[Qdrant] delete_user_vectors failed:", e)



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
def health(user=Depends(get_current_user)):
    return {
        "ok": True,
        "qdrant": init_qdrant() is not None,
        "embedder": init_embedder() is not None,
        "me": {
            "email": user.email if not is_admin_principal(user) else "admin",
            "username": "admin" if is_admin_principal(user) else username_from_email(user.email),
            "is_admin": is_admin_principal(user),
        },
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
    return {"access_token": token, "token_type": "bearer", "is_admin": False}

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    # Hardcoded admin login
    if payload.identifier == "admin" and payload.password == "admin":
        token = create_access_token("admin")
        return {"access_token": token, "token_type": "bearer", "is_admin": True}

    # Normal user flow
    user = db.query(User).filter(User.email == payload.identifier).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email/username or password")
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer", "is_admin": False}

# Aliases
@app.post("/register")
def register_alias(payload: RegisterIn, db: Session = Depends(get_db)):
    return register(payload, db)

@app.post("/login")
def login_alias(payload: LoginIn, db: Session = Depends(get_db)):
    return login(payload, db)


# ---------------- Sessions/Chats ----------------
@app.get("/sessions")
def list_sessions(user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Admin has no personal sessions; return empty list
    if is_admin_principal(user):
        return []
    rows = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat()} for r in rows]

@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admins don't have chat history.")
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == session_id, Message.user_id == user.id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in msgs]

@app.patch("/sessions/{session_id}")
def update_session_title(session_id: str, payload: UpdateTitleIn, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admins cannot rename user sessions.")
    conv = db.query(Conversation).filter(Conversation.id == session_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")
    conv.title = payload.title[:255]
    db.commit()
    return {"id": conv.id, "title": conv.title}


# ---------------- Ingest ----------------
@app.post("/ingest")
def ingest(file: UploadFile = File(...), user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admin cannot ingest documents.")
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt supported.")
    raw = file.file.read().decode("utf-8", errors="ignore")
    chunks = [raw[i:i+800].strip() for i in range(0, len(raw), 800) if raw[i:i+800].strip()]
    payloads = [{
        "id": str(uuid.uuid4()),
        "text": c,
        "meta": {"uploader": user.email, "filename": file.filename},
        "user_id": user.id,  # ✅ tag vectors with user_id
    } for c in chunks]
    db.add(Document(id=str(uuid.uuid4()), uploader=user.email, filename=file.filename)); db.commit()
    count = upsert_documents_to_qdrant(payloads)
    return {"ok": True, "ingested_chunks": count}


# ---------------- Chat (updated: numbered citations + emoji OOC prefix) ----------------
@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admin cannot chat.")
    # RAG search
    try:
        contexts = search_qdrant(body.message, limit=TOP_K, user_id=user.id)
    except Exception as e:
        print("[RAG] search error:", e, traceback.format_exc())
        contexts = []

    prompt, numbered_ctx, ooc = build_prompt_with_numbered_context(body.message, contexts)
    max_toks = body.num_predict or 512
    raw_reply = call_groq(prompt, num_predict=max_toks) if body.provider == "groq" else call_ollama(prompt, num_predict=max_toks)

    # Guarantee first line for out-of-context
    reply = (raw_reply or "").strip()
    if ooc and not reply.startswith(OOC_PREFIX):
        reply = f"{OOC_PREFIX}\n\n{reply}"

    # Attach ONLY cited sources (based on [n] in reply)
    reply_with_marker = append_cited_sources_marker(reply, numbered_ctx)

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

    # Build sources payload for response model (only cited)
    _, items = split_reply_and_sources_for_api(reply_with_marker)
    return {"reply": reply_with_marker, "sources": items}


@app.post("/chat_stream")
def chat_stream(body: ChatIn, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admin cannot chat.")
    def gen():
        # RAG search
        try:
            contexts = search_qdrant(body.message, limit=TOP_K, user_id=user.id)
        except Exception as e:
            print("[RAG] search error:", e, traceback.format_exc())
            contexts = []

        prompt, numbered_ctx, ooc = build_prompt_with_numbered_context(body.message, contexts)
        max_toks = body.num_predict or 512

        chunks: List[str] = []

        try:
            # Show the emoji line FIRST if OOC
            if ooc:
                yield json.dumps({"type":"delta","text": f"{OOC_PREFIX}\n\n"}) + "\n"

            if body.provider == "groq":
                if not _groq_client:
                    yield json.dumps({"type":"delta","text":"[Groq key missing]\n"}) + "\n"
                else:
                    stream = _groq_client.chat.completions.create(
                        model=GROQ_MODEL_DEFAULT,
                        messages=[
                            {"role":"system","content":"You are a helpful, compassionate assistant."},
                            {"role":"user","content":prompt}
                        ],
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
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        d = ev.get("response")
                        if d:
                            chunks.append(d)
                            yield json.dumps({"type":"delta","text": d}) + "\n"
        except requests.exceptions.Timeout:
            yield json.dumps({"type":"delta","text":"[Provider timed out]\n"}) + "\n"
        finally:
            body_text = "".join(chunks).strip()
            final_text = body_text
            # If model ignored instruction, ensure the first line is present
            if ooc and not final_text.startswith(OOC_PREFIX):
                final_text = f"{OOC_PREFIX}\n\n{final_text}"

            final_text_with_marker = append_cited_sources_marker(final_text, numbered_ctx)

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

            # Send any trailing text (e.g., sources marker that wasn't streamed)
            yield json.dumps({"type":"delta","text": final_text_with_marker[len(final_text):]}) + "\n"
            yield json.dumps({"type":"done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/jsonl")

# ---------------- Chat Delete By USer----------------

@app.delete("/admin/users/{user_id}/vectors")
def admin_clear_user_vectors(user_id: int, db: Session = Depends(get_db), admin=Depends(require_admin)):
    delete_user_vectors(user_id)
    return {"ok": True}

@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str = Path(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admins cannot delete user sessions.")

    conv = (
        db.query(Conversation)
        .filter(Conversation.id == session_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")

    # delete messages belonging to this user & conversation
    db.query(Message).filter(
        Message.conversation_id == session_id,
        Message.user_id == user.id
    ).delete()

    # delete the conversation row
    db.delete(conv)
    db.commit()
    return {"ok": True}

# ---------------- Questionnaire logic ----------------

def is_continue_prompt_text(text: str) -> bool:
    """Detect our 'continue or score' prompt without using hidden markers."""
    t = (text or "").strip().lower()
    return ("reply with 'continue' or 'score'" in t) or t.startswith("you've answered")

def is_regular_question_text(text: str) -> bool:
    """Count only real questionnaire questions (end with ? and not a continue-prompt)."""
    if not text:
        return False
    t = text.strip()
    if not t.endswith("?"):
        return False
    return not is_continue_prompt_text(t)

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
            t.append(f"Q: {m['content'].strip()}")
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
def questionnaire_start(user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admin cannot start questionnaires.")
    # Create a new conversation
    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        title=f"[Questionnaire] {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
    )
    db.add(conv); db.commit()

    # Create q_session row with target = MIN_QUESTIONS
    qs = QuestionnaireSession(session_id=conv.id, user_id=user.id, target_questions=MIN_QUESTIONS, is_active=1)
    db.add(qs); db.commit()

    # First question (no markers)
    first_q = "To get started, how would you describe your overall mood today?"
    db.add(Message(conversation_id=conv.id, user_id=user.id, role="assistant", content=first_q))
    db.commit()

    return {"session_id": conv.id, "question": first_q}



@app.post("/questionnaire/answer", response_model=QAnswerOut)
def questionnaire_answer(body: QAnswerIn, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admin cannot answer questionnaires.")

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

    # Load history (assistant + user)
    hist = q_history(db, user.id, body.session_id)

    # Find last assistant message before this user answer
    last_assistant = None
    for m in reversed(hist[:-1]):  # exclude just-added user message
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    # Count actual questions asked so far (assistant messages that look like questions, not continue-prompt)
    actual_q_count = sum(
        1 for m in hist if m["role"] == "assistant" and is_regular_question_text(m["content"])
    )

    # If last message was the continue prompt, interpret user's reply
    if last_assistant and is_continue_prompt_text(last_assistant):
        ans = (body.answer or "").strip().lower()

        wants_continue = any(x in ans for x in ["continue", "yes", "y", "go on", "more", "add"])
        wants_score    = any(x in ans for x in ["score", "finish", "end", "done", "result"])

        if wants_continue and not wants_score:
            # bump target by 3 and ask next adaptive question
            qs.target_questions = (qs.target_questions or MIN_QUESTIONS) + CONTINUE_STEP
            db.commit()
            q = next_question_from_llm(hist)
            db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=q))
            db.commit()
            return {"done": False, "question": q}

        if wants_score and not wants_continue:
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

        # If unclear answer to continue prompt, politely re-ask the same continue question
        prompt_text = (
            f"You've answered {actual_q_count} questions so far. "
            f"Would you like to continue with {CONTINUE_STEP} more questions, or get your score now? "
            "Reply with 'continue' or 'score'."
        )
        db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=prompt_text))
        db.commit()
        return {"done": False, "question": prompt_text}

    # If we've reached target, ask the clear 'continue or score' question (no markers)
    if actual_q_count >= (qs.target_questions or MIN_QUESTIONS):
        prompt_text = (
            f"You've answered {actual_q_count} questions so far. "
            f"Would you like to continue with {CONTINUE_STEP} more questions, or get your score now? "
            "Reply with 'continue' or 'score'."
        )
        db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=prompt_text))
        db.commit()
        return {"done": False, "question": prompt_text}

    # Otherwise: ask next adaptive question
    q = next_question_from_llm(hist)
    db.add(Message(conversation_id=body.session_id, user_id=user.id, role="assistant", content=q))
    db.commit()
    return {"done": False, "question": q}


# ---------------- Questionnaire dashboard (fast) ----------------
class QDashItem(BaseModel):
    day: str
    attempts: int
    avg_score: float
    latest_comment: Optional[str] = None

@app.get("/questionnaire/dashboard", response_model=List[QDashItem])
def questionnaire_dashboard(user=Depends(get_current_user), db: Session = Depends(get_db)):
    if is_admin_principal(user):
        return []
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


# ---------------- Admin Endpoints ----------------
@app.get("/admin/users")
def admin_list_users(db: Session = Depends(get_db), admin=Depends(require_admin)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    out = []
    for u in users:
        chat_cnt = db.query(Conversation).filter(Conversation.user_id == u.id).count()
        msg_cnt = db.query(Message).filter(Message.user_id == u.id).count()
        qa_cnt = db.query(QuestionnaireAttempt).filter(QuestionnaireAttempt.user_id == u.id).count()
        active_qs = db.query(QuestionnaireSession).filter(QuestionnaireSession.user_id == u.id, QuestionnaireSession.is_active == 1).count()
        out.append({
            "id": u.id,
            "email": u.email,
            "created_at": u.created_at.isoformat(),
            "conversations": chat_cnt,
            "messages": msg_cnt,
            "questionnaire_attempts": qa_cnt,
            "active_questionnaires": active_qs
        })
    return out


@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: int, db: Session = Depends(get_db), admin=Depends(require_admin)):
    # ✅ delete their vectors first
    delete_user_vectors(user_id)

    db.query(Message).filter(Message.user_id == user_id).delete()
    db.query(Conversation).filter(Conversation.user_id == user_id).delete()
    db.query(QuestionnaireAttempt).filter(QuestionnaireAttempt.user_id == user_id).delete()
    db.query(QuestionnaireSession).filter(QuestionnaireSession.user_id == user_id).delete()
    db.query(User).filter(User.id == user_id).delete()
    db.commit()
    return {"ok": True}


@app.delete("/admin/users/{user_id}/chats")
def admin_clear_user_chats(
    user_id: int,
    purge_vectors: bool = False,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    db.query(Message).filter(Message.user_id == user_id).delete()
    db.query(Conversation).filter(Conversation.user_id == user_id).delete()
    db.commit()
    if purge_vectors:
        delete_user_vectors(user_id)
    return {"ok": True}



@app.delete("/admin/users/{user_id}/questionnaires")
def admin_clear_user_questionnaires(user_id: int, db: Session = Depends(get_db), admin=Depends(require_admin)):
    db.query(QuestionnaireAttempt).filter(QuestionnaireAttempt.user_id == user_id).delete()
    db.query(QuestionnaireSession).filter(QuestionnaireSession.user_id == user_id).delete()
    db.commit()
    return {"ok": True}


@app.patch("/admin/users/{user_id}/password")
def admin_update_user_password(user_id: int, new_pw: str = Body(..., embed=True), db: Session = Depends(get_db), admin=Depends(require_admin)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = hash_password(new_pw)
    db.commit()
    return {"ok": True}


@app.delete("/admin/vectorstore")
def admin_clear_vector_db(db: Session = Depends(get_db), admin=Depends(require_admin)):
    qc = init_qdrant()
    if qc:
        try:
            qc.delete_collection(COLLECTION)
        except Exception:
            # ignore if already deleted
            pass
    # also clear Document metadata table since they correspond to the vector store
    db.query(Document).delete()
    db.commit()
    return {"ok": True, "message": f"Vector collection '{COLLECTION}' deleted and document records cleared."}
