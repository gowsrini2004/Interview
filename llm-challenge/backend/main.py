import os
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Literal

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
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


# ---------------- helpers ----------------
SOURCES_MARKER_PREFIX = "<!--SOURCES_JSON:"
SOURCES_MARKER_SUFFIX = "-->"

def build_source_items(contexts: List[dict]) -> List[dict]:
    """Return [{filename, excerpt}] from qdrant contexts."""
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
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": num_predict},
            },
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
    # still return structured sources if you want to inspect/debug in UI
    sources: List[dict] = Field(default_factory=list)

class UpdateTitleIn(BaseModel):
    title: str


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "qdrant": init_qdrant() is not None,
        "embedder": init_embedder() is not None,
        "provider_defaults": {
            "mistral": OLLAMA_MODEL,
            "groq": GROQ_MODEL_DEFAULT if GROQ_API_KEY else None,
        },
    }


@app.post("/auth/register")
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered. Please log in instead.")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)
    conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title="New Chat")
    db.add(conv); db.commit()
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/register")
def register_alias(payload: RegisterIn, db: Session = Depends(get_db)):
    return register(payload, db)

@app.post("/login")
def login_alias(payload: LoginIn, db: Session = Depends(get_db)):
    return login(payload, db)


@app.post("/sessions/create")
def create_session(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title="New Chat")
    db.add(conv); db.commit()
    return {"id": conv.id, "title": conv.title}


@app.patch("/sessions/{session_id}")
def update_session_title(session_id: str, payload: UpdateTitleIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == session_id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")
    conv.title = payload.title[:255]
    db.commit()
    return {"id": conv.id, "title": conv.title}


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


def _short_title_from_text(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return "New Chat"
    words = t.split()
    s = " ".join(words[:8])
    if len(words) > 8:
        s += "…"
    return s[:80]


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # ensure session
    conv_id = body.session_id
    if not conv_id:
        conv = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).first()
        if not conv:
            conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title="New Chat")
            db.add(conv); db.commit()
        conv_id = conv.id

    # RAG search
    try:
        contexts = search_qdrant(body.message, limit=TOP_K)
    except Exception as e:
        print("[RAG] search error:", e, traceback.format_exc())
        contexts = []

    best_score = max((c["score"] for c in contexts), default=0.0)
    is_out = best_score < OUT_OF_CONTEXT_THRESHOLD

    system_prompt = (
        "You are a compassionate, knowledgeable mental wellness assistant. Give well-structured, practical, "
        "non-clinical advice. If content refers to documented sources, cite them as [source1], [source2], etc. "
        "If user is in crisis, advise contacting local emergency services."
    )

    if is_out or not contexts:
        prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{body.message}\n\n"
            "ASSISTANT:\nProvide a clear, supportive answer. "
            "Note: This answer is based on general knowledge and not from your uploaded documents."
        )
    else:
        MAX_CTX = 2000
        used = 0
        blocks = []
        for i, c in enumerate(contexts[:TOP_K], start=1):
            meta = c.get("meta", {})
            src = meta.get("filename") or meta.get("uploader") or c.get("id")
            text = (c["text"] or "")[:800]
            block = f"[source{i}] filename={src} score={c['score']:.4f}\n{text}"
            if used + len(block) > MAX_CTX:
                break
            blocks.append(block); used += len(block)
        ctx = "\n\n".join(blocks)
        prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"USER:\n{body.message}\n\n"
            "ASSISTANT:\nUse the context above; cite like [source1]."
        )

    # route provider
    provider = (body.provider or "mistral").lower()
    max_toks = body.num_predict or 512
    reply = call_groq(prompt, num_predict=max_toks) if provider == "groq" else call_ollama(prompt, num_predict=max_toks)

    # append invisible sources JSON marker (excerpts)
    reply_with_marker = append_sources_marker(reply, contexts)

    # save + auto-title
    try:
        db.add_all([
            Message(conversation_id=conv_id, user_id=user.id, role="user", content=body.message),
            Message(conversation_id=conv_id, user_id=user.id, role="assistant", content=reply_with_marker),
        ])
        conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user.id).first()
        if conv and (not conv.title or conv.title in ("New Chat", "Welcome Chat")):
            conv.title = _short_title_from_text(body.message)
        db.commit()
    except Exception as e:
        print("[DB] save failed:", e, traceback.format_exc())

    # still return structured sources for convenience (UI may ignore)
    return {"reply": reply_with_marker, "sources": build_source_items(contexts)}


@app.post("/chat_stream")
def chat_stream(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Streams JSON lines:
      {"type":"delta","text":"..."}  (many)
      {"type":"done"}
    The final delta contains the hidden <!--SOURCES_JSON:...--> marker appended.
    """
    def gen():
        # ensure session
        conv_id = body.session_id
        if not conv_id:
            conv = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).first()
            if not conv:
                conv = Conversation(id=str(uuid.uuid4()), user_id=user.id, title="New Chat")
                db.add(conv); db.commit()
            conv_id = conv.id

        # RAG search
        try:
            contexts = search_qdrant(body.message, limit=TOP_K)
        except Exception as e:
            print("[RAG] search error:", e, traceback.format_exc())
            contexts = []

        best_score = max((c["score"] for c in contexts), default=0.0)
        is_out = best_score < OUT_OF_CONTEXT_THRESHOLD

        system_prompt = (
            "You are a compassionate, knowledgeable mental wellness assistant. Give well-structured, practical, "
            "non-clinical advice. If content refers to documented sources, cite them as [source1], [source2], etc. "
            "If user is in crisis, advise contacting local emergency services."
        )

        if is_out or not contexts:
            prompt = (
                f"SYSTEM:\n{system_prompt}\n\n"
                f"USER:\n{body.message}\n\n"
                "ASSISTANT:\nProvide a clear, supportive answer. "
                "Note: This answer is based on general knowledge and not from your uploaded documents."
            )
        else:
            MAX_CTX = 2000
            used = 0
            blocks = []
            for i, c in enumerate(contexts[:TOP_K], start=1):
                meta = c.get("meta", {})
                src = meta.get("filename") or meta.get("uploader") or c.get("id")
                text = (c["text"] or "")[:800]
                block = f"[source{i}] filename={src} score={c['score']:.4f}\n{text}"
                if used + len(block) > MAX_CTX:
                    break
                blocks.append(block); used += len(block)
            ctx = "\n\n".join(blocks)
            prompt = (
                f"SYSTEM:\n{system_prompt}\n\n"
                f"CONTEXT:\n{ctx}\n\n"
                f"USER:\n{body.message}\n\n"
                "ASSISTANT:\nUse the context above; cite like [source1]."
            )

        provider = (body.provider or "mistral").lower()
        max_toks = body.num_predict or 512

        chunks: List[str] = []
        try:
            if provider == "groq":
                if not _groq_client:
                    yield json.dumps({"type":"delta","text":"[Groq key missing]\n"}) + "\n"
                else:
                    stream = _groq_client.chat.completions.create(
                        model=GROQ_MODEL_DEFAULT,
                        messages=[
                            {"role": "system", "content": "You are a helpful, compassionate assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_toks,
                        temperature=0.2,
                        stream=True,
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
                    stream=True,
                    timeout=600,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
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
            final_text = "".join(chunks)
            final_text_with_marker = append_sources_marker(final_text, contexts)

            try:
                db.add_all([
                    Message(conversation_id=conv_id, user_id=user.id, role="user", content=body.message),
                    Message(conversation_id=conv_id, user_id=user.id, role="assistant", content=final_text_with_marker),
                ])
                conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user.id).first()
                if conv and (not conv.title or conv.title in ("New Chat", "Welcome Chat")):
                    conv.title = _short_title_from_text(body.message)
                db.commit()
            except Exception as e:
                print("[DB] save failed (stream):", e, traceback.format_exc())

            # also show the marker at the end of the stream so UI can render sources immediately
            yield json.dumps({"type":"delta","text": final_text_with_marker[len(final_text):]}) + "\n"
            yield json.dumps({"type":"done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/jsonl")


if __name__ == "__main__":
    import uvicorn
    print("Starting app — Groq SDK enabled; sources embedded as excerpts via hidden marker.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
