# app/config.py

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# --- Core Settings ---
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "240"))
DATABASE_URL = os.getenv("DATABASE_URL")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# --- RAG/VectorDB Settings ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "psych_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
OUT_OF_CONTEXT_THRESHOLD = float(os.getenv("OUT_OF_CONTEXT_THRESHOLD", "0.22"))

# --- LLM Provider Settings ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- Email/SMTP Settings ---
SMTP_HOST = os.getenv("SMTP_HOST", "mailhog")
SMTP_PORT = int(os.getenv("SMTP_PORT", "1025"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "Psych Support <no-reply@psychsupport.local>")
SMTP_SECURE = os.getenv("SMTP_SECURE", "none").lower()
SMTP_TIMEOUT = int(os.getenv("SMTP_TIMEOUT", "20"))
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
ENABLE_EMAIL_SCHEDULER = os.getenv("ENABLE_EMAIL_SCHEDULER", "1") == "1"

# --- Application Logic Constants ---
MIN_QUESTIONS = 7
CONTINUE_STEP = 3
OOC_PREFIX = "🌐 I couldn’t find context in your uploaded sources — this answer is drawn from general knowledge!!"
SOURCES_MARKER_PREFIX = "<!--SOURCES_JSON:"
SOURCES_MARKER_SUFFIX = "-->"