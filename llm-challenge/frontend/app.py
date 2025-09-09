
import os
import time
import requests
import streamlit as st
from datetime import datetime

# =============================
# Config
# =============================
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
APP_TITLE = "Psych Support — RAG Chat (Streamlit)"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)

# =============================
# Session State
# =============================
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []  # list of {id,title,created_at}

# =============================
# Helpers
# =============================

def auth_headers():
    if not st.session_state.token:
        return {}
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_get(path: str, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, headers=auth_headers(), timeout=30, **kwargs)
        return r
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None


def api_post(path: str, json=None, data=None, files=None, params=None):
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, headers=auth_headers(), json=json, data=data, files=files, params=params, timeout=120)
        return r
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return None


def refresh_sessions():
    r = api_get("/sessions")
    if r is not None and r.status_code == 200:
        st.session_state.sessions = r.json()
    else:
        st.session_state.sessions = []


def human_time(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_iso

# =============================
# Auth Box (Left Column)
# =============================
with st.sidebar:
    st.header("Account")

    if not st.session_state.token:
        mode = st.radio("Mode", ["Login", "Register"], horizontal=True)
        email = st.text_input("Email", key="email")
        password = st.text_input("Password", type="password", key="password")

        if st.button(mode):
            endpoint = "/login" if mode == "Login" else "/register"
            r = api_post(endpoint, json={"email": email, "password": password})
            if r is None:
                st.stop()
            if r.status_code == 200:
                if mode == "Register":
                    st.success("Registration successful. Please log in now.")
                else:
                    st.session_state.token = r.json()["access_token"]
                    st.success("Logged in.")
                    refresh_sessions()
                    st.rerun()
            else:
                st.error(r.json().get("detail", "Authentication error"))
    else:
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("🔄 Refresh Chats"):
                refresh_sessions()
        with cols[1]:
            if st.button("🚪 Logout"):
                st.session_state.token = None
                st.session_state.session_id = None
                st.session_state.sessions = []
                st.rerun()

        st.divider()
        st.subheader("Your Chats")

        new_title = st.text_input("New chat title", value="")
        if st.button("➕ New Chat"):
            params = {"title": new_title or None}
            r = api_post("/sessions/create", params=params)
            if r is not None and r.status_code == 200:
                sid = r.json()["id"]
                st.session_state.session_id = sid
                refresh_sessions()
                st.rerun()
            else:
                st.error(r.json().get("detail", "Failed to create chat") if r is not None else "Server error")

        if not st.session_state.sessions:
            refresh_sessions()
        if st.session_state.sessions:
            for row in st.session_state.sessions:
                btn_label = f"🗨️ {row['title'] or 'Untitled'}\n{human_time(row['created_at'])}"
                if st.button(btn_label, key=f"chat_{row['id']}"):
                    st.session_state.session_id = row["id"]
                    st.rerun()
        else:
            st.info("No chats yet.")

        st.divider()
        st.subheader("Upload Context (.txt)")
        up = st.file_uploader("Select a .txt file", type=["txt"])
        if up and st.button("Upload"):
            r = api_post("/ingest", files={"file": up})
            if r is not None and r.status_code == 200:
                st.success(f"Ingested {r.json().get('ingested_chunks', 0)} chunks.")
            else:
                st.error(r.json().get("detail", "Upload failed") if r is not None else "Server error")

# =============================
# Main Panel — Chat
# =============================
if not st.session_state.token:
    st.info("Login or register from the left panel to begin.")
    st.stop()

health_col1, health_col2, health_col3 = st.columns(3)
with health_col1:
    r = api_get("/health")
    if r is not None and r.status_code == 200:
        h = r.json()
        st.metric("API", "online" if h.get("ok") else "issue")
with health_col2:
    st.metric("Qdrant", "ready" if (r is not None and r.json().get("qdrant")) else "not ready")
with health_col3:
    st.metric("Embedder", "ready" if (r is not None and r.json().get("embedder")) else "not ready")

st.divider()

if not st.session_state.session_id:
    st.warning("Select a chat from the left or create a new one.")
    st.stop()

msgs_resp = api_get(f"/sessions/{st.session_state.session_id}/messages")
messages = []
if msgs_resp is not None and msgs_resp.status_code == 200:
    messages = msgs_resp.json()
else:
    st.error("Failed to load messages.")

for m in messages:
    role = m.get("role", "user")
    content = m.get("content", "")
    ts = datetime.fromisoformat(m.get("created_at", datetime.utcnow().isoformat())).strftime("%Y-%m-%d %H:%M")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
            st.caption(ts)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            st.caption(ts)

user_msg = st.chat_input("Type your message…")
if user_msg:
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.spinner("Thinking…"):
        r = api_post(
            "/chat",
            json={
                "message": user_msg,
                "session_id": st.session_state.session_id,
            },
        )
        if r is not None and r.status_code == 200:
            data = r.json()
            reply = data.get("reply", "")
            sources = data.get("sources", [])

            with st.chat_message("assistant"):
                st.markdown(reply)
                if sources:
                    with st.expander("Sources"):
                        for i, s in enumerate(sources, start=1):
                            meta = s.get("meta") or {}
                            fname = meta.get("filename") or ""
                            score = s.get("score")
                            st.write(f"[{i}] {fname} — score: {score}")

            time.sleep(0.1)
            st.rerun()
        else:
            try:
                err = r.json().get("detail", "Error") if r is not None else "Server error"
            except Exception:
                err = "Server error"
            st.error(err)