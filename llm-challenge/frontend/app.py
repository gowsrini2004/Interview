import os
import time
import requests
import streamlit as st
from datetime import datetime

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
APP_TITLE = "Psych Support — RAG Chat"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)

# ------------- session state -------------
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "provider" not in st.session_state:
    st.session_state.provider = "mistral"  # mistral (ollama) | groq

# ------------- helpers -------------

def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

def api_get(path: str, **kwargs):
    try:
        return requests.get(f"{API_BASE}{path}", headers=auth_headers(), timeout=30, **kwargs)
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None

def api_post(path: str, json=None, files=None, params=None, timeout=300):
    try:
        return requests.post(
            f"{API_BASE}{path}", headers=auth_headers(), json=json, files=files, params=params, timeout=timeout
        )
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return None

def api_patch(path: str, json=None, timeout=60):
    try:
        return requests.patch(f"{API_BASE}{path}", headers=auth_headers(), json=json, timeout=timeout)
    except Exception as e:
        st.error(f"PATCH {path} failed: {e}")
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

# ------------- SIDEBAR -------------
with st.sidebar:
    st.header("Account")
    if not st.session_state.token:
        mode = st.radio("Mode", ["Login", "Register"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button(mode):
            endpoint = "/login" if mode == "Login" else "/auth/register"
            r = api_post(endpoint, json={"email": email, "password": password})
            if r is None:
                st.stop()
            if r.status_code == 200:
                if mode == "Register":
                    st.success("Registration successful. Please log in now.")
                else:
                    st.session_state.token = r.json()["access_token"]
                    refresh_sessions(); st.rerun()
            else:
                try:
                    st.error(r.json().get("detail", "Authentication error"))
                except Exception:
                    st.error("Authentication error")
    else:
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("🔄 Refresh Chats"):
                refresh_sessions()
        with colB:
            if st.button("🚪 Logout"):
                st.session_state.token = None
                st.session_state.session_id = None
                st.session_state.sessions = []
                st.session_state.edit_mode = False
                st.rerun()

        st.divider()
        st.subheader("Your Chats")
        if st.button("➕ New Chat"):
            r = api_post("/sessions/create")
            if r is not None and r.status_code == 200:
                st.session_state.session_id = r.json()["id"]
                refresh_sessions(); st.rerun()
            else:
                st.error(r.json().get("detail", "Failed to create chat") if r is not None else "Server error")

        if not st.session_state.sessions:
            refresh_sessions()
        if st.session_state.sessions:
            for row in st.session_state.sessions:
                is_current = (row['id'] == st.session_state.session_id)
                if st.button(f"🗨️ {row['title'] or 'New Chat'}\n{human_time(row['created_at'])}", key=f"chat_{row['id']}"):
                    st.session_state.session_id = row['id']
                    st.session_state.edit_mode = False
                    st.rerun()
                if is_current:
                    if not st.session_state.edit_mode:
                        if st.button("✏️ Edit title"):
                            st.session_state.edit_mode = True
                            st.rerun()
                    else:
                        new_title = st.text_input("Title", value=row['title'] or "", key=f"title_{row['id']}")
                        ecols = st.columns([1,1])
                        with ecols[0]:
                            if st.button("💾 Save"):
                                r2 = api_patch(f"/sessions/{row['id']}", json={"title": new_title})
                                if r2 is not None and r2.status_code == 200:
                                    st.success("Title updated")
                                    st.session_state.edit_mode = False
                                    refresh_sessions(); st.rerun()
                                else:
                                    st.error(r2.json().get("detail", "Failed to update title") if r2 is not None else "Server error")
                        with ecols[1]:
                            if st.button("✖ Cancel"):
                                st.session_state.edit_mode = False
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

        st.divider()
        st.subheader("Model Provider")
        provider = st.selectbox(
            "Choose provider",
            options=["mistral", "groq"],
            index=(0 if st.session_state.provider == "mistral" else 1),
            help="Use **groq** for big questions. **mistral** (local) may be slow on long prompts.",
        )
        st.session_state.provider = provider

# ------------- main panel -------------
if not st.session_state.token:
    st.info("Login or register from the left panel to begin.")
    st.stop()

h = api_get("/health")
if h is not None and h.status_code == 200:
    j = h.json()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("API", "online" if j.get("ok") else "issue")
    with c2: st.metric("Qdrant", "ready" if j.get("qdrant") else "not ready")
    with c3: st.metric("Embedder", "ready" if j.get("embedder") else "not ready")

st.divider()

if not st.session_state.session_id:
    st.warning("Select a chat from the left or create a new one.")
    st.stop()

msgs_resp = api_get(f"/sessions/{st.session_state.session_id}/messages")
messages = msgs_resp.json() if (msgs_resp is not None and msgs_resp.status_code == 200) else []

for m in messages:
    role = m.get("role", "user")
    content = m.get("content", "")
    ts = datetime.fromisoformat(m.get("created_at", datetime.utcnow().isoformat())).strftime("%Y-%m-%d %H:%M")
    with st.chat_message("user" if role == "user" else "assistant"):
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
                "provider": st.session_state.get("provider", "mistral"),
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
