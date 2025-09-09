import os
import json
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
    st.session_state.provider = "mistral"
if "streaming" not in st.session_state:
    st.session_state.streaming = True

# ------------- helpers -------------
SOURCES_MARKER_PREFIX = "<!--SOURCES_JSON:"
SOURCES_MARKER_SUFFIX = "-->"

def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

def api_get(path: str, **kwargs):
    try:
        return requests.get(f"{API_BASE}{path}", headers=auth_headers(), timeout=30, **kwargs)
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None

def api_post(path: str, json=None, files=None, params=None, timeout=600):
    try:
        return requests.post(f"{API_BASE}{path}", headers=auth_headers(), json=json, files=files, params=params, timeout=timeout)
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

def split_reply_and_sources(text: str):
    """
    Returns (clean_text, items) where items = [{'filename','excerpt'}, ...]
    If no sources marker found, items = [].
    """
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
        # if parsing fails, show text as-is
        return text, []

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
        st.subheader("Model Provider")
        st.session_state.provider = st.selectbox(
            "Choose provider",
            options=["mistral", "groq"],
            index=(0 if st.session_state.provider == "mistral" else 1),
            help="Use **groq** for big questions. **mistral** (local) may be slow on long prompts.",
        )
        st.checkbox("Stream responses", value=st.session_state.streaming, key="streaming")

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
        st.subheader("Your Chats")
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("➕ New Chat"):
                r = api_post("/sessions/create")
                if r is not None and r.status_code == 200:
                    st.session_state.session_id = r.json()["id"]
                    refresh_sessions(); st.rerun()
                else:
                    st.error(r.json().get("detail", "Failed to create chat") if r is not None else "Server error")
        with colB:
            if st.button("🔄 Refresh"):
                refresh_sessions()
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

        st.divider()
        if st.button("🚪 Logout"):
            st.session_state.token = None
            st.session_state.session_id = None
            st.session_state.sessions = []
            st.session_state.edit_mode = False
            st.rerun()

# ------------- main panel -------------
if not st.session_state.token:
    st.info("Login or register from the left panel to begin.")
    st.stop()

if not st.session_state.session_id:
    st.warning("Select a chat from the left or create a new one.")
    st.stop()

msgs_resp = api_get(f"/sessions/{st.session_state.session_id}/messages")
messages = msgs_resp.json() if (msgs_resp is not None and msgs_resp.status_code == 200) else []

# Render history (assistant messages show dropdown with excerpts if available)
for m in messages:
    role = m.get("role", "user")
    text = m.get("content", "")
    ts = datetime.fromisoformat(m.get("created_at", datetime.utcnow().isoformat())).strftime("%Y-%m-%d %H:%M")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(text)
            st.caption(ts)
    else:
        clean_text, items = split_reply_and_sources(text)
        with st.chat_message("assistant"):
            st.markdown(clean_text)
            if items:
                with st.expander("📚 Sources"):
                    for i, it in enumerate(items, start=1):
                        fname = it.get("filename", "")
                        excerpt = it.get("excerpt", "")
                        st.markdown(f"**{i}. {fname}**")
                        if excerpt:
                            st.markdown(f"> {excerpt}")
            st.caption(ts)

# Chat input
user_msg = st.chat_input("Type your message…")
if user_msg:
    with st.chat_message("user"):
        st.markdown(user_msg)

    if st.session_state.get("streaming", True):
        # STREAM path
        msg_ph = st.chat_message("assistant")
        with msg_ph:
            text_area = st.empty()
            acc = ""
            try:
                with requests.post(
                    f"{API_BASE}/chat_stream",
                    headers=auth_headers(),
                    json={
                        "message": user_msg,
                        "session_id": st.session_state.session_id,
                        "provider": st.session_state.get("provider", "mistral"),
                    },
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
                        if ev.get("type") == "delta":
                            acc += ev.get("text", "")
                            # live strip marker part from display
                            clean, _items = split_reply_and_sources(acc)
                            text_area.markdown(clean)
                        elif ev.get("type") == "done":
                            break
            except Exception as e:
                st.error(f"Streaming failed: {e}")
        st.rerun()
    else:
        # NON STREAM path
        r = api_post(
            "/chat",
            json={
                "message": user_msg,
                "session_id": st.session_state.session_id,
                "provider": st.session_state.get("provider", "mistral"),
            },
        )
        if r is not None and r.status_code == 200:
            st.rerun()
        else:
            try:
                st.error(r.json().get("detail", "Server error"))
            except Exception:
                st.error("Server error")
