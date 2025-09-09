import os
import json
import requests
import streamlit as st
from datetime import datetime
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
APP_TITLE = "Psych Support — RAG Chat"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)

# -------- state --------
if "token" not in st.session_state: st.session_state.token = None
if "session_id" not in st.session_state: st.session_state.session_id = None   # selected conversation
if "sessions" not in st.session_state: st.session_state.sessions = []
if "edit_mode" not in st.session_state: st.session_state.edit_mode = False
if "provider" not in st.session_state: st.session_state.provider = "mistral"
if "streaming" not in st.session_state: st.session_state.streaming = True
if "view" not in st.session_state: st.session_state.view = "chat"  # chat | dashboard
if "username" not in st.session_state: st.session_state.username = ""
if "q_session_id" not in st.session_state: st.session_state.q_session_id = None  # current questionnaire session

SOURCES_MARKER_PREFIX = "<!--SOURCES_JSON:"
SOURCES_MARKER_SUFFIX = "-->"

# -------- helpers --------
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
    st.session_state.sessions = r.json() if (r is not None and r.status_code == 200) else []

def human_time(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_iso

def split_reply_and_sources(text: str):
    if not text: return text, []
    start = text.rfind(SOURCES_MARKER_PREFIX)
    if start == -1: return text, []
    end = text.find(SOURCES_MARKER_SUFFIX, start)
    if end == -1: return text, []
    try:
        payload = text[start + len(SOURCES_MARKER_PREFIX): end]
        items = json.loads(payload)
        clean = text[:start].rstrip()
        return clean, items if isinstance(items, list) else []
    except Exception:
        return text, []

# -------- login / register (compact centered) --------
if not st.session_state.token:
    colL, colC, colR = st.columns([1,2,1])
    with colC:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.write("### 🧠 Psych Support — AI Chat with RAG")
        st.caption("A supportive mental wellness companion with document-aware answers and an adaptive questionnaire.")
        st.markdown("</div>", unsafe_allow_html=True)
        mode = st.radio(" ", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submit = st.button(mode, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        endpoint = "/login" if mode == "Login" else "/auth/register"
        r = api_post(endpoint, json={"email": email, "password": password})
        if r is not None and r.status_code == 200:
            if mode == "Register":
                st.success("Registered. Please login.")
            else:
                st.session_state.token = r.json()["access_token"]
                h = api_get("/health")
                if h is not None and h.status_code == 200:
                    me = h.json().get("me", {})
                    st.session_state.username = me.get("username", "")
                st.rerun()
        else:
            try: st.error(r.json().get("detail", "Authentication error"))
            except Exception: st.error("Authentication error")
    st.stop()

# -------- sidebar --------
with st.sidebar:
    # Greeting
    greet_name = st.session_state.username
    if not greet_name:
        h = api_get("/health")
        if h is not None and h.status_code == 200:
            greet_name = h.json().get("me", {}).get("username", "there")
    st.markdown(f"### Hi {greet_name} 🙂")

    # Setup dropdown
    with st.expander("Setup"):
        st.subheader("Model Provider")
        st.session_state.provider = st.selectbox("Choose provider", ["mistral", "groq"], index=(0 if st.session_state.provider=="mistral" else 1))
        st.checkbox("Stream responses", value=st.session_state.streaming, key="streaming")

        st.subheader("Upload Context (.txt)")
        up = st.file_uploader("Select a .txt file", type=["txt"])
        if up and st.button("Upload"):
            r = api_post("/ingest", files={"file": up})
            if r is not None and r.status_code == 200:
                st.success(f"Ingested {r.json().get('ingested_chunks', 0)} chunks.")
            else:
                st.error(r.json().get("detail","Upload failed") if r is not None else "Server error")

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    # Questionnaire dashboard
    if st.button("📊 Questionnaire Dashboard"):
        st.session_state.view = "dashboard"
        st.stop()

    # Chats list
    st.subheader("Your Chats")
    colA, colB = st.columns([1,1])
    with colA:
        # Keep New Chat button — first message will create history
        if st.button("➕ New Chat"):
            st.session_state.session_id = None
            st.session_state.view = "chat"
            st.rerun()
    with colB:
        if st.button("🔄 Refresh"):
            refresh_sessions()

    if not st.session_state.sessions:
        refresh_sessions()
    for row in st.session_state.sessions:
        label = f"🗨️ {row['title'] or 'Chat'}\n{human_time(row['created_at'])}"
        if st.button(label, key=f"chat_{row['id']}"):
            st.session_state.view = "chat"
            st.session_state.session_id = row["id"]
            st.rerun()

# -------- dashboard view --------
if st.session_state.view == "dashboard":
    st.subheader("📊 Your Questionnaire Progress")
    with st.spinner("Loading dashboard…"):
        r = api_get("/questionnaire/dashboard")
    data = r.json() if (r is not None and r.status_code == 200) else []
    if not data:
        st.info("No questionnaires taken yet. Start one from the New Chat page.")
    else:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.stop()

# -------- chat / new-chat page --------
st.subheader("💬 New Chat" if not st.session_state.session_id else "💬 Conversation")

# Questionnaire CTA on New Chat page too
st.markdown("#### AI Questionnaire")
st.caption("Answer at least 10 adaptive questions. Then choose to continue (adds 3 more) or get your score.")
colq1, colq2 = st.columns([1,3])
with colq1:
    if st.button("Start Questionnaire"):
        r = api_post("/questionnaire/start")
        if r is not None and r.status_code == 200:
            data = r.json()
            st.session_state.q_session_id = data["session_id"]
            st.session_state.session_id = data["session_id"]   # show in chats later
            st.session_state.view = "chat"
            st.rerun()
with colq2:
    if st.session_state.q_session_id:
        st.success("Questionnaire in progress.")

st.markdown("---")

# If an existing chat is open, render messages; otherwise show "no messages yet"
if st.session_state.session_id:
    msgs = api_get(f"/sessions/{st.session_state.session_id}/messages")
    messages = msgs.json() if (msgs is not None and msgs.status_code == 200) else []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        ts = datetime.fromisoformat(m.get("created_at", datetime.utcnow().isoformat())).strftime("%Y-%m-%d %H:%M")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(text); st.caption(ts)
        else:
            clean, items = split_reply_and_sources(text)
            with st.chat_message("assistant"):
                st.markdown(clean)
                if items:
                    with st.expander("📚 Sources"):
                        for i, it in enumerate(items, start=1):
                            st.markdown(f"**{i}. {it.get('filename','')}**")
                            ex = it.get("excerpt","")
                            if ex:
                                st.markdown(f"> {ex}")
                st.caption(ts)
else:
    st.info("This is a fresh chat. Your first message will create a chat history automatically.")

# --- Questionnaire input or normal chat input
if st.session_state.q_session_id and st.session_state.session_id == st.session_state.q_session_id:
    # questionnaire mode
    ans = st.chat_input("Answer (Questionnaire)…")
    if ans:
        with st.chat_message("user"):
            st.markdown(ans)
        r = api_post("/questionnaire/answer", json={"session_id": st.session_state.q_session_id, "answer": ans})
        if r is not None and r.status_code == 200:
            data = r.json()
            if data.get("done"):
                with st.chat_message("assistant"):
                    st.success(f"Completed. Score: {data.get('score',0):.1f}/100")
                    if data.get("comment"):
                        st.markdown(data["comment"])
                    tips = data.get("tips", [])
                    if tips:
                        st.markdown("**Tips**")
                        for t in tips: st.markdown(f"- {t}")
                st.session_state.q_session_id = None  # end questionnaire
                st.rerun()
            else:
                with st.chat_message("assistant"):
                    st.markdown(data.get("question","(next question)"))
        else:
            st.error("Failed to submit answer.")
else:
    # normal chat mode
    user_msg = st.chat_input("Type your message…")
    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)
        if st.session_state.streaming:
            # streaming
            ph = st.chat_message("assistant")
            with ph:
                text_area = st.empty()
                acc = ""
                try:
                    with requests.post(
                        f"{API_BASE}/chat_stream",
                        headers=auth_headers(),
                        json={"message": user_msg, "session_id": st.session_state.session_id, "provider": st.session_state.provider},
                        stream=True, timeout=600,
                    ) as r:
                        r.raise_for_status()
                        for line in r.iter_lines(decode_unicode=True):
                            if not line: continue
                            try: ev = json.loads(line)
                            except Exception: continue
                            if ev.get("type") == "delta":
                                acc += ev.get("text", "")
                                clean,_ = split_reply_and_sources(acc)
                                text_area.markdown(clean)
                            elif ev.get("type") == "done":
                                break
                except Exception as e:
                    st.error(f"Streaming failed: {e}")
            st.rerun()
        else:
            # non-streaming
            r = api_post("/chat", json={"message": user_msg, "session_id": st.session_state.session_id, "provider": st.session_state.provider})
            if r is not None and r.status_code == 200:
                if not st.session_state.session_id:
                    refresh_sessions()
                st.rerun()
            else:
                try: st.error(r.json().get("detail", "Server error"))
                except Exception: st.error("Server error")
