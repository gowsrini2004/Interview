# app.py — Streamlit Frontend (dual sidebar; no @st.dialog; version-agnostic modals)
# ACCOUNT > greeting, optional Setup (users) or Vector tools (admin)
# YOUR CHATS > Tabs: Questionnaire | Chats (date-wise groups)
# Includes: admin panel, pseudo-modals (no @st.dialog), grouped chats, convo header actions,
# and refresh/selection fixes so new chats appear immediately.

import os
import json
import requests
import streamlit as st
from datetime import datetime
import pandas as pd
from collections import defaultdict

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
APP_TITLE = "Psych Support — RAG Chat"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)

# -------- state --------
if "token" not in st.session_state: st.session_state.token = None
if "is_admin" not in st.session_state: st.session_state.is_admin = False
if "session_id" not in st.session_state: st.session_state.session_id = None
if "sessions" not in st.session_state: st.session_state.sessions = []
if "provider" not in st.session_state: st.session_state.provider = "mistral"
if "streaming" not in st.session_state: st.session_state.streaming = True
if "view" not in st.session_state: st.session_state.view = "chat"  # chat | dashboard | admin
if "username" not in st.session_state: st.session_state.username = ""
if "q_session_id" not in st.session_state: st.session_state.q_session_id = None

# Admin table action state (3-dots)
if "actions_user" not in st.session_state: st.session_state.actions_user = None

# Delete conversation state
if "del_conv_id" not in st.session_state: st.session_state.del_conv_id = None

# Version-agnostic "modal" panel state
if "modal_open" not in st.session_state: st.session_state.modal_open = False
if "modal_type" not in st.session_state: st.session_state.modal_type = None
if "modal_payload" not in st.session_state: st.session_state.modal_payload = {}

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

def api_delete(path: str, timeout=60):
    try:
        return requests.delete(f"{API_BASE}{path}", headers=auth_headers(), timeout=timeout)
    except Exception as e:
        st.error(f"DELETE {path} failed: {e}")
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

def get_session_by_id(sid: str):
    for s in st.session_state.sessions:
        if s["id"] == sid:
            return s
    return None

def refresh_sessions_and_select_latest_if_needed():
    """Refresh list; if no session selected, select the most recent one (just-created)."""
    refresh_sessions()
    if not st.session_state.session_id and st.session_state.sessions:
        st.session_state.session_id = st.session_state.sessions[0]["id"]

# -------------- modal helpers (no @st.dialog) --------------
def open_modal(modal_type: str, payload: dict | None = None):
    st.session_state.modal_open = True
    st.session_state.modal_type = modal_type
    st.session_state.modal_payload = payload or {}

def close_modal():
    st.session_state.modal_open = False
    st.session_state.modal_type = None
    st.session_state.modal_payload = {}

def render_modal_panel():
    """Renders a simple confirm/action panel at the top of the page when modal_open=True."""
    if not st.session_state.modal_open or not st.session_state.modal_type:
        return
    mtype = st.session_state.modal_type
    data = st.session_state.modal_payload or {}

    st.markdown("---")
    box = st.container()
    with box:
        if mtype == "vector_clear":
            st.markdown("### Clear Vector DB")
            st.warning("This will delete the entire Qdrant collection and all saved document records. This action cannot be undone.")
            confirm = st.text_input("Type DELETE to confirm", key="vs_confirm")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel"):
                    close_modal(); st.experimental_rerun()
            with c2:
                if st.button("Delete"):
                    if confirm.strip().upper() == "DELETE":
                        r = api_delete("/admin/vectorstore")
                        if r is not None and r.status_code in (200, 204):
                            st.success("Vector DB cleared.")
                        else:
                            st.error("Failed to clear vector DB.")
                        close_modal(); st.experimental_rerun()
                    else:
                        st.error("Please type DELETE to confirm.")

        elif mtype == "clear_chats":
            email = data.get("email","")
            user_id = data.get("user_id")
            st.markdown("### Clear Chats")
            st.warning(f"Clear **all chat history** for user: **{email}** (ID: {user_id})? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel"):
                    close_modal(); st.experimental_rerun()
            with c2:
                if st.button("Clear Chats"):
                    r = api_delete(f"/admin/users/{user_id}/chats")
                    if r is not None and r.status_code in (200, 204):
                        st.success("Chats cleared.")
                    else:
                        st.error("Failed clearing chats.")
                    close_modal(); st.experimental_rerun()

        elif mtype == "clear_questionnaires":
            email = data.get("email","")
            user_id = data.get("user_id")
            st.markdown("### Clear Questionnaires")
            st.warning(f"Clear **all questionnaire attempts & sessions** for user: **{email}** (ID: {user_id})?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel"):
                    close_modal(); st.experimental_rerun()
            with c2:
                if st.button("Clear Questionnaires"):
                    r = api_delete(f"/admin/users/{user_id}/questionnaires")
                    if r is not None and r.status_code in (200, 204):
                        st.success("Questionnaires cleared.")
                    else:
                        st.error("Failed clearing questionnaires.")
                    close_modal(); st.experimental_rerun()

        elif mtype == "user_actions":
            email = data.get("email","")
            user_id = data.get("user_id")
            st.markdown("### User Actions")
            st.write(f"**{email}** (ID: {user_id})")
            new_pw = st.text_input("New password", type="password", key=f"pw_{user_id}")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button("Close"):
                    close_modal(); st.experimental_rerun()
            with c2:
                if st.button("Update Password"):
                    if not new_pw:
                        st.warning("Enter a new password.")
                    else:
                        r = api_patch(f"/admin/users/{user_id}/password", json={"new_pw": new_pw})
                        if r is not None and r.status_code == 200:
                            st.success("Password updated.")
                        else:
                            st.error("Failed updating password.")
                        close_modal(); st.experimental_rerun()
            with c3:
                if st.button("Delete User"):
                    r = api_delete(f"/admin/users/{user_id}")
                    if r is not None and r.status_code in (200, 204):
                        st.success("User deleted.")
                    else:
                        st.error("Failed deleting user.")
                    close_modal(); st.experimental_rerun()

        elif mtype == "delete_conversation":
            sid = data.get("session_id")
            s = get_session_by_id(sid) if sid else None
            title = (s["title"] if s else sid) or "Chat"
            st.markdown("### Delete Conversation")
            st.error(f"Delete **{title}**? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel"):
                    close_modal(); st.experimental_rerun()
            with c2:
                if st.button("Delete"):
                    r = api_delete(f"/sessions/{sid}")
                    if r is not None and r.status_code in (200, 204):
                        if st.session_state.session_id == sid:
                            st.session_state.session_id = None
                        refresh_sessions()
                        st.success("Conversation deleted.")
                    else:
                        st.error("Failed to delete conversation.")
                    close_modal(); st.experimental_rerun()

    st.markdown("---")

# -------- login / register (compact centered) --------
if not st.session_state.token:
    colL, colC, colR = st.columns([1,2,1])
    with colC:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.write("### 🧠 Psych Support — AI Chat with RAG")
        st.caption("A supportive mental wellness companion with document-aware answers and an adaptive questionnaire.")
        st.markdown("</div>", unsafe_allow_html=True)

        # 👇 wrap login/register in a form
        with st.form("auth_form", clear_on_submit=False):
            mode = st.radio(" ", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
            identifier = st.text_input("Email or Username", placeholder="you@example.com (or admin)")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button(mode, use_container_width=True)

        # handle submit only once
        if submit:
            if mode == "Login":
                r = api_post("/auth/login", json={"identifier": identifier, "password": password})
            else:
                r = api_post("/auth/register", json={"email": identifier, "password": password})

            if r is not None and r.status_code == 200:
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.is_admin = data.get("is_admin", False)
                h = api_get("/health")
                if h is not None and h.status_code == 200:
                    me = h.json().get("me", {})
                    st.session_state.username = me.get("username", "there")
                st.session_state.view = "admin" if st.session_state.is_admin else "chat"
                st.experimental_rerun()
            else:
                try:
                    st.error(r.json().get("detail", "Authentication error"))
                except Exception:
                    st.error("Authentication error")
    st.stop()



# ======= render modal panel (if open) =======
render_modal_panel()

# =========================
# Dual Sidebar (ACCOUNT + YOUR CHATS)
# =========================
with st.sidebar:
    # ======= top-right logout (outside sidebar) =======
    if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()
    greet_name = st.session_state.username or "there"

    # ACCOUNT
    st.markdown(f"### Hi {greet_name} 🙂")
    with st.expander("## ⚙️ Utils", expanded=True):
        if st.session_state.is_admin:
            st.caption("Role: **Admin**")
            st.markdown("**Vector Store (Qdrant)**")
            st.caption("Delete the entire vector collection and document records.")
            if st.button("Clear Vector DB"):
                open_modal("vector_clear"); st.experimental_rerun()
        else:
                st.markdown("**Model Provider**")
                st.session_state.provider = st.selectbox(
                    "Choose provider",
                    ["groq","mistral"],
                    index=(0 if st.session_state.provider == "mistral" else 1),
                )
                st.checkbox("Stream responses", value=st.session_state.streaming, key="streaming")

                st.markdown("**Upload Context (.txt)**")
                up = st.file_uploader("Select a .txt file", type=["txt"])
                if up and st.button("Upload"):
                    r = api_post("/ingest", files={"file": up})
                    if r is not None and r.status_code == 200:
                        st.success(f"Ingested {r.json().get('ingested_chunks', 0)} chunks.")
                    else:
                        try:
                            st.error(r.json().get("detail","Upload failed"))
                        except Exception:
                            st.error("Server error during upload")
                            
    # QUESTIONERS
    with st.expander("## 📊 Questionnaire", expanded=True):
            if st.session_state.is_admin:
                st.info("No chats for admin account.")
            else:
                st.caption("Answer at least 10 adaptive questions. Then continue (+3) or get your score.")
                if st.button("📊 Your Dashboard"):
                        st.session_state.view = "dashboard"
                        st.experimental_rerun()     
                if st.button("▶ Start Questionnaire"):
                        r = api_post("/questionnaire/start")
                        if r is not None and r.status_code == 200:
                            data = r.json()
                            st.session_state.q_session_id = data["session_id"]
                            st.session_state.session_id = data["session_id"]
                            st.session_state.view = "chat"
                            st.experimental_rerun()
                if st.session_state.q_session_id:
                    st.success("Questionnaire in progress.")
        
        
    # YOUR CHATS
    with st.expander("## 🤖 Your Chats", expanded=True):
        if st.session_state.is_admin:
            st.info("No chats for admin account.")
        else:
                if not st.session_state.sessions:
                    refresh_sessions()
                
                colA, colB = st.columns([1,1])
                with colA:
                    if st.button("➕ New Chat"):
                        st.session_state.session_id = None
                        st.session_state.view = "chat"
                        st.rerun()
                with colB:
                    if st.button("🔄 Refresh"):
                        refresh_sessions()

                groups = defaultdict(list)
                for row in st.session_state.sessions:
                    d = human_time(row["created_at"])[:10]  # 'YYYY-MM-DD'
                    groups[d].append(row)

                date_keys = sorted(groups.keys(), reverse=True)
                for day in date_keys:
                    dt = datetime.strptime(day, "%Y-%m-%d")
                    label = dt.strftime("%b %d %y")
                    st.markdown(f"**{label}**")
                    for row in groups[day]:
                        btn_label = f"🗨️ {row['title'] or 'Chat'}"
                        if st.button(btn_label, key=f"chat_{row['id']}"):
                            st.session_state.view = "chat"
                            st.session_state.session_id = row["id"]
                            st.experimental_rerun()
                    st.markdown("---")

# -------- ADMIN view --------
if st.session_state.view == "admin":
    st.subheader("🛠 Admin Panel")

    # Users list
    ru = api_get("/admin/users")
    users = ru.json() if (ru is not None and ru.status_code == 200) else []
    if not users:
        st.info("No users found.")
        st.stop()

    # Search
    q = st.text_input("Search by email", key="admin_user_search")
    filtered = [u for u in users if (not q or q.lower() in u["email"].lower())]

    # Table header
    h1, h2, h3, h4, h5, h6 = st.columns([5,2,2,2,3,1])
    with h1: st.markdown("**Email**")
    with h2: st.markdown("**Conversations**")
    with h3: st.markdown("**Questionnaires**")
    with h4: st.markdown("**Clear Chats**")
    with h5: st.markdown("**Clear Questionnaire**")
    with h6: st.markdown("** **")

    # Rows
    for u in filtered:
        c1, c2, c3, c4, c5, c6 = st.columns([5,2,2,2,3,1])
        created = human_time(u["created_at"])
        email_html = f"<span title='Created: {created}'>{u['email']}</span>"
        with c1: st.markdown(email_html, unsafe_allow_html=True)
        with c2: st.markdown(str(u["conversations"]))
        with c3: st.markdown(str(u["questionnaire_attempts"]))
        with c4:
            if st.button("Clear", key=f"cc_{u['id']}"):
                open_modal("clear_chats", {"user_id": u["id"], "email": u["email"]}); st.experimental_rerun()
        with c5:
            if st.button("Clear", key=f"cq_{u['id']}"):
                open_modal("clear_questionnaires", {"user_id": u["id"], "email": u["email"]}); st.experimental_rerun()
        with c6:
            if st.button("⋯", key=f"menu_{u['id']}"):
                st.session_state.actions_user = {"id": u["id"], "email": u["email"]}
                open_modal("user_actions", {"user_id": u["id"], "email": u["email"]}); st.experimental_rerun()

    st.stop()

# -------- dashboard view --------
if st.session_state.view == "dashboard":
    st.subheader("📊 Your Questionnaire Progress")
    with st.spinner("Loading dashboard…"):
        r = api_get("/questionnaire/dashboard")
    data = r.json() if (r is not None and r.status_code == 200) else []
    if not data:
        st.info("No questionnaires taken yet. Start one from the sidebar.")
    else:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.stop()

# -------- chat / conversation page --------
st.subheader("💬 New Chat" if not st.session_state.session_id else "💬 Conversation")

# Conversation header actions (title edit + delete), users only
def get_current_title():
    s = get_session_by_id(st.session_state.session_id) if st.session_state.session_id else None
    return ((s["title"] if s else "") or "Chat")

if not st.session_state.is_admin and st.session_state.session_id:
    cA, cB, cC = st.columns([6,1,1])
    with cA:
        new_title = st.text_input("Title", value=get_current_title(), label_visibility="collapsed", key=f"title_{st.session_state.session_id}")
    with cB:
        if st.button("Save Title", key=f"save_title_{st.session_state.session_id}"):
            r = api_patch(f"/sessions/{st.session_state.session_id}", json={"title": new_title})
            if r is not None and r.status_code == 200:
                refresh_sessions()
                st.success("Title updated.")
                st.experimental_rerun()
            else:
                st.error("Failed to update title.")
    with cC:
        if st.button("Delete Chat", key=f"delete_chat_{st.session_state.session_id}"):
            st.session_state.del_conv_id = st.session_state.session_id
            open_modal("delete_conversation", {"session_id": st.session_state.session_id})
            st.experimental_rerun()

# If an existing chat is open, render messages; otherwise show "no messages yet"
if st.session_state.session_id and not st.session_state.is_admin:
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
    if not st.session_state.is_admin:
        st.info("This is a fresh chat. Your first message will create a chat history automatically.")

# --- Questionnaire input or normal chat input
if (not st.session_state.is_admin) and st.session_state.q_session_id and st.session_state.session_id == st.session_state.q_session_id:
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
                st.session_state.q_session_id = None
                st.experimental_rerun()
            else:
                with st.chat_message("assistant"):
                    st.markdown(data.get("question","(next question)"))
        else:
            st.error("Failed to submit answer.")
elif not st.session_state.is_admin:
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
            # Ensure new conversation appears immediately
            if not st.session_state.session_id:
                refresh_sessions_and_select_latest_if_needed()
            st.experimental_rerun()
        else:
            # non-streaming
            r = api_post("/chat", json={"message": user_msg, "session_id": st.session_state.session_id, "provider": st.session_state.provider})
            if r is not None and r.status_code == 200:
                if not st.session_state.session_id:
                    refresh_sessions_and_select_latest_if_needed()
                st.experimental_rerun()
            else:
                try: st.error(r.json().get("detail", "Server error"))
                except Exception: st.error("Server error")
