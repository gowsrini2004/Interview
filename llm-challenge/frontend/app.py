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
from datetime import datetime
from zoneinfo import ZoneInfo
# ---------- sensible defaults ----------
def ensure_session_defaults():
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("is_admin", False)
    st.session_state.setdefault("provider", "groq")
    st.session_state.setdefault("streaming", True)
    st.session_state.setdefault("view", "chat")         # "chat" | "dashboard" | "videos" | "email"
    st.session_state.setdefault("sessions", [])
    st.session_state.setdefault("messages_cache", {})

def is_admin() -> bool:
    return bool(st.session_state.get("is_admin", False))



LOCAL_TZ = ZoneInfo("Asia/Kolkata")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
APP_TITLE = "Psych Support — RAG Chat"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)

# -------- state --------
if "token" not in st.session_state: st.session_state.token = None
if "is_admin" not in st.session_state: st.session_state.is_admin = False
if "session_id" not in st.session_state: st.session_state.session_id = None
if "sessions" not in st.session_state: st.session_state.sessions = []
if "provider" not in st.session_state: st.session_state.provider = "groq"
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
                    close_modal(); st.rerun()
            with c2:
                if st.button("Delete"):
                    if confirm.strip().upper() == "DELETE":
                        r = api_delete("/admin/vectorstore")
                        if r is not None and r.status_code in (200, 204):
                            st.success("Vector DB cleared.")
                        else:
                            st.error("Failed to clear vector DB.")
                        close_modal(); st.rerun()
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
                    close_modal(); st.rerun()
            with c2:
                if st.button("Clear Chats"):
                    r = api_delete(f"/admin/users/{user_id}/chats")
                    if r is not None and r.status_code in (200, 204):
                        st.success("Chats cleared.")
                    else:
                        st.error("Failed clearing chats.")
                    close_modal(); st.rerun()

        elif mtype == "clear_questionnaires":
            email = data.get("email","")
            user_id = data.get("user_id")
            st.markdown("### Clear Questionnaires")
            st.warning(f"Clear **all questionnaire attempts & sessions** for user: **{email}** (ID: {user_id})?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel"):
                    close_modal(); st.rerun()
            with c2:
                if st.button("Clear Questionnaires"):
                    r = api_delete(f"/admin/users/{user_id}/questionnaires")
                    if r is not None and r.status_code in (200, 204):
                        st.success("Questionnaires cleared.")
                    else:
                        st.error("Failed clearing questionnaires.")
                    close_modal(); st.rerun()

        elif mtype == "user_actions":
            email = data.get("email","")
            user_id = data.get("user_id")
            st.markdown("### User Actions")
            st.write(f"**{email}** (ID: {user_id})")
            new_pw = st.text_input("New password", type="password", key=f"pw_{user_id}")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button("Close"):
                    close_modal(); st.rerun()
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
                        close_modal(); st.rerun()
            with c3:
                if st.button("Delete User"):
                    r = api_delete(f"/admin/users/{user_id}")
                    if r is not None and r.status_code in (200, 204):
                        st.success("User deleted.")
                    else:
                        st.error("Failed deleting user.")
                    close_modal(); st.rerun()

        elif mtype == "delete_conversation":
            sid = data.get("session_id")
            s = get_session_by_id(sid) if sid else None
            title = (s["title"] if s else sid) or "Chat"
            st.markdown("### Delete Conversation")
            st.error(f"Delete **{title}**? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel"):
                    close_modal(); st.rerun()
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
                    close_modal(); st.rerun()

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
                st.rerun()
            else:
                try:
                    st.error(r.json().get("detail", "Authentication error"))
                except Exception:
                    st.error("Authentication error")
    st.stop()



# ======= render modal panel (if open) =======
render_modal_panel()

def render_sidebar():
    ensure_session_defaults()

    with st.sidebar:
        # Logout (in sidebar). If you prefer top-right, use the optional block above instead and remove this:
        if st.button("Logout", key="logout_sidebar"):
            st.session_state.clear()
            st.rerun()

        greet_name = st.session_state.get("username") or "there"
        st.markdown(f"### Hi {greet_name} 🙂")

        # ------------------ ADMIN VIEW ------------------
        if is_admin():
            st.subheader("Admin Console")

            # Utility (Admin)
            with st.expander("⚙️ Utility", expanded=False):
                st.caption("Role: **Admin**")
                st.markdown("**Vector Store (Qdrant)**")
                st.caption("Delete the entire vector collection and document records.")
                if st.button("🧹 Clear Vector DB", key="btn_clear_vector_db"):
                    open_modal("vector_clear")
                    st.rerun()

            # Email tool (Admin only)
            with st.expander("📧 Email Tool", expanded=False):
                st.caption("Visible only to Admins")
                if st.button("Remind - incomplete users"):
                        r = api_post("/admin/emails/reminders")
                        if r is not None and r.status_code == 200:
                            st.success(f"Sent: {r.json().get('sent', 0)}")
                        else:
                            st.error("Failed to send reminders")

                if st.button("Send Daily Scores (today)"):
                        r = api_post("/admin/emails/daily-summaries")
                        if r is not None and r.status_code == 200:
                            st.success(f"Sent: {r.json().get('sent', 0)}")
                        else:
                            st.error("Failed to send daily summaries")

                if st.button("Send Summary (overall + monthly)"):
                        r = api_post("/admin/emails/periodic-summaries")
                        if r is not None and r.status_code == 200:
                            st.success(f"Sent: {r.json().get('sent', 0)}")
                        else:
                            st.error("Failed to send periodic summaries")
                            
            # Videos (optional for Admin too)
            with st.expander("🎬 Videos", expanded=False):
                if st.button("🎬 Open Library", key="btn_videos_admin", use_container_width=True):
                    st.session_state.view = "videos"
                    st.rerun()

            # NOTE: Questionnaire & Your Chats are intentionally hidden for Admin.

        # ------------------ USER VIEW ------------------
        else:
            # Utility (User)
            with st.expander("⚙️ Utility", expanded=False):
                st.caption("Role: **User**")

                st.markdown("**Model Provider**")
                st.session_state.provider = st.selectbox(
                    "Choose provider",
                    ["groq", "mistral"],
                    index=0 if st.session_state.get("provider") == "groq" else 1,
                    key="select_provider",
                )

                st.checkbox(
                    "Stream responses",
                    value=st.session_state.get("streaming", True),
                    key="streaming"
                )

                st.markdown("**Upload Context (.txt)**")
                up = st.file_uploader("Select a .txt file", type=["txt"], key="txt_uploader")
                if up and st.button("Upload", key="btn_upload_txt"):
                    r = api_post("/ingest", files={"file": up})
                    if r is not None and r.status_code == 200:
                        st.success(f"Ingested {r.json().get('ingested_chunks', 0)} chunks.")
                    else:
                        try:
                            st.error(r.json().get("detail", "Upload failed"))
                        except Exception:
                            st.error("Server error during upload")

            # Questionnaire (User only)
            with st.expander("📊 Questionnaire", expanded=False):
                st.caption("Answer at least 10 adaptive questions. Then continue (+3) or get your score.")
                if st.button("📊 Your Dashboard", key="btn_user_dashboard", use_container_width=True):
                        st.session_state.view = "dashboard"
                        st.rerun()
                if st.button("▶ Start Questionnaire", key="btn_start_questionnaire", use_container_width=True):
                        r = api_post("/questionnaire/start")
                        if r is not None and r.status_code == 200:
                            data = r.json()
                            st.session_state.q_session_id = data["session_id"]
                            st.session_state.session_id = data["session_id"]
                            st.session_state.view = "chat"
                            st.rerun()
                        else:
                            st.error("Could not start questionnaire")

            # Videos (User)
            with st.expander("🎬 Videos", expanded=False):
                if st.button("🎬 Open Library", key="btn_videos_user", use_container_width=True):
                    st.session_state.view = "videos"
                    st.rerun()

            # Your Chats (User only)
            with st.expander("🤖 Your Chats", expanded=False):
                # Quick actions
                colA, colB = st.columns([1, 1])
                with colA:
                    if st.button("➕ New Chat", key="btn_new_chat", use_container_width=True):
                        st.session_state.session_id = None
                        st.session_state.view = "chat"
                        st.rerun()
                with colB:
                    if st.button("🔄 Refresh", key="btn_refresh_chats", use_container_width=True):
                        # clear cache on refresh
                        st.session_state.pop("messages_cache", None)
                        refresh_sessions()

                # Ensure sessions are loaded
                if not st.session_state.get("sessions"):
                    refresh_sessions()

                # Search
                search_q = st.text_input(
                    "Search chats",
                    key="chat_search",
                    placeholder="Search by title or message…",
                )

                row1 = st.container()
                col1, col2 = row1.columns([4, 1])
                with col1:
                    search_in_messages = st.checkbox(
                        "Search inside messages",
                        key="chat_search_messages",
                        value=False
                    )

                def _clear_chat_search():
                    st.session_state.update({
                        "chat_search": "",
                        "chat_search_messages": False,
                    })
                    st.session_state.pop("messages_cache", None)

                with col2:
                    st.button("❌", key="chat_clear_btn", on_click=_clear_chat_search)

                # --- Filter logic ---
                sess_list = list(st.session_state.get("sessions", []))  # copy
                q = (search_q or "").strip().lower()

                # simple cache for session messages to avoid re-fetching repeatedly
                if "messages_cache" not in st.session_state:
                    st.session_state.messages_cache = {}

                if q:
                    if search_in_messages:
                        matches = []
                        for s in sess_list:
                            title = (s.get("title") or "").lower()
                            if q in title:
                                matches.append(s)
                                continue
                            sid = s["id"]
                            cache = st.session_state.messages_cache.get(sid)
                            if cache is None:
                                rmsgs = api_get(f"/sessions/{sid}/messages")
                                cache = rmsgs.json() if (rmsgs is not None and rmsgs.status_code == 200) else []
                                st.session_state.messages_cache[sid] = cache
                            if any(q in (m.get("content", "").lower()) for m in cache):
                                matches.append(s)
                        sess_list = matches
                    else:
                        # title-only search
                        sess_list = [s for s in sess_list if q in ((s.get("title") or "").lower())]

                # --- Render results ---
                if not sess_list:
                    st.info("No chats matched your search.")
                else:
                    groups = defaultdict(list)
                    for row in sess_list:
                        d = human_time(row["created_at"])[:10]  # 'YYYY-MM-DD'
                        groups[d].append(row)

                    for day in sorted(groups.keys(), reverse=True):
                        dt = datetime.strptime(day, "%Y-%m-%d")
                        label = dt.strftime("%b %d %y")
                        st.markdown(f"**{label}**")
                        for row in groups[day]:
                            btn_label = f"🗨️ {row['title'] or 'Chat'}"
                            if st.button(btn_label, key=f"chat_{row['id']}"):
                                st.session_state.view = "chat"
                                st.session_state.session_id = row["id"]
                                st.rerun()
                        st.markdown("---")

render_sidebar()

if st.session_state.view == "admin":
    st.header("🛠 Admin Panel")

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
                open_modal("clear_chats", {"user_id": u["id"], "email": u["email"]}); st.rerun()
        with c5:
            if st.button("Clear", key=f"cq_{u['id']}"):
                open_modal("clear_questionnaires", {"user_id": u["id"], "email": u["email"]}); st.rerun()
        with c6:
            if st.button("⋯", key=f"menu_{u['id']}"):
                st.session_state.actions_user = {"id": u["id"], "email": u["email"]}
                open_modal("user_actions", {"user_id": u["id"], "email": u["email"]}); st.rerun()

    
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
    
# -------- videos view --------
if st.session_state.view == "videos":
    st.subheader("🎬 Video Library")

    # Admin add form (visible only to admin)
    if st.session_state.is_admin:
        with st.expander("➕ Add a YouTube Video (Admin)", expanded=False):
            with st.form("add_video_form", clear_on_submit=True):
                v_title = st.text_input("Title")
                v_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                v_desc = st.text_area("Description (optional)", height=80)
                v_tags = st.text_input("Tags (comma-separated, optional)", placeholder="stress, breathing, sleep")
                v_public = st.checkbox("Public", value=True)
                submitted = st.form_submit_button("Add Video")
            if submitted:
                payload = {
                    "title": v_title,
                    "url": v_url,
                    "description": v_desc or None,
                    "tags": [t.strip() for t in (v_tags or "").split(",") if t.strip()],
                    "is_public": v_public,
                }
                r = api_post("/admin/videos", json=payload)
                if r is not None and r.status_code == 200:
                    st.success("Video added.")
                else:
                    try: st.error(r.json().get("detail", "Failed to add video"))
                    except Exception: st.error("Server error while adding video")

    # Search
    q = st.text_input("Search videos (title / description / tags)", key="video_search", placeholder="e.g., anxiety, breathing")
    params = {"q": q} if q else None
    with st.spinner("Loading videos…"):
        rv = api_get("/videos", params=params)
    videos = rv.json() if (rv is not None and rv.status_code == 200) else []

    if not videos:
        st.info("No videos found.")
        st.stop()

    # Render in a responsive 2-column grid
    def render_tags(tags: list[str]):
        if not tags: return
        st.markdown(" ".join([f"<span style='padding:2px 8px;border:1px solid #444;border-radius:12px;font-size:12px;margin-right:4px;display:inline-block'>{t}</span>" 
                              for t in tags]), unsafe_allow_html=True)

    cols_per_row = 2
    for i in range(0, len(videos), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(videos): break
            v = videos[idx]
            with col:
                st.video(v["url"])
                st.markdown(f"**{v['title']}**")
                if v.get("description"):
                    st.caption(v["description"])
                render_tags(v.get("tags", []))
                st.caption(f"Added by: {v.get('added_by','admin')}  •  Platform: {v.get('platform','youtube').title()}  •  Visibility: {'Public' if v.get('is_public') else 'Hidden'}")
                # admin controls
                if st.session_state.is_admin:
                    c1, c2 = st.columns([1,1])
                    with c1:
                        if st.button("Delete", key=f"del_{v['id']}"):
                            rr = api_delete(f"/admin/videos/{v['id']}")
                            if rr is not None and rr.status_code in (200, 204):
                                st.success("Deleted.")
                                st.rerun()
                            else:
                                try: st.error(rr.json().get("detail","Delete failed"))
                                except Exception: st.error("Server error")
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
                st.rerun()
            else:
                st.error("Failed to update title.")
    with cC:
        if st.button("Delete Chat", key=f"delete_chat_{st.session_state.session_id}"):
            st.session_state.del_conv_id = st.session_state.session_id
            open_modal("delete_conversation", {"session_id": st.session_state.session_id})
            st.rerun()

# If an existing chat is open, render messages; otherwise show "no messages yet"
if st.session_state.session_id and not st.session_state.is_admin:
    msgs = api_get(f"/sessions/{st.session_state.session_id}/messages")
    messages = msgs.json() if (msgs is not None and msgs.status_code == 200) else []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        dt = datetime.fromisoformat(m.get("created_at", datetime.utcnow().isoformat()))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))   # backend saves UTC
        ts = dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(text); st.caption(ts)
        else:
            clean, items = split_reply_and_sources(text)
            with st.chat_message("assistant"):
                st.markdown(clean)

                if items:
                    with st.expander("📚 Sources", expanded=False):
                        # Title line as blockquote
                        st.markdown("> Sources")

                        # Items are already the exact cited subset in order of first appearance
                        for it in items:
                            idx = it.get("index") or 0
                            excerpt = it.get("excerpt", "")
                            fname = it.get("filename", "")
                            vid = it.get("id") or ""

                            # Line 1: "n - excerpt"
                            st.markdown(f"{idx} - {excerpt}")

                            # Line 2: "   - File name (vector id)"
                            # Use non-breaking spaces to indent in markdown
                            if fname or vid:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {fname}{f' ({vid})' if vid else ''}",
                                            unsafe_allow_html=True)

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
                st.session_state.session_id = None   # optional: also close that chat
                st.rerun()
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
            st.rerun()
        else:
            # non-streaming
            r = api_post("/chat", json={"message": user_msg, "session_id": st.session_state.session_id, "provider": st.session_state.provider})
            if r is not None and r.status_code == 200:
                if not st.session_state.session_id:
                    refresh_sessions_and_select_latest_if_needed()
                st.rerun()
            else:
                try: st.error(r.json().get("detail", "Server error"))
                except Exception: st.error("Server error")
