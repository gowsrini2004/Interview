# frontend/views/sidebar.py

import streamlit as st
from collections import defaultdict
from datetime import datetime

from session import ensure_session_defaults, is_admin, open_modal
from api import api_get, api_post, refresh_sessions
from helpers import human_time

def render_sidebar():
    """Renders the entire sidebar for both admin and user views."""
    ensure_session_defaults()

    with st.sidebar:
        if st.button("Logout", key="logout_sidebar"):
            st.session_state.clear()
            st.rerun()

        greet_name = st.session_state.get("username") or "there"
        st.markdown(f"### Hi {greet_name} 🙂")

        if is_admin():
            render_admin_sidebar()
        else:
            render_user_sidebar()

def render_admin_sidebar():
    """Renders the admin-specific sections of the sidebar."""
    st.subheader("Admin Console")

    with st.expander("⚙️ Utility", expanded=False):
        st.caption("Role: **Admin**")
        st.markdown("**Vector Store (Qdrant)**")
        st.caption("Delete the entire vector collection and document records.")
        if st.button("🧹 Clear Vector DB", key="btn_clear_vector_db"):
            open_modal("vector_clear")
            st.rerun()

    with st.expander("📧 Email Tool", expanded=False):
        st.caption("Visible only to Admins")
        if st.button("Remind - incomplete users"):
            r = api_post("/admin/emails/reminders")
            if r and r.status_code == 200:
                st.success(f"Sent: {r.json().get('sent', 0)}")
            else:
                st.error("Failed to send reminders")

        if st.button("Send Daily Scores (today)"):
            r = api_post("/admin/emails/daily-summaries")
            if r and r.status_code == 200:
                st.success(f"Sent: {r.json().get('sent', 0)}")
            else:
                st.error("Failed to send daily summaries")

        if st.button("Send Summary (overall)"):
            r = api_post("/admin/emails/periodic-summaries")
            if r and r.status_code == 200:
                st.success(f"Sent: {r.json().get('sent', 0)}")
            else:
                st.error("Failed to send periodic summaries")

    with st.expander("🎬 Videos", expanded=False):
        if st.button("🎬 Open Library", key="btn_videos_admin", use_container_width=True):
            st.session_state.view = "videos"
            st.rerun()

def render_user_sidebar():
    """Renders the user-specific sections of the sidebar."""
    with st.expander("⚙️ Utility", expanded=False):
        st.caption("Role: **User**")
        st.markdown("**Model Provider**")
        st.session_state.provider = st.selectbox(
            "Choose provider", ["groq", "mistral"],
            index=0 if st.session_state.get("provider") == "groq" else 1, key="select_provider"
        )
        st.checkbox("Stream responses", value=st.session_state.get("streaming", True), key="streaming")
        st.markdown("**Upload Context (.txt)**")
        up = st.file_uploader("Select a .txt file", type=["txt"], key="txt_uploader")
        if up and st.button("Upload", key="btn_upload_txt"):
            r = api_post("/ingest", files={"file": up})
            if r and r.status_code == 200:
                st.success(f"Ingested {r.json().get('ingested_chunks', 0)} chunks.")
            else:
                st.error(r.json().get("detail", "Upload failed") if r else "Server error")

    with st.expander("📊 Questionnaire", expanded=False):
        st.caption("Answer adaptive questions to get your score.")
        if st.button("📊 Your Dashboard", key="btn_user_dashboard", use_container_width=True):
            st.session_state.view = "dashboard"
            st.rerun()
        if st.button("▶ Start Questionnaire", key="btn_start_questionnaire", use_container_width=True):
            r = api_post("/questionnaire/start")
            if r and r.status_code == 200:
                data = r.json()
                st.session_state.q_session_id = data["session_id"]
                st.session_state.session_id = data["session_id"]
                st.session_state.view = "chat"
                st.rerun()
            else:
                st.error("Could not start questionnaire")

    with st.expander("🎬 Videos", expanded=False):
        if st.button("🎬 Open Library", key="btn_videos_user", use_container_width=True):
            st.session_state.view = "videos"
            st.rerun()

    with st.expander("🧘 Meditate", expanded=False):
        if st.button("🧘 Guided Meditation"):
            st.session_state.view = "meditation"; st.session_state.med_page = "guided"; st.rerun()
        if st.button("⛩ Meditation Room"):
            st.session_state.view = "meditation"; st.session_state.med_page = "room"; st.rerun()

    with st.expander("ૐ Yoga", expanded=False):
        if st.button("🧘Open Yoga Studio", key="sidebar_yoga"):
            st.session_state.view = "yoga"
            st.rerun()

    with st.expander("🤖 Your Chats", expanded=True):
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("➕ New Chat", key="btn_new_chat", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.view = "chat"
                st.rerun()
        with colB:
            if st.button("🔄 Refresh", key="btn_refresh_chats", use_container_width=True):
                st.session_state.pop("messages_cache", None)
                refresh_sessions()

        if not st.session_state.get("sessions"):
            refresh_sessions()

        # Chat Search Logic
        search_q = st.text_input("Search chats", key="chat_search", placeholder="Search by title…")
        sess_list = list(st.session_state.get("sessions", []))
        if search_q:
            q = search_q.strip().lower()
            sess_list = [s for s in sess_list if q in ((s.get("title") or "").lower())]

        # Render Chat List
        if not sess_list:
            st.info("No chats found.")
        else:
            groups = defaultdict(list)
            for row in sess_list:
                d = human_time(row["created_at"])[:10]
                groups[d].append(row)
            for day in sorted(groups.keys(), reverse=True):
                dt = datetime.strptime(day, "%Y-%m-%d")
                label = dt.strftime("%b %d, %Y")
                st.markdown(f"**{label}**")
                for row in groups[day]:
                    btn_label = f"🗨️ {row['title'] or 'Chat'}"
                    if st.button(btn_label, key=f"chat_{row['id']}"):
                        st.session_state.view = "chat"
                        st.session_state.session_id = row["id"]
                        st.rerun()