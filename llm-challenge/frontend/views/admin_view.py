# frontend/views/admin_view.py

import streamlit as st
from api import api_get
from helpers import human_time
from session import open_modal

def render_admin_page():
    """Renders the admin panel for user management."""
    st.header("🛠 Admin Panel")

    ru = api_get("/admin/users")
    users = ru.json() if ru and ru.status_code == 200 else []
    if not users:
        st.info("No users found.")
        return

    q = st.text_input("Search by email", key="admin_user_search")
    filtered = [u for u in users if (not q or q.lower() in u["email"].lower())]

    # Table header
    h1, h2, h3, h4, h5, h6 = st.columns([5, 2, 3, 2, 3, 1])
    h1.markdown("**Email**")
    h2.markdown("**Chats**")
    h3.markdown("**Questionnaires**")
    h4.markdown("**Clear Chats**")
    h5.markdown("**Clear Q's**")

    # Table rows
    for u in filtered:
        c1, c2, c3, c4, c5, c6 = st.columns([5, 2, 3, 2, 3, 1])
        created = human_time(u["created_at"])
        c1.markdown(f"<span title='Created: {created}'>{u['email']}</span>", unsafe_allow_html=True)
        c2.markdown(str(u["conversations"]))
        c3.markdown(str(u["questionnaire_attempts"]))
        
        with c4:
            if st.button("Clear", key=f"cc_{u['id']}"):
                open_modal("clear_chats", {"user_id": u["id"], "email": u["email"]}); st.rerun()
        with c5:
            if st.button("Clear", key=f"cq_{u['id']}"):
                open_modal("clear_questionnaires", {"user_id": u["id"], "email": u["email"]}); st.rerun()
        with c6:
            if st.button("⋯", key=f"menu_{u['id']}"):
                open_modal("user_actions", {"user_id": u["id"], "email": u["email"]}); st.rerun()