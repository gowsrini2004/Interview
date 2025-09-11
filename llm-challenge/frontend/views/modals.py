# frontend/views/modals.py

import streamlit as st
from session import close_modal
from api import api_delete, api_patch
from helpers import get_session_by_id

def render_modal_panel():
    """Renders the active modal dialog at the top of the page."""
    if not st.session_state.modal_open or not st.session_state.modal_type:
        return

    st.markdown("---")
    with st.container():
        modal_map = {
            "vector_clear": render_vector_clear_modal,
            "clear_chats": render_clear_chats_modal,
            "clear_questionnaires": render_clear_questionnaires_modal,
            "user_actions": render_user_actions_modal,
            "delete_conversation": render_delete_conversation_modal,
        }
        render_function = modal_map.get(st.session_state.modal_type)
        if render_function:
            render_function(st.session_state.modal_payload)
    st.markdown("---")

def render_vector_clear_modal(payload):
    st.markdown("### Clear Vector DB")
    st.warning("This will delete the entire Qdrant collection and all saved document records. This action cannot be undone.")
    confirm = st.text_input("Type DELETE to confirm", key="vs_confirm")
    c1, c2 = st.columns(2)
    if c1.button("Cancel"):
        close_modal(); st.rerun()
    if c2.button("Delete"):
        if confirm.strip().upper() == "DELETE":
            r = api_delete("/admin/vectorstore")
            st.success("Vector DB cleared.") if r and r.ok else st.error("Failed to clear vector DB.")
            close_modal(); st.rerun()
        else:
            st.error("Please type DELETE to confirm.")

def render_clear_chats_modal(payload):
    st.markdown("### Clear Chats")
    st.warning(f"Clear all chat history for user: **{payload.get('email')}**? This cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel"):
        close_modal(); st.rerun()
    if c2.button("Clear Chats"):
        r = api_delete(f"/admin/users/{payload.get('user_id')}/chats")
        st.success("Chats cleared.") if r and r.ok else st.error("Failed clearing chats.")
        close_modal(); st.rerun()

def render_clear_questionnaires_modal(payload):
    st.markdown("### Clear Questionnaires")
    st.warning(f"Clear all questionnaire attempts for user: **{payload.get('email')}**?")
    c1, c2 = st.columns(2)
    if c1.button("Cancel"):
        close_modal(); st.rerun()
    if c2.button("Clear Questionnaires"):
        r = api_delete(f"/admin/users/{payload.get('user_id')}/questionnaires")
        st.success("Questionnaires cleared.") if r and r.ok else st.error("Failed clearing questionnaires.")
        close_modal(); st.rerun()

def render_user_actions_modal(payload):
    st.markdown(f"### User Actions for **{payload.get('email')}**")
    new_pw = st.text_input("New password", type="password", key=f"pw_{payload.get('user_id')}")
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("Close"):
        close_modal(); st.rerun()
    if c2.button("Update Password"):
        if new_pw:
            r = api_patch(f"/admin/users/{payload.get('user_id')}/password", json={"new_pw": new_pw})
            st.success("Password updated.") if r and r.ok else st.error("Failed updating password.")
            close_modal(); st.rerun()
        else:
            st.warning("Enter a new password.")
    if c3.button("Delete User", type="primary"):
        r = api_delete(f"/admin/users/{payload.get('user_id')}")
        st.success("User deleted.") if r and r.ok else st.error("Failed deleting user.")
        close_modal(); st.rerun()

def render_delete_conversation_modal(payload):
    sid = payload.get("session_id")
    s = get_session_by_id(sid)
    title = (s["title"] if s else sid) or "Chat"
    st.markdown("### Delete Conversation")
    st.error(f"Delete **{title}**? This cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel"):
        close_modal(); st.rerun()
    if c2.button("Delete"):
        r = api_delete(f"/sessions/{sid}")
        if r and r.ok:
            if st.session_state.session_id == sid:
                st.session_state.session_id = None
            st.success("Conversation deleted.")
        else:
            st.error("Failed to delete conversation.")
        close_modal(); st.rerun()