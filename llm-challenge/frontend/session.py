# frontend/session.py

import streamlit as st

def ensure_session_defaults():
    """Initializes all required keys in the session state with default values."""
    defaults = {
        "token": None,
        "is_admin": False,
        "username": "",
        "view": "chat",
        "provider": "groq",
        "streaming": True,
        "sessions": [],
        "session_id": None,
        "q_session_id": None,
        "messages_cache": {},
        "actions_user": None,
        "del_conv_id": None,
        "modal_open": False,
        "modal_type": None,
        "modal_payload": {},
        "med_page": "guided",  # Default meditation sub-page
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def is_admin() -> bool:
    """Checks if the current user is an admin."""
    return bool(st.session_state.get("is_admin", False))

def open_modal(modal_type: str, payload: dict | None = None):
    """Opens a modal dialog by setting session state variables."""
    st.session_state.modal_open = True
    st.session_state.modal_type = modal_type
    st.session_state.modal_payload = payload or {}

def close_modal():
    """Closes a modal dialog by clearing session state variables."""
    st.session_state.modal_open = False
    st.session_state.modal_type = None
    st.session_state.modal_payload = {}