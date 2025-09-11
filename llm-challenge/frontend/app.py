# frontend/app.py

import streamlit as st
import os

# Import all our new modules
from session import ensure_session_defaults
from views.sidebar import render_sidebar
from views.auth_view import render_auth_page
from views.admin_view import render_admin_page
from views.dashboard_view import render_dashboard_page
from views.videos_view import render_videos_page
from views.meditation_view import render_meditation_page
from views.yoga_view import render_yoga_page
from views.chat_view import render_chat_page
from views.modals import render_modal_panel

APP_TITLE = "Psych Support — RAG Chat"
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)

# Initialize session state with default values
ensure_session_defaults()

# --- Main App Router ---
if not st.session_state.token:
    render_auth_page()
else:
    # Render sidebar on every page for logged-in users
    render_sidebar()

    # Render the modal panel if it's open (it will only show UI if st.session_state.modal_open is True)
    render_modal_panel()

    # Main content area router
    view = st.session_state.view
    if view == "admin":
        render_admin_page()
    elif view == "dashboard":
        render_dashboard_page()
    elif view == "videos":
        render_videos_page()
    elif view == "meditation":
        render_meditation_page()
    elif view == "yoga":
        render_yoga_page()
    else:  # Default to "chat" view
        render_chat_page()