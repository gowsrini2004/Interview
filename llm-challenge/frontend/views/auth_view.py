# frontend/views/auth_view.py

import streamlit as st
import requests
from api import api_get, api_post

def render_auth_page():
    """Renders the login and registration form, handling authentication."""
    colL, colC, colR = st.columns([1, 2, 1])
    with colC:
        st.markdown("<div style='text-align:center;'><h3>🧠 Psych Support — AI Chat with RAG</h3><p>A supportive mental wellness companion.</p></div>", unsafe_allow_html=True)

        mode = st.radio(" ", ["Login", "Register"], horizontal=True, label_visibility="collapsed")

        with st.form("auth_form", clear_on_submit=False):
            identifier_label = "Email" if mode == "Register" else "Email"
            identifier = st.text_input(identifier_label, placeholder="you@example.com")
            
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button(mode, use_container_width=True)

        if submit:
            endpoint = "/auth/login" if mode == "Login" else "/auth/register"
            payload = {"identifier": identifier, "password": password} if mode == "Login" else {"email": identifier, "password": password}
            r = api_post(endpoint, json=payload)

            if r and r.status_code == 200:
                # This is the success path, it remains the same
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.is_admin = data.get("is_admin", False)
                
                h = api_get("/health")
                if h and h.status_code == 200:
                    me = h.json().get("me", {})
                    st.session_state.username = me.get("username", "there")
                
                st.session_state.view = "admin" if st.session_state.is_admin else "chat"
                st.rerun()
            else:
                # --- MODIFIED ERROR HANDLING BLOCK ---
                # This block now correctly handles different types of errors.
                if r is None:
                    # This happens if api_post fails completely (e.g., backend is down)
                    st.error("Server connection error. Is the backend running?")
                else:
                    # The backend responded, but with an error (e.g., status 400, 401)
                    try:
                        # Try to get the specific detail message from the JSON response
                        error_data = r.json()
                        detail = error_data.get("detail", "An unknown authentication error occurred.")
                        st.error(detail)
                    except requests.exceptions.JSONDecodeError:
                        # If the response isn't valid JSON, show a generic error
                        st.error(f"An unexpected error occurred. Status code: {r.status_code}")
                # --- END OF MODIFICATION ---
    st.stop()