# frontend/api.py

import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def auth_headers():
    """Returns authorization headers if a token is available in session state."""
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

def api_get(path: str, **kwargs):
    """Performs a GET request to the API."""
    try:
        return requests.get(f"{API_BASE}{path}", headers=auth_headers(), timeout=30, **kwargs)
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None

def api_post(path: str, json=None, files=None, params=None, timeout=600):
    """Performs a POST request to the API."""
    try:
        return requests.post(f"{API_BASE}{path}", headers=auth_headers(), json=json, files=files, params=params, timeout=timeout)
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return None

def api_patch(path: str, json=None, timeout=60):
    """Performs a PATCH request to the API."""
    try:
        return requests.patch(f"{API_BASE}{path}", headers=auth_headers(), json=json, timeout=timeout)
    except Exception as e:
        st.error(f"PATCH {path} failed: {e}")
        return None

def api_delete(path: str, timeout=60):
    """Performs a DELETE request to the API."""
    try:
        return requests.delete(f"{API_BASE}{path}", headers=auth_headers(), timeout=timeout)
    except Exception as e:
        st.error(f"DELETE {path} failed: {e}")
        return None

def refresh_sessions():
    """Fetches the list of user sessions from the API and updates the session state."""
    r = api_get("/sessions")
    st.session_state.sessions = r.json() if (r is not None and r.status_code == 200) else []

def refresh_sessions_and_select_latest_if_needed():
    """Refreshes the session list and selects the most recent one if no session is currently selected."""
    refresh_sessions()
    if not st.session_state.session_id and st.session_state.sessions:
        st.session_state.session_id = st.session_state.sessions[0]["id"]