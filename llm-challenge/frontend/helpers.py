# frontend/helpers.py

import os
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

SOURCES_MARKER_PREFIX = "<!--SOURCES_JSON:"
SOURCES_MARKER_SUFFIX = "-->"
LOCAL_TZ = ZoneInfo("Asia/Kolkata")

def _now_ts() -> int:
    """Returns the current Unix timestamp."""
    return int(time.time())

def human_time(ts_iso: str) -> str:
    """Converts an ISO timestamp string to a human-readable format."""
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_iso

def split_reply_and_sources(text: str) -> tuple[str, list]:
    """
    Splits the AI's reply from the hidden sources JSON marker.
    This version is robust against leading/trailing whitespace.
    """
    original_text = text
    text_to_process = (text or "").strip()

    if not text_to_process:
        return original_text, []

    start = text_to_process.rfind(SOURCES_MARKER_PREFIX)
    if start == -1:
        return original_text, []

    end = text_to_process.find(SOURCES_MARKER_SUFFIX, start)
    if end == -1:
        return original_text, []

    try:
        payload = text_to_process[start + len(SOURCES_MARKER_PREFIX): end]
        items = json.loads(payload)
        clean = text_to_process[:start].rstrip()
        return clean, items if isinstance(items, list) else []
    except Exception:
        return original_text, []

def get_session_by_id(sid: str) -> dict | None:
    """Finds a session dictionary by its ID from the session state list."""
    for s in st.session_state.sessions:
        if s["id"] == sid:
            return s
    return None