# frontend/views/chat_view.py

import streamlit as st
import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from api import api_get, api_post, auth_headers, API_BASE, refresh_sessions_and_select_latest_if_needed
from helpers import split_reply_and_sources, get_session_by_id
from session import open_modal

LOCAL_TZ = ZoneInfo("Asia/Kolkata")

def render_chat_page():
    """Renders the main chat interface, including messages and input handling."""
    st.subheader("💬 New Chat" if not st.session_state.session_id else "💬 Conversation")
    render_chat_header()
    render_message_history()
    render_chat_input()

def render_chat_header():
    """Renders the editable title and delete button for a conversation."""
    if st.session_state.session_id:
        s = get_session_by_id(st.session_state.session_id)
        current_title = (s["title"] if s else "") or "Chat"

        cA, cB, cC = st.columns([6, 1, 1])
        with cA:
            new_title = st.text_input("Title", value=current_title, label_visibility="collapsed", key=f"title_{st.session_state.session_id}")
        with cB:
            if st.button("Save", key=f"save_title_{st.session_state.session_id}"):
                # In the refactored backend, this is a PATCH request
                from api import api_patch
                r = api_patch(f"/sessions/{st.session_state.session_id}", json={"title": new_title})
                st.rerun()
        with cC:
            if st.button("Delete", key=f"delete_chat_{st.session_state.session_id}"):
                open_modal("delete_conversation", {"session_id": st.session_state.session_id})
                st.rerun()

def render_message_history():
    """Fetches and displays the messages for the current conversation."""
    if not st.session_state.session_id:
        st.info("This is a fresh chat. Your first message will create a chat history automatically.")
        return

    msgs = api_get(f"/sessions/{st.session_state.session_id}/messages")
    messages = msgs.json() if msgs and msgs.status_code == 200 else []

    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        dt = datetime.fromisoformat(m.get("created_at", datetime.utcnow().isoformat()))
        ts = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")

        with st.chat_message(role):
            if role == "user":
                st.markdown(text)
            else:
                clean, items = split_reply_and_sources(text)
                st.markdown(clean)
                
                # --- NEW, CLEARER SOURCE FORMATTING ---
                if items:
                    with st.expander("📚 Sources", expanded=False):
                        for item in items:
                            with st.container():
                                st.markdown(f"**Source [{item.get('index', 'N/A')}]**")
                                st.markdown(
                                    f"""
                                    > {item.get('excerpt', 'No excerpt available.')}
                                    """
                                )
                                st.caption(f"File: {item.get('filename', 'Unknown')}")
                                st.markdown("---")
                # --- END OF NEW FORMATTING ---

            st.caption(ts)


def render_chat_input():
    """Renders the correct chat input box (normal vs. questionnaire)."""
    is_q_session = st.session_state.q_session_id and st.session_state.session_id == st.session_state.q_session_id
    prompt = "Answer (Questionnaire)…" if is_q_session else "Type your message…"
    user_msg = st.chat_input(prompt, key=f"chat_input_{st.session_state.session_id}")

    if user_msg:
        if is_q_session:
            handle_questionnaire_answer(user_msg)
        else:
            handle_normal_chat(user_msg)

def handle_questionnaire_answer(answer):
    """Submits a questionnaire answer and handles the response."""
    with st.chat_message("user"):
        st.markdown(answer)
    r = api_post("/questionnaire/answer", json={"session_id": st.session_state.q_session_id, "answer": answer})
    if r and r.status_code == 200:
        data = r.json()
        if data.get("done"):
            st.session_state.q_session_id = None
            # Do not clear session_id, just refresh to show final message
        st.rerun()
    else:
        st.error("Failed to submit answer.")


def handle_normal_chat(user_msg):
    """Submits a normal chat message and handles streaming or non-streaming responses."""
    with st.chat_message("user"):
        st.markdown(user_msg)
    
    if st.session_state.streaming:
        stream_chat(user_msg)
    else:
        post_chat(user_msg)
    
    if not st.session_state.session_id:
        refresh_sessions_and_select_latest_if_needed()
    st.rerun()

def stream_chat(user_msg):
    """Handles a streaming chat response."""
    ph = st.chat_message("assistant")
    with ph:
        text_area = st.empty()
        acc = ""
        try:
            with requests.post(
                f"{API_BASE}/chat_stream", headers=auth_headers(),
                json={"message": user_msg, "session_id": st.session_state.session_id, "provider": st.session_state.provider},
                stream=True, timeout=600,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line: continue
                    try:
                        ev = json.loads(line)
                        if ev.get("type") == "delta":
                            acc += ev.get("text", "")
                            clean, _ = split_reply_and_sources(acc)
                            text_area.markdown(clean + " ▌")
                        elif ev.get("type") == "done":
                            break
                    except json.JSONDecodeError: continue
            clean, _ = split_reply_and_sources(acc)
            text_area.markdown(clean)
        except Exception as e:
            st.error(f"Streaming failed: {e}")

def post_chat(user_msg):
    """Handles a non-streaming chat response."""
    r = api_post("/chat", json={"message": user_msg, "session_id": st.session_state.session_id, "provider": st.session_state.provider})
    if not (r and r.status_code == 200):
        st.error(r.json().get("detail", "Server error") if r else "Server connection error")