import os
import json
import requests
import time
import streamlit.components.v1 as components
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from zoneinfo import ZoneInfo
from pathlib import Path
import base64
# ---------- sensible defaults ----------
def ensure_session_defaults():
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("is_admin", False)
    st.session_state.setdefault("provider", "groq")
    st.session_state.setdefault("streaming", True)
    st.session_state.setdefault("view", "chat")         # "chat" | "dashboard" | "videos" | "email"
    st.session_state.setdefault("sessions", [])
    st.session_state.setdefault("messages_cache", {})

if "is_admin" not in st.session_state: st.session_state.is_admin = False
if "is_counsellor" not in st.session_state: st.session_state.is_counsellor = False

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
if "user_id" not in st.session_state: st.session_state.user_id = None



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
def _now_ts():
    return int(time.time())

# ---------------- Full meditation page ----------------
def render_meditation_page():
    """
    Updated meditation page:
     - Guided page: video + step prompts; Speak Steps button uses browser SpeechSynthesis to speak them.
     - After speaking (or directly), user can press Enter Meditation Room (Start) to begin room session.
     - Meditation Room: improved audio element injection with controls & JS play attempt.
    """

    # --- session keys
    st.session_state.setdefault("med_page", None)
    st.session_state.setdefault("view", "chat")

    # Guided state
    st.session_state.setdefault("guided_minutes", 10)
    st.session_state.setdefault("guided_steps", 5)
    # A flag set in-memory (per-run) to indicate step-speech was played. If you navigate away it resets.
    # Use an explicit session key:
    st.session_state.setdefault("guided_spoken", False)

    # Room state
    st.session_state.setdefault("room_start_ts", None)
    st.session_state.setdefault("room_end_ts", None)
    st.session_state.setdefault("room_is_running", False)
    st.session_state.setdefault("room_sound_choice", "Omkara (chant)")
    st.session_state.setdefault("room_minutes", 15)
    
    

    st.title("🧘 Meditation Space")

    # Back
    if st.button("⬅ Back to Dashboard", key="med_back_v2"):
        st.session_state.view = "chat"
        st.rerun()


    # ---------------- GUIDED PAGE ----------------
    if st.session_state.med_page == "guided":
        st.header("📖 Guided Meditation — Learn the steps")

        # Show video (optional)
        st.video("https://www.youtube.com/watch?v=inpok4MKVLM")

        st.markdown("Select duration and number of steps. Then listen to the steps (🔊 Speak Steps). When ready, press **Enter Meditation Room (Start)** to begin your practice with the chosen length & sound.")

        steps = st.slider("Number of steps", min_value=1, max_value=12, value=st.session_state.guided_steps, key="guided_steps_slider")

        st.session_state.guided_steps = steps

        # generate simple step prompts (you can refine text)
        base_prompts = [
            "Breathe in slowly and deeply, then breathe out fully.",
            "Scan your body from head to toes, releasing tension.",
            "Bring awareness to the breath — feel the belly or chest rising.",
            "If thoughts arise, label them and gently return to the breath.",
            "Soften the jaw and shoulders, relax the face, and let the breath be natural."
        ]
        prompts = [base_prompts[(i) % len(base_prompts)] for i in range(steps)]

        st.markdown("**Steps you'll practice**")
        for i, p in enumerate(prompts, start=1):
            st.markdown(f"**Step {i}:** {p}")

        # --- Speak Steps button: uses SpeechSynthesis in the browser
        # It will read each step with a short pause. After speech finishes user can press the Enter button.
        speak_button = st.button("🔊 Speak Steps", key="speak_steps_btn")
        if speak_button:
            # create javascript to speak the steps sequentially
            # We join steps with a small pause; SpeechSynthesis is used (works in modern browsers)
            joined = " <break> ".join(prompts)  # break token for readability; we'll use setTimeout pauses
            # Build JS that speaks each sentence sequentially (with pause) and then sets a visible flag in the DOM.
            # We cannot directly set st.session_state from JS, so we set a simple visual message and set guided_spoken=True on next run.
            js = f"""
            <script>
            const prompts = {prompts};
            // speak each prompt sequentially
            const synth = window.speechSynthesis;
            let idx = 0;
            function speakNext() {{
                if(idx >= prompts.length) {{
                    // done - create a small visible DOM element so user sees it's finished
                    const doneDiv = document.createElement('div');
                    doneDiv.id = 'guided_spoken_done';
                    doneDiv.innerText = '🔊 Steps spoken. You can now enter the Meditation Room below.';
                    doneDiv.style.color = 'green';
                    document.body.appendChild(doneDiv);
                    return;
                }}
                const u = new SpeechSynthesisUtterance(prompts[idx]);
                u.rate = 0.95;
                u.pitch = 1.0;
                u.volume = 1.0;
                u.onend = function(e) {{
                    // small pause then next
                    setTimeout(function(){{ idx++; speakNext(); }}, 700);
                }};
                synth.speak(u);
            }}
            // start speaking after a short delay
            setTimeout(function(){{ speakNext(); }}, 200);
            </script>
            """
            components.html(js, height=40)
            # note: JS runs in embedded component; we also set a server-side flag so UI can show a note
            st.session_state.guided_spoken = True
            st.success("Speaking steps in your browser. Wait for the spoken steps to finish; then press Enter Meditation Room (Start).")

        # Button to go to Meditation Room *and start* the session with these chosen minutes
        if st.button("➡️ Enter Meditation Room (Start)"):
            # Setup room session using guided choices

            # you can choose a default sound if you want
            st.session_state.room_sound_choice = "Omkara (chant)"
            st.session_state.med_page = "room"
            st.rerun()

    # ---------------- MEDITATION ROOM (local only with controlled looping) ----------------
    if st.session_state.med_page == "room":
        st.header("🏯 Meditation Room — Practice")

        # project root = go up from frontend/
        BASE_DIR = Path(__file__).resolve().parent.parent
        SOURCES_DIR = BASE_DIR / "sources"

        # map display names to file paths
        sound_map = {
            "Omkara (chant)": SOURCES_DIR / "om.mp3",
            "Natural (forest / rain)": SOURCES_DIR / "rain_sound.mp3",
            "40Hz Tone (focus)": SOURCES_DIR / "gamma.mp3",
        }

        # select sound
        sound_names = list(sound_map.keys())
        default_index = (
            sound_names.index(st.session_state.room_sound_choice)
            if st.session_state.room_sound_choice in sound_names else 0
        )
        sound_choice = st.selectbox(
            "Choose background sound",
            sound_names,
            index=default_index,
            key="room_sound_select"
        )
        st.session_state.room_sound_choice = sound_choice
        chosen_path = sound_map.get(sound_choice)

        # loop option
        loop_enabled = st.checkbox("🔁 Loop background sound", value=False, key="room_loop_checkbox")

        # ---- Timer / session state ----
        if not st.session_state.room_is_running:
            if st.button("▶ Start Meditation Room", key="room_start_btn"):
                start_ts = _now_ts()
                st.session_state.room_start_ts = start_ts
                st.session_state.room_is_running = True
                st.session_state.room_loop_enabled = loop_enabled
                st.success("Meditation room started.")
                st.info(f"Started at: **{datetime.fromtimestamp(start_ts).strftime('%H:%M:%S')}**")
        else:
            now = _now_ts()
            start_ts = st.session_state.room_start_ts


            

        # ---- Local audio playback (ONLY if running) ----
        if st.session_state.room_is_running and chosen_path and chosen_path.exists():
            st.markdown(f"**Background sound: {sound_choice}**")
            try:
                with open(chosen_path, "rb") as f:
                    audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode()

                loop_attr = "loop" if st.session_state.get("room_loop_enabled", False) else ""

                audio_html = f"""
                <audio autoplay {loop_attr} controls id="bg_audio">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                components.html(audio_html, height=80)
            except Exception as e:
                st.error(f"Failed to load audio: {e}")
        elif st.session_state.room_is_running:
            st.warning("⚠️ Selected audio file not found in sources/ folder.")

        # ---- Controls ----
        if st.button("⏹ Stop Meditation", key="room_stop_btn"):
            st.session_state.room_start_ts = None
            st.session_state.room_end_ts = None
            st.session_state.room_is_running = False
            # stop audio explicitly by rendering a silent <audio> element
            components.html("<audio id='bg_audio'></audio>", height=0)
            st.success("Meditation stopped. To restart, press Start.")
            st.rerun()

        # ---- Auto-finish ----
        if st.session_state.room_is_running and st.session_state.room_end_ts:
            now = _now_ts()
            if now >= st.session_state.room_end_ts:
                st.balloons()
                st.success("🪷 Meditation finished.")
                st.session_state.room_start_ts = None
                st.session_state.room_end_ts = None
                st.session_state.room_is_running = False
                # stop audio at end
                components.html("<audio id='bg_audio'></audio>", height=0)


import streamlit as st

def render_yoga_page():
    """
    Yoga page: user selects type of yoga, displays corresponding videos and description.
    """
    st.title("🧘 Yoga Studio")

    # Back button
    if st.button("⬅ Back to Dashboard", key="yoga_back"):
        st.session_state.view = "chat"
        st.experimental_rerun()

    st.subheader("Select the type of Yoga you want to practice:")

    yoga_types = [
        "Basic Asanas",
        "Advanced Asanas",
        "Surya Namaskar",
        "Pranayama Breathing",
        "Yoga for Flexibility"
    ]

    selection = st.selectbox("Choose Yoga Type", yoga_types, key="yoga_type_select")

    # Define videos and descriptions for each type (multiple videos)
    yoga_videos = {
        "Basic Asanas": [
            {
                "title": "Basic Yoga for Beginners",
                "url": "https://www.youtube.com/watch?v=v7AYKMP6rOE",
                "description": "Learn simple yoga poses suitable for beginners."
            },
            {
                "title": "10-Minute Beginner Yoga",
                "url": "https://www.youtube.com/watch?v=4pKly2JojMw",
                "description": "Quick session for daily beginner practice."
            }
        ],
        "Advanced Asanas": [
            {
                "title": "Advanced Yoga Poses",
                "url": "https://youtu.be/nY1z7LXhmGc?si=Ig4b7s5mwTk9qrQe",
                "description": "Challenging yoga poses for experienced practitioners."
            },
            {
                "title": "Yoga Challenge for Advanced Users",
                "url": "https://youtu.be/MVVi5Ikq2TU?si=ZCawoId93eVXUKeX",
                "description": "Advanced asanas with flow sequences."
            }
        ],
        "Surya Namaskar": [
            {
                "title": "Sun Salutation (Surya Namaskar)",
                "url": "https://youtu.be/38GTnjg_aBA?si=AlGjmiSxfGzB6PWQ",
                "description": "A full Surya Namaskar routine to energize your morning."
            },
            {
                "title": "Surya Namaskar Flow with Breath",
                "url": "https://youtu.be/YAq_oCjnkWY?si=LNg4BCIU01togUAG",
                "description": "Focus on synchronized breathing with Sun Salutations."
            }
        ],
        "Pranayama Breathing": [
            {
                "title": "Pranayama Breathing Techniques",
                "url": "https://youtu.be/395ZloN4Rr8?si=lbbErJ8eXN43JrfC",
                "description": "Learn controlled breathing exercises for relaxation."
            },
            {
                "title": "Advanced Pranayama Practice",
                "url": "https://youtu.be/I77hh5I69gA?si=SIqKr6IBLNi0RA4M",
                "description": "Extended breathing techniques to improve lung capacity."
            }
        ],
        "Yoga for Flexibility": [
            {
                "title": "Full Body Yoga Stretch",
                "url": "https://youtu.be/uKVAT9sghQ0?si=1B0jFLUZmYOtJv1W",
                "description": "Yoga routine to increase overall flexibility."
            },
            {
                "title": "Morning Flexibility Yoga Flow",
                "url": "https://youtu.be/0h7taISrO7c?si=s67aQI7p7j50E_v7",
                "description": "Quick morning session to loosen stiff muscles."
            }
        ]
    }

    st.markdown(f"### Videos for: **{selection}**")
    for vid in yoga_videos.get(selection, []):
        st.markdown(f"**{vid['title']}**")
        st.video(vid["url"])
        st.markdown(f"*{vid['description']}*")
        st.markdown("---")

                
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

def api_request_access_mail(user_id: int, access_type: str):
    return api_post("/counsellor/request-access", {"user_id": user_id, "request_type": access_type})
# user access helpers
def api_user_post_access(json_body):
    return api_post("/user/access", json=json_body)

def api_user_get_access():
    return api_get("/user/access")

def api_create_appointment(json_body):
    return api_post("/appointments", json=json_body)

def api_user_patch_access(access_id, json_body):
    return api_patch(f"/user/access/{access_id}", json=json_body)

def api_my_appointments():
    return api_get("/appointments/my")

def api_delete_appointment(ap_id):
    return api_delete(f"/appointments/{ap_id}")


def api_user_delete_access(access_id):
    return api_delete(f"/user/access/{access_id}")
def api_counsellor_get_access():
    return api_get("/counsellor/access")

def api_counsellor_user_sessions(user_id):
    return api_get(f"/counsellor/users/{user_id}/sessions")

def api_counsellor_user_messages(user_id, session_id):
    return api_get(f"/counsellor/users/{user_id}/sessions/{session_id}/messages")

def api_counsellor_user_dashboard(user_id):
    return api_get(f"/counsellor/users/{user_id}/questionnaire/dashboard")

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
        st.markdown(
            """
            <div style="text-align:center;">
                <h3>🧠 Psych Support — AI Chat with RAG</h3>
                <p style="color: gray; font-size: 14px;">
                    A supportive mental wellness companion with document-aware answers and an adaptive questionnaire.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 👇 wrap login/register in a form
        mode = st.radio(" ", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        with st.form("auth_form", clear_on_submit=False):
            identifier = st.text_input("Email", placeholder="you@example.com")
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
                # st.json(data)
                st.session_state.token = data["access_token"]
                st.session_state.is_admin = data.get("is_admin", False)
                st.session_state.is_counsellor = data.get("is_counsellor", False)
                st.session_state.user_id = data.get("user_id")
                h = api_get("/health")
                if h is not None and h.status_code == 200:
                    me = h.json().get("me", {})
                    st.session_state.username = me.get("username", "there")

                if st.session_state.is_admin:
                    st.session_state.view = "admin"
                elif st.session_state.is_counsellor:
                    st.session_state.view = "counsellor"
                else:
                    st.session_state.view = "chat"
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
            if st.button("👨‍💼 Go to User Management", key="btn_admin_view"):
                st.session_state.view = "admin"
                st.rerun()
            with st.expander("⚙️ Utility", expanded=False):
                st.caption("Role: **Admin**")
                st.markdown("**Vector Store (Qdrant)**")
                st.caption("Delete the entire vector collection and document records.")
                if st.button("🧹 Clear Vector DB", key="btn_clear_vector_db"):
                    open_modal("vector_clear")
                    st.rerun()
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

                if st.button("Send Summary (overall)"):
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

            
            with st.expander("👥 Manage Counsellors", expanded=False):
                if st.button("👨‍⚕️Manage Counsellors", key="btn_manage_counsellors"):
                    st.session_state.view = "manage_counsellors"
                    st.rerun()

            # NOTE: Questionnaire & Your Chats are intentionally hidden for Admin.
        
        # ------------------ COUNSELLOR VIEW ------------------
        elif st.session_state.get("is_counsellor"):
            st.subheader("Counsellor Console")

            with st.expander("⚙️ Utility", expanded=False):
                st.caption("Role: **Counsellor**")

                st.markdown("**Model Provider**")
                st.session_state.provider = st.selectbox(
                    "Choose provider",
                    ["groq", "mistral"],
                    index=0 if st.session_state.get("provider") == "groq" else 1,
                    key="select_provider_counsellor",
                )

                st.checkbox(
                    "Stream responses",
                    value=st.session_state.get("streaming", True),
                    key="streaming_counsellor"
                )

                st.markdown("**Upload Context (.txt)**")
                up = st.file_uploader("Select a .txt file", type=["txt"], key="txt_uploader_counsellor")
                if up and st.button("Upload", key="btn_upload_txt_counsellor"):
                    r = api_post("/ingest", files={"file": up})
                    if r is not None and r.status_code == 200:
                        st.success(f"Ingested {r.json().get('ingested_chunks', 0)} chunks.")
                    else:
                        try:
                            st.error(r.json().get("detail", "Upload failed"))
                        except Exception:
                            st.error("Server error during upload")
            
            with st.expander("👥 User Chats / Questionnaires", expanded=False):
                if st.button("📂 View Chats & Questionnaires", key="btn_view_granted"):
                    st.session_state.view = "granted_users"
                    st.rerun()
            
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

            with st.sidebar.expander("🎬 Video Gallery", expanded=False):
                if st.button("🌍 All Video Gallery", key="btn_counsellor_videos_all", use_container_width=True):
                    st.session_state.view = "counsellor_videos_all"
                    st.rerun()

                if st.button("👤 User Video Gallery", key="btn_counsellor_videos_users", use_container_width=True):
                    st.session_state.view = "counsellor_videos_users"
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



        # ------------------ USER VIEW ------------------
        else:
            # Utility (User)
            with st.expander("⚙️ Utility", expanded=False):
                st.caption("Role: **User**")
                
                if st.button("🔐 Grant Permission to Counsellors", key="btn_grant_access"):
                    st.session_state.view = "grant_access"
                    st.rerun()
                
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

            # Questionnaire (User only)
            with st.expander("🩺 Consultation", expanded=False):
                if st.button("📅 My Appointments"):
                    st.session_state.view = "appointments"
                    st.rerun()
            with st.expander("📊 Questionnaire", expanded=False):
                st.caption("Answer at least 10 adaptive questions. Then continue (+3) or get your score.")
                if st.button("📊 Your Dashboard", key="btn_user_dashboard", use_container_width=True):
                        st.session_state.view = "dashboard"
                        st.rerun()

                # Start Questionnaire button
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
                if st.button("🎬 My Assigned Videos", key="btn_user_videos", use_container_width=True):
                    st.session_state.view = "user_videos"
                    st.rerun()
                    
            with st.sidebar:
                with st.expander("🧘 Meditate", expanded=False):
                    if st.button("🧘 Guided Meditation"):
                        st.session_state.view = "meditation"; st.session_state.med_page = "guided"; st.rerun()
                    if st.button("⛩ Meditation Room"):
                        st.session_state.view = "meditation"; st.session_state.med_page = "room"; st.rerun()
                        
            with st.expander("ૐ Yoga", expanded=False):
                if st.button("🧘Open Yoga Studio", key="sidebar_yoga"):
                    st.session_state.view = "yoga"
                    st.rerun()


           

render_sidebar()

if st.session_state.view == "user_videos":
    if st.button("⬅️ Back to Chatbot", key="back_user_vids"):
        st.session_state.view = "chat"
        st.rerun()

    st.markdown("## 👤 My Assigned Videos")
    q = st.text_input("Search videos (title / description / tags)", key="video_search", placeholder="e.g., anxiety, breathing")
    params = {"q": q} if q else None
    with st.spinner("Loading videos…"):
        rv = api_get("/videos", params=params)
    videos = rv.json() if (rv is not None and rv.status_code == 200) else []
    # st.json(videos)

    my_id = str(st.session_state.user_id)  
    # st.markdown(my_id)# always string
    my_videos = [v for v in videos if v.get("access") == f"user:{my_id}"]

    if not my_videos:
        st.info("No videos assigned to you yet.")
        st.stop()

    cols_per_row = 2
    for i in range(0, len(my_videos), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(my_videos): break
            v = my_videos[idx]
            with col:
                st.video(v["url"])
                st.markdown(f"**{v['title']}**")
                if v.get("description"): st.caption(v["description"])
                if v.get("tags"): st.caption("Tags: " + ", ".join(v["tags"]))
                st.caption(f"Added by: {v.get('added_by', 'unknown')}")
    st.stop()

if st.session_state.view == "appointments":
    st.title("📅 My Appointments")

    # --- Add Appointment ---
    with st.expander("➕ Add Appointment"):
        rc = api_get("/public/counsellors")
        counsellors = rc.json() if rc and rc.status_code == 200 else []
        counsellor_map = {str(c["id"]): c["email"] for c in counsellors}

        sel = st.selectbox("Select Counsellor", options=list(counsellor_map.keys()),
                           format_func=lambda x: counsellor_map.get(x, ""))

        # 👇 Persist date selection
        if "appointment_date" not in st.session_state:
            st.session_state.appointment_date = datetime.now().date()
        date_sel = st.date_input("Date", key="appointment_date")

        # 👇 Persist time selection
        if "appointment_time" not in st.session_state:
            st.session_state.appointment_time = datetime.now().time()
        time_sel = st.time_input("Time", key="appointment_time")

        # 👇 Persist duration
        if "appointment_duration" not in st.session_state:
            st.session_state.appointment_duration = 30
        duration = st.number_input("Duration (minutes)", min_value=15, step=15,
                                   key="appointment_duration")

        if st.button("Book Appointment", key="book_appt_btn"):
            start = datetime.combine(date_sel, time_sel)
            if start <= datetime.now():
                st.error("⛔ You cannot book an appointment in the past.")
            else:
                body = {"counsellor_id": int(sel), "start_time": start.isoformat(),
                        "duration_minutes": duration}
                r = api_create_appointment(body)
                if r and r.status_code == 200:
                    st.success(r.json().get("message", "Appointment created"))
                    st.rerun()
                else:
                    try:
                        st.error(r.json().get("detail", "Failed to create appointment"))
                    except:
                        st.error("Failed to create appointment")

    # --- Tabs for appointments ---
    r = api_my_appointments()
    appts = r.json() if r and r.status_code == 200 else []

    tab_today, tab_future, tab_closed = st.tabs(["Today Appointments", "Future Appointments", "Closed Appointments"])

    today = datetime.now().date()
    now = datetime.now()

    with tab_today:
        st.subheader("Today's Appointments")
        todays = [
            a for a in appts
            if datetime.fromisoformat(a["start_time"]).date() == today and a["status"] != "closed"
        ]
        if not todays:
            st.info("No appointments today.")
        else:
            # Table header
            h2, h3, h5, h6, h7,h9 = st.columns([2, 2, 2, 2, 3,1])
            h2.write("Counsellor")
            h3.write("Start")
            h5.write("Duration")
            h6.write("Status")
            h7.write("Meeting Link")
            h9.write("Action")

            for a in todays:
                c_email = counsellor_map.get(str(a["counsellor_id"]), f"Counsellor {a['counsellor_id']}")
                c2, c3, c5, c6, c7,c9 = st.columns([2, 2,2, 2, 3,1])
                c2.write(c_email)
                c3.write(datetime.fromisoformat(a["start_time"]).strftime("%Y-%m-%d %H:%M"))
                c5.write(f"{a['duration_minutes']} min")
                c6.write(a["status"].capitalize())
                c7.write(a.get("meeting_link") or "—")
                if c9.button("❌", key=f"del_today_{a['id']}"):
                    resp = api_delete_appointment(a["id"])
                    if resp and resp.status_code == 200:
                        st.success("Appointment deleted.")
                    else:
                        st.error("Delete failed.")
                    st.query_params(refresh=str(datetime.now().timestamp()))

        with tab_future:
            st.subheader("Future Appointments")
            future = [
                a for a in appts
                if datetime.fromisoformat(a["start_time"]) > now and a["status"] != "closed"
            ]
            if not future:
                st.info("No future appointments.")
            else:
                # Table header
                h2, h3,h5, h6, h7,h9 = st.columns([2, 2, 2, 2, 3,1])
                h2.write("Counsellor")
                h3.write("Start")
                h5.write("Duration")
                h6.write("Status")
                h7.write("Meeting Link")
                h9.write("Action")

                for a in future:
                    c_email = counsellor_map.get(str(a["counsellor_id"]), f"Counsellor {a['counsellor_id']}")
                    c2, c3, c5, c6, c7,c9 = st.columns([2, 2, 2, 2, 3,1])
                    c2.write(c_email)
                    c3.write(datetime.fromisoformat(a["start_time"]).strftime("%Y-%m-%d %H:%M"))
                    c5.write(f"{a['duration_minutes']} min")
                    c6.write(a["status"].capitalize())
                    c7.write(a.get("meeting_link") or "—")
                    if c9.button("❌", key=f"del_future_{a['id']}"):
                        resp = api_delete_appointment(a["id"])
                        if resp and resp.status_code == 200:
                            st.success("Appointment deleted.")
                        else:
                            st.error("Delete failed.")
                        st.query_params(refresh=str(datetime.now().timestamp()))

    with tab_closed:
        st.subheader("Closed Appointments")
        closed = [a for a in appts if a["status"] == "closed"]
        if not closed:
            st.info("No closed appointments.")
        else:
            # Table header
            h2, h3, h5, h6, h7, h8 = st.columns([2, 2, 2, 2, 3, 3])
            h2.write("Counsellor")
            h3.write("Start")
            h5.write("Duration")
            h6.write("Status")
            h7.write("Meeting Link")
            h8.write("Report")

            for a in closed:
                c_email = counsellor_map.get(str(a["counsellor_id"]), f"Counsellor {a['counsellor_id']}")
                c2, c3, c5, c6, c7, c8 = st.columns([2, 2, 2, 2, 3, 3])
                c2.write(c_email)
                c3.write(datetime.fromisoformat(a["start_time"]).strftime("%Y-%m-%d %H:%M"))
                c5.write(f"{a['duration_minutes']} min")
                c6.write(a["status"].capitalize())
                c7.write(a.get("meeting_link") or "—")
                c8.write(a.get("report") or "—")


    st.stop()

# ---------------- Counsellor: All videos ----------------
if st.session_state.view == "counsellor_videos_all":
    # Back button (small)
    if st.button("⬅️ Back to Chatbot", key="cback_all"):
        st.session_state.view = "counsellor"
        st.rerun()

    st.markdown("## 🌍 All Video Gallery")

    # --- Add new video (visible to all users) ---
    with st.expander("➕ Add Video for All Users", expanded=False):
        with st.form("counsellor_add_common_video", clear_on_submit=True):
            title = st.text_input("Video Title")
            url = st.text_input("YouTube URL")
            desc = st.text_area("Description", height=100)
            tags = st.text_input("Tags (comma-separated)")
            submitted = st.form_submit_button("Add Video")

            if submitted:
                payload = {
                    "title": title,
                    "url": url,
                    "description": desc or "",
                    "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
                    "is_public": True,
                    "target_user_id": None
                }
                resp = api_post("/counsellor/videos", json=payload)
                if resp and resp.status_code in (200, 201):
                    st.success("Video added for all users.")
                    st.rerun()
                else:
                    st.error("Failed to add video.")

    # --- Search ---
    q = st.text_input("Search videos (title / description / tags)", key="counsellor_all_video_search")
    params = {"q": q} if q else None
    rv = api_get("/videos", params=params)
    videos = rv.json() if (rv and rv.status_code == 200) else []
    videos_all = [v for v in videos if v.get("access") == "all"]

    if not videos_all:
        st.info("No videos for all users yet.")
        st.stop()

    # Render as 2-column grid with YouTube players
    cols_per_row = 2
    for i in range(0, len(videos_all), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(videos_all): break
            v = videos_all[idx]
            with col:
                st.video(v["url"])
                st.markdown(f"**{v['title']}**")
                if v.get("description"): st.caption(v["description"])
                if v.get("tags"): st.caption("Tags: " + ", ".join(v["tags"]))
                st.caption(f"Added by: {v.get('added_by', 'unknown')}")
                if st.session_state.get("is_counsellor", False):
                    if st.button("🗑️ Delete", key=f"del_all_{v['id']}"):
                        rr = api_delete(f"/counsellor/videos/{v['id']}") or api_delete(f"/admin/videos/{v['id']}")
                        if rr and rr.status_code in (200, 204):
                            st.success("Deleted.")
                            st.rerun()
                        else:
                            st.error("Delete failed")

    st.stop()


# ---------------- Counsellor: User-specific videos ----------------
if st.session_state.view == "counsellor_videos_users":
    if st.button("⬅️ Back to Chatbot", key="cback_users"):
        st.session_state.view = "counsellor"
        st.rerun()

    st.markdown("## 👤 User Video Gallery")

    # --- Add new video for a user ---
    with st.expander("➕ Add Video for Specific User", expanded=False):
        r = api_get("/counsellor/users")
        user_choices = {u["email"]: u["id"] for u in r.json()} if r and r.status_code == 200 else {}

        with st.form("counsellor_add_user_video", clear_on_submit=True):
            choice = st.selectbox("Select User", list(user_choices.keys())) if user_choices else None
            title = st.text_input("Video Title")
            url = st.text_input("YouTube URL")
            desc = st.text_area("Description", height=100)
            tags = st.text_input("Tags (comma-separated)")
            submitted = st.form_submit_button("Add Video")

            if submitted and choice:
                user_id = user_choices[choice]
                payload = {
                    "title": title,
                    "url": url,
                    "description": desc or "",
                    "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
                    "is_public": False,
                    "target_user_id": int(user_id)   # must be int, not str
                }
                resp = api_post("/counsellor/videos", json=payload)
                if resp and resp.status_code in (200, 201):
                    st.success(f"Video added for {choice}")
                    st.rerun()
                else:
                    st.error("Failed to add video.")

    # --- Search ---
    q = st.text_input("Search user videos", key="counsellor_user_video_search")
    rv = api_get("/videos")
    all_videos = rv.json() if (rv and rv.status_code == 200) else []
    user_vids = [v for v in all_videos if v.get("access", "").startswith("user:")]

    if q:
        ql = q.lower()
        user_vids = [v for v in user_vids if ql in v.get("title", "").lower() or ql in (v.get("description") or "").lower()]

    if not user_vids:
        st.info("No user-specific videos yet.")
        st.stop()

    # Fetch map of user_id → email (use counsellor endpoint)
    users_resp = api_get("/counsellor/users")
    user_map = {str(u["id"]): u["email"] for u in users_resp.json()} if users_resp and users_resp.status_code == 200 else {}

    # --- Render table ---
    st.markdown("### 📋 Videos Assigned to Users")
    header = st.columns([3, 3, 3, 1])
    header[0].markdown("**User**")
    header[1].markdown("**Title**")
    header[2].markdown("**Video**")
    header[3].markdown("**Action**")

    for v in user_vids:
        uid = v["access"].split("user:")[-1]
        uname = user_map.get(uid, f"User {uid}")
        cols = st.columns([3, 3, 3, 1])
        with cols[0]:
            st.write(uname)
        with cols[1]:
            st.write(v["title"])
        with cols[2]:
            st.markdown(f"[Watch Video]({v['url']})")
        with cols[3]:
                if st.session_state.get("is_counsellor", False):
                    if st.button("🗑️", key=f"del_user_{v['id']}"):
                        rr = api_delete(f"/counsellor/videos/{v['id']}") or api_delete(f"/admin/videos/{v['id']}")
                        if rr and rr.status_code in (200, 204):
                            st.success("Deleted.")
                            st.rerun()
                        else:
                            st.error("Delete failed")

    st.stop()
    


if st.session_state.view == "manage_counsellors":
    st.header("👥 Manage Counsellors")

    if st.button("⬅ Back to Users"):
        st.session_state.view = "admin"
        st.rerun()

    # ---- Add Counsellor Expander ----
    with st.expander("➕ Add Counsellor", expanded=True):
        r = api_get("/admin/users/non_counsellors")
        users = r.json() if (r is not None and r.status_code == 200) else []

        if not users:
            st.info("No users available to promote.")
        else:
            u_map = {f"{u['email']} (ID: {u['id']})": u["id"] for u in users}

            # --- Form start ---
            with st.form("add_counsellor_form", clear_on_submit=True):
                choice = st.selectbox("Select user", list(u_map.keys()), key="AddCounsellor_choice")
                name = st.text_input("Counsellor Name", key="AddCounsellor_name")
                desc = st.text_area("Description", height=100, key="AddCounsellor_desc")
                exp = st.number_input("Years of Experience", min_value=0, step=1, key="AddCounsellor_exp")

                submitted = st.form_submit_button("Add Counsellor")
                if submitted:
                    payload = {
                        "user_id": u_map.get(choice),
                        "name": name,
                        "description": desc,
                        "experience_years": exp,
                    }
                    resp = api_post("/admin/counsellors/add", json=payload)
                    if resp is not None and resp.status_code in (200, 201):
                        st.success("User promoted to counsellor.")
                        st.rerun()
                    else:
                        try:
                            st.error(resp.json().get("detail", "Failed to add counsellor"))
                        except Exception:
                            st.error("Failed to add counsellor")
        # --- Form end ---


    # ---- Counsellors List ----
    rc = api_get("/admin/counsellors")
    counsellors = rc.json() if (rc is not None and rc.status_code == 200) else []
    if not counsellors:
        st.info("No counsellors yet.")
    else:
        h1, h2, h3, h4, h5 = st.columns([3, 3, 2, 2, 2])
        with h1: st.markdown("**Name**")
        with h2: st.markdown("**Email**")
        with h3: st.markdown("**Edit**")
        with h4: st.markdown("**Demote**")
        with h5: st.markdown("**Delete**")

        for c in counsellors:
            r1, r2, r3, r4, r5 = st.columns([3, 3, 2, 2, 2])
            with r1:
                st.markdown(f"**{c['name']}** - ({c['experience_years']} yrs exp)")
                st.markdown(f"_{c['description']}_")
            with r2:
                st.markdown(c["email"])
            with r3:
                if st.button("✏️ Edit", key=f"edit_c_{c['id']}"):
                    with st.form(f"edit_c_form_{c['id']}", clear_on_submit=True):
                        name = st.text_input("Name", value=c["name"])
                        desc = st.text_area("Description", value=c["description"])
                        exp = st.number_input("Experience (yrs)", min_value=0, value=c["experience_years"])
                        submitted = st.form_submit_button("Save")
                        if submitted:
                            r = api_patch(f"/admin/counsellors/{c['id']}", json={
                                "name": name, "description": desc, "experience": exp
                            })
                            if r is not None and r.status_code == 200:
                                st.success("Updated.")
                                st.rerun()
                            else:
                                st.error("Failed to update")
            with r4:
                if st.button("⬇ Demote", key=f"demote_c_{c['id']}"):
                    r = api_post(f"/admin/counsellors/{c['id']}/demote")
                    if r is not None and r.status_code == 200:
                        st.success("Demoted to user.")
                        st.rerun()
            with r5:
                if st.button("🗑 Delete", key=f"del_c_{c['id']}"):
                    r = api_delete(f"/admin/users/{c['id']}")
                    if r is not None and r.status_code in (200, 204):
                        st.success("User deleted.")
                        st.rerun()
    st.stop()
    


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
    h1, h2, h3, h4, h5, h6, h7 = st.columns([4,2,2,2,2,3,1])
    with h1: st.markdown("**Email**")
    with h2: st.markdown("**Role**")
    with h3: st.markdown("**Conversations**")
    with h4: st.markdown("**Questionnaires**")
    with h5: st.markdown("**Clear Chats**")
    with h6: st.markdown("**Clear Questionnaire**")
    with h7: st.markdown("**More**")

    # Rows
    # Rows (show Role; hide Clear Questionnaire for counsellors)
    for u in filtered:
        c1, c2, c3, c4, c5, c6, c7 = st.columns([5,2,2,2,3,2,1])
        created = human_time(u["created_at"])
        email_html = f"<span title='Created: {created}'>{u['email']}</span>"

        with c1: st.markdown(email_html, unsafe_allow_html=True)
        with c2: st.markdown(u.get("role", "user"))   # new role column
        with c3: st.markdown(str(u["conversations"]))
        with c4: st.markdown(str(u["questionnaire_attempts"]))

        # Hide questionnaire clear button for counsellors
        with c5:
            if st.button("Clear", key=f"cc_{u['id']}"):
                open_modal("clear_chats", {"user_id": u["id"], "email": u["email"]})
                st.rerun()

        # c6 = Clear Questionnaire
        with c6:
            if u.get("role") != "counsellor":
                if st.button("Clear", key=f"cq_{u['id']}"):
                    open_modal("clear_questionnaires", {"user_id": u["id"], "email": u["email"]})
                    st.rerun()
            else:
                st.caption("NA")

        with c7:
            if st.button("⋯", key=f"menu_{u['id']}"):
                st.session_state.actions_user = {"id": u["id"], "email": u["email"]}
                open_modal("user_actions", {"user_id": u["id"], "email": u["email"]})
                st.rerun()

    
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
    

# -------- Meditate view --------
if st.session_state.view == "meditation":
    render_meditation_page()
    st.stop()
    
if st.session_state.view == "yoga":
    render_yoga_page()
    st.stop()
    
if st.session_state.view == "grant_access":
    st.title("🔐 Grant Access to Counsellors")

    # Load counsellors
    rc = api_get("/public/counsellors")
    counsellors = rc.json() if (rc and rc.status_code == 200) else []
    counsellor_map = {str(c["id"]): c["email"] for c in counsellors}
    
    st.markdown("### Grant permission to a counsellor")
    with st.form("grant_access_form", clear_on_submit=True):
        sel = st.selectbox("Select counsellor", options=list(counsellor_map.keys()),
                           format_func=lambda x: counsellor_map.get(x, ""), key="grant_sel")
        allow_chats = st.checkbox("Allow: Chats", value=False)
        allow_dashboard = st.checkbox("Allow: Questionnaire dashboard", value=False)
        submitted = st.form_submit_button("Grant Access")
        if submitted:
            body = {
                "counsellor_id": int(sel),
                "allow_chats": allow_chats,
                "allow_dashboard": allow_dashboard
            }
            r = api_user_post_access(body)
            if r and r.status_code == 200:
                st.success("Access granted/updated.")
                st.rerun()
            else:
                try: st.error(r.json().get("detail","Failed"))
                except: st.error("Failed")

    st.markdown("---")
    st.markdown("### Your current grants")
    r = api_user_get_access()
    grants = r.json() if (r and r.status_code == 200) else []
    if not grants:
        st.info("No grants yet.")
    else:
        for g in grants:
            cols = st.columns([3,2,1,1])
            cols[0].markdown(f"**{g['counsellor_email']}**")
            cols[1].markdown(f"Chats: {g['allow_chats']}<br>Dashboard: {g['allow_dashboard']}", unsafe_allow_html=True)
            if cols[2].button("Edit", key=f"edit_acc_{g['id']}"):
                new_chats = st.checkbox("Allow Chats", value=g['allow_chats'], key=f"ec_{g['id']}_ch")
                new_dash = st.checkbox("Allow Dashboard", value=g['allow_dashboard'], key=f"ec_{g['id']}_d")
                if st.button("Save", key=f"save_acc_{g['id']}"):
                    patch = {"allow_chats": new_chats, "allow_dashboard": new_dash}
                    rr = api_user_patch_access(g['id'], patch)
                    if rr and rr.status_code == 200:
                        st.success("Updated."); st.rerun()
                    else:
                        st.error("Update failed.")
            if cols[3].button("Revoke", key=f"del_acc_{g['id']}"):
                rr = api_user_delete_access(g['id'])
                if rr and rr.status_code == 200:
                    st.success("Revoked."); st.rerun()
                else:
                    st.error("Failed to revoke")
    st.stop()
   
if st.session_state.view == "granted_users":
    st.title("👥 Users Who Granted You Access")

    if st.button("⬅ Back"):
        st.session_state.view = "counsellor"; st.rerun()

    r = api_counsellor_get_access()
    grants = r.json() if (r and r.status_code == 200) else []

    if not grants:
        st.info("No users have granted you access yet.")
        st.stop()
        
    if st.session_state.get("_selected_user"):
        uid = st.session_state._selected_user
        mode = st.session_state.get("granted_view_mode")
        uname = st.session_state.get("_selected_user_email", "")
        st.markdown(f"#### 🔎 {uname} — {mode.capitalize()}")

        if st.button("⬅ Back to User List"):
            st.session_state._selected_user = None
            st.session_state.granted_view_mode = None
            st.rerun()

        if mode == "chats":
            r = api_counsellor_user_sessions(uid)
            sess = r.json() if (r and r.status_code == 200) else []
            if not sess:
                st.info("No chat sessions found.")
            else:
                for s in sess:
                    with st.expander(f"💬 {s['title'] or 'Chat'} — {s['created_at']}"):
                        msgs_r = api_counsellor_user_messages(uid, s['id'])
                        msgs = msgs_r.json() if (msgs_r and msgs_r.status_code == 200) else []
                        if not msgs:
                            st.info("No messages yet.")
                        else:
                            for m in msgs:
                                role = m.get("role")
                                ts = m.get("created_at", "")
                                if role == "user":
                                    st.markdown(f"**Q:** {m.get('content')}  _(at {ts})_")
                                else:
                                    st.markdown(f"**A:** {m.get('content')}  _(at {ts})_")

        elif mode == "dashboard":
            r = api_counsellor_user_dashboard(uid)
            data = r.json() if (r and r.status_code == 200) else []
            if not data:
                st.info("No dashboard data.")
            else:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("---")

                
    st.markdown("### Your Granted Users")
    for g in grants:
        with st.expander(f"{g['user_email']} (ID: {g['user_id']})", expanded=False):
            cols = st.columns(2)
            with cols[0]:
            # CHATS ACCESS
                if g["allow_chats"]:
                    if st.button("💬 View Chats", key=f"chats_{g['user_id']}"):
                        st.session_state._selected_user = g["user_id"]
                        st.session_state._selected_user_email = g["user_email"]
                        st.session_state.granted_view_mode = "chats"
                        st.rerun()
                else:
                    cols = st.columns([1,1])
                    with cols[1]:
                        st.markdown("❌ Chats: No access")
                    with cols[0]:
                        if st.button("Request Chat Access", key=f"req_chats_{g['user_id']}"):
                            rr = api_request_access_mail(g["user_id"], "chats")
                            if rr and rr.status_code == 200:
                                st.success("Request for chat access sent via email.")
                            else:
                                st.error("Failed to send email request.")
            with cols[1]:
            # DASHBOARD ACCESS
                if g["allow_dashboard"]:
                    if st.button("📊 View Dashboard", key=f"dash_{g['user_id']}"):
                        st.session_state._selected_user = g["user_id"]
                        st.session_state._selected_user_email = g["user_email"]
                        st.session_state.granted_view_mode = "dashboard"
                        st.rerun()
                else:
                    cols = st.columns([1,1])
                    with cols[1]:
                        st.markdown("❌ Dashboard: No access")
                    with cols[0]:
                        if st.button("Request Dashboard Access", key=f"req_dash_{g['user_id']}"):
                            rr = api_request_access_mail(g["user_id"], "dashboard")
                            if rr and rr.status_code == 200:
                                st.success("Request for dashboard access sent via email.")
                            else:
                                st.error("Failed to send email request.")

    # --- DETAIL VIEWS ---
    

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
    if st.button("⬅ Back to Users", key="med_back_v2"):
        st.session_state.view = "chat"
        st.rerun()
    q = st.text_input("Search videos (title / description / tags)", key="video_search", placeholder="e.g., anxiety, breathing")
    params = {"q": q} if q else None
    with st.spinner("Loading videos…"):
        rv = api_get("/videos", params=params)
    videos = rv.json() if (rv is not None and rv.status_code == 200) else []
    videos_all = [v for v in videos if v.get("access") == "all"]

    if not videos_all:
        st.info("No videos found.")
        st.stop()

        # Render as 2-column grid with YouTube players
    cols_per_row = 2
    for i in range(0, len(videos_all), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(videos_all): break
            v = videos_all[idx]
            with col:
                st.video(v["url"])
                st.markdown(f"**{v['title']}**")
                if v.get("description"): st.caption(v["description"])
                if v.get("tags"): st.caption("Tags: " + ", ".join(v["tags"]))
                st.caption(f"Added by: {v.get('added_by', 'unknown')}")
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