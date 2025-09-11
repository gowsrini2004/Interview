# frontend/views/meditation_view.py

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64
from datetime import datetime
import json

# Import the helper function from our new helpers module
from helpers import _now_ts

def render_meditation_page():
    """
    Renders the meditation page with sub-views for "guided" setup and the "room" for practice.
    """
    # --- session keys for meditation state ---
    st.session_state.setdefault("med_page", "guided")
    st.session_state.setdefault("guided_spoken", False)
    st.session_state.setdefault("guided_steps", 5)
    st.session_state.setdefault("room_is_running", False)
    st.session_state.setdefault("room_start_ts", None)
    st.session_state.setdefault("room_end_ts", None)
    st.session_state.setdefault("room_sound_choice", "Omkara (chant)")
    st.session_state.setdefault("room_loop_enabled", True)

    st.title("🧘 Meditation Space")

    if st.button("⬅ Back to Dashboard", key="med_back_v2"):
        st.session_state.view = "chat"
        st.rerun()

    # ---------------- GUIDED PAGE (COMPLETE LOGIC) ----------------
    if st.session_state.med_page == "guided":
        st.header("📖 Guided Meditation — Learn the steps")
        st.video("https://www.youtube.com/watch?v=inpok4MKVLM")
        st.markdown("Select the number of steps, listen to the guidance, then press **Enter Meditation Room** to begin your practice.")
        
        steps = st.slider(
            "Number of steps", 
            min_value=1, 
            max_value=12, 
            value=st.session_state.guided_steps, 
            key="guided_steps_slider"
        )
        st.session_state.guided_steps = steps

        base_prompts = [
            "Breathe in slowly and deeply, then breathe out fully.",
            "Scan your body from head to toes, releasing any tension you find.",
            "Bring your awareness to the breath. Feel the sensation of the belly or chest rising and falling.",
            "If thoughts arise, simply notice them without judgment and gently return your focus to the breath.",
            "Soften your jaw and shoulders, relax your face, and allow the breath to be natural."
        ]
        prompts = [base_prompts[i % len(base_prompts)] for i in range(steps)]

        st.markdown("**Steps you'll practice**")
        for i, p in enumerate(prompts, start=1):
            st.markdown(f"**Step {i}:** {p}")

        if st.button("🔊 Speak Steps", key="speak_steps_btn"):
            # Safely embed the list of prompts into the JavaScript string
            prompts_json = json.dumps(prompts)
            js = f"""
            <script>
            const prompts = {prompts_json};
            const synth = window.speechSynthesis;
            let idx = 0;
            function speakNext() {{
                if(idx >= prompts.length) {{ return; }}
                const u = new SpeechSynthesisUtterance(prompts[idx]);
                u.rate = 0.95;
                u.onend = () => {{ setTimeout(() => {{ idx++; speakNext(); }}, 700); }};
                synth.speak(u);
            }}
            setTimeout(() => speakNext(), 200);
            </script>
            """
            components.html(js, height=0)
            st.success("Speaking steps in your browser...")
            st.session_state.guided_spoken = True

        if st.button("➡️ Enter Meditation Room"):
            st.session_state.med_page = "room"
            st.rerun()

    # ---------------- MEDITATION ROOM (COMPLETE LOGIC) ----------------
    elif st.session_state.med_page == "room":
        st.header("🏯 Meditation Room — Practice")

        BASE_DIR = Path(__file__).resolve().parent.parent 
        SOURCES_DIR = BASE_DIR / "sources"

        sound_map = {
            "Omkara (chant)": SOURCES_DIR / "om.mp3",
            "Natural (forest / rain)": SOURCES_DIR / "rain_sound.mp3",
            "40Hz Tone (focus)": SOURCES_DIR / "gamma.mp3",
        }

        sound_choice = st.selectbox(
            "Choose background sound",
            list(sound_map.keys()),
            key="room_sound_select"
        )
        st.session_state.room_sound_choice = sound_choice
        chosen_path = sound_map.get(sound_choice)
        st.session_state.room_loop_enabled = st.checkbox("🔁 Loop background sound", value=True)

        if not st.session_state.room_is_running:
            if st.button("▶ Start Meditation", key="room_start_btn"):
                st.session_state.room_is_running = True
                st.rerun()
        else:
            st.info(f"Meditation in progress... started at {datetime.fromtimestamp(st.session_state.room_start_ts or _now_ts()).strftime('%H:%M:%S')}")
            
            if st.button("⏹ Stop Meditation", key="room_stop_btn"):
                st.session_state.room_is_running = False
                st.session_state.room_start_ts = None
                components.html("<audio id='bg_audio' muted></audio>", height=0)
                st.rerun()

        if st.session_state.room_is_running and chosen_path and chosen_path.exists():
            try:
                with open(chosen_path, "rb") as f:
                    audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode()
                loop_attr = "loop" if st.session_state.room_loop_enabled else ""
                audio_html = f'<audio autoplay {loop_attr} controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                components.html(audio_html, height=80)
            except Exception as e:
                st.error(f"Failed to load audio: {e}")
        elif st.session_state.room_is_running:
            st.warning("⚠️ Selected audio file not found in frontend/sources/ folder.")