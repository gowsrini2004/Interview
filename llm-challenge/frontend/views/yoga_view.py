# frontend/views/yoga_view.py

import streamlit as st

def render_yoga_page():
    """Renders the Yoga page with video selections."""
    st.title("🧘 Yoga Studio")

    if st.button("⬅ Back to Dashboard", key="yoga_back"):
        st.session_state.view = "chat"
        st.rerun()

    yoga_types = ["Basic Asanas", "Advanced Asanas", "Surya Namaskar", "Pranayama Breathing", "Yoga for Flexibility"]
    selection = st.selectbox("Choose Yoga Type", yoga_types, key="yoga_type_select")

    yoga_videos = {
        "Basic Asanas": [{"title": "Basic Yoga for Beginners", "url": "https://www.youtube.com/watch?v=v7AYKMP6rOE", "description": "Learn simple yoga poses suitable for beginners."},],
        "Advanced Asanas": [{"title": "Advanced Yoga Poses", "url": "https://youtu.be/nY1z7LXhmGc?si=Ig4b7s5mwTk9qrQe", "description": "Challenging yoga poses for experienced practitioners."},],
        "Surya Namaskar": [{"title": "Sun Salutation (Surya Namaskar)", "url": "https://youtu.be/38GTnjg_aBA?si=AlGjmiSxfGzB6PWQ", "description": "A full Surya Namaskar routine to energize your morning."},],
        "Pranayama Breathing": [{"title": "Pranayama Breathing Techniques", "url": "https://youtu.be/395ZloN4Rr8?si=lbbErJ8eXN43JrfC", "description": "Learn controlled breathing exercises for relaxation."},],
        "Yoga for Flexibility": [{"title": "Full Body Yoga Stretch", "url": "https://youtu.be/uKVAT9sghQ0?si=1B0jFLUZmYOtJv1W", "description": "Yoga routine to increase overall flexibility."},]
    }

    st.markdown(f"### Videos for: **{selection}**")
    for vid in yoga_videos.get(selection, []):
        st.markdown(f"**{vid['title']}**")
        st.video(vid["url"])
        st.markdown(f"*{vid['description']}*")
        st.markdown("---")