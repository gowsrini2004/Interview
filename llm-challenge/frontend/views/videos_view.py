# frontend/views/videos_view.py

import streamlit as st
from api import api_get, api_post, api_delete

def render_videos_page():
    """Renders the video library, including admin controls if applicable."""
    st.subheader("🎬 Video Library")
    
    if st.button("⬅ Back to Dashboard", key="videos_back"):
        st.session_state.view = "chat"
        st.rerun()

    if st.session_state.is_admin:
        render_admin_video_form()

    q = st.text_input("Search videos (title / description / tags)", key="video_search", placeholder="e.g., anxiety, breathing")
    params = {"q": q} if q else None
    
    with st.spinner("Loading videos…"):
        rv = api_get("/videos", params=params)
    videos = rv.json() if rv and rv.status_code == 200 else []

    if not videos:
        st.info("No videos found.")
        return
    
    render_video_grid(videos)

def render_admin_video_form():
    """Renders the form for admins to add a new video."""
    with st.expander("➕ Add a YouTube Video (Admin)", expanded=False):
        with st.form("add_video_form", clear_on_submit=True):
            payload = {
                "title": st.text_input("Title"),
                "url": st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=..."),
                "description": st.text_area("Description (optional)", height=80),
                "tags_str": st.text_input("Tags (comma-separated, optional)", placeholder="stress, breathing, sleep"),
                "is_public": st.checkbox("Public", value=True)
            }
            submitted = st.form_submit_button("Add Video")
        
        if submitted:
            payload["tags"] = [t.strip() for t in (payload["tags_str"] or "").split(",") if t.strip()]
            del payload["tags_str"]
            r = api_post("/admin/videos", json=payload)
            if r and r.status_code == 200:
                st.success("Video added.")
            else:
                st.error(r.json().get("detail", "Failed to add video") if r else "Server error")

def render_video_grid(videos):
    """Renders a list of videos in a responsive grid."""
    cols_per_row = 2
    for i in range(0, len(videos), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(videos): break
            v = videos[idx]
            with col:
                st.video(v["url"])
                st.markdown(f"**{v['title']}**")
                if v.get("description"):
                    st.caption(v["description"])
                if v.get("tags"):
                    tags_html = " ".join([f"<span style='padding:2px 8px;border:1px solid #444;border-radius:12px;font-size:12px;'>{t}</span>" for t in v["tags"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                
                if st.session_state.is_admin:
                    if st.button("Delete", key=f"del_{v['id']}", use_container_width=True):
                        api_delete(f"/admin/videos/{v['id']}")
                        st.rerun()