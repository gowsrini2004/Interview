# frontend/views/dashboard_view.py

import streamlit as st
import pandas as pd
from api import api_get

def render_dashboard_page():
    """Renders the questionnaire progress dashboard for the user."""
    st.subheader("📊 Your Questionnaire Progress")
    with st.spinner("Loading dashboard…"):
        r = api_get("/questionnaire/dashboard")
    
    data = r.json() if r and r.status_code == 200 else []
    
    if not data:
        st.info("No questionnaires taken yet. Start one from the sidebar.")
    else:
        df = pd.DataFrame(data)
        df.rename(columns={
            "day": "Date",
            "attempts": "Attempts",
            "avg_score": "Avg Score",
            "latest_comment": "Latest Comment"
        }, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)