import streamlit as st

st.set_page_config(
    page_title="A²ML Autonomous Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
try:
    with open("src/ui/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Sidebar Navigation
st.sidebar.title("A²ML System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["System Dashboard", "ML Knowledge Base"])

import src.ui.dashboard as dashboard
import src.ui.ml_knowledge as ml_knowledge

if page == "System Dashboard":
    dashboard.render_dashboard()
elif page == "ML Knowledge Base":
    ml_knowledge.render_knowledge()
