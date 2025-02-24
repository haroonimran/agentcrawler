import streamlit as st
from get_ollama_models import get_available_ollama_models

def model_selection():
    LLM_MODEL = st.sidebar.selectbox("Select Model", get_available_ollama_models())
    st.sidebar.write(f"Selected Model: {LLM_MODEL}")
    return LLM_MODEL