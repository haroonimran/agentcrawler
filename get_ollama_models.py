import subprocess
import streamlit as st

def get_available_ollama_models():
    try:
        # Run the command and capture the output as text.
        output = subprocess.check_output(["ollama", "list"], text=True)
        
        # Split the output into lines.
        lines = output.strip().splitlines()
        
        # Assuming the first line is a header, skip it.
        # Also assuming that the model name is the first column.
        models = []
        for line in lines[1:]:
            # Split line into columns (assuming whitespace separation).
            parts = line.split()
            if parts:  # Make sure the line isn't empty.
                models.append(parts[0])
                
        return models
    except Exception as e:
        print(f"Error retrieving models: {e}")
        return []

import streamlit as st


def model_selection():
    LLM_MODEL = st.sidebar.selectbox("Select Model", get_available_ollama_models())
    st.sidebar.write(f"Selected Model: {LLM_MODEL}")
    return LLM_MODEL
