import streamlit as st
import requests

def query_ollama(prompt, model="llama3.1:latest"):
    """
    Query the locally hosted Ollama LLM and return the response.
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Ensure we get a full response instead of streaming
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            # Return raw text to avoid JSON parsing issues
            return response.text
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Streamlit UI
st.title("Test Local LLM via Ollama")
st.write("Interact with your locally hosted LLM using Ollama.")

# Model selection dropdown
model = st.selectbox("Select Model", ["llama3.1:latest"])

# Text input for user prompt
user_input = st.text_area("Enter your prompt:", height=100)

# Button to generate a response
if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            # Query the LLM and display its response
            response = query_ollama(user_input, model)
            st.text_area("LLM Response:", value=response, height=300)
    else:
        st.warning("Please enter a prompt before generating a response.")
