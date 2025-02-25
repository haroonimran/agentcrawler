
import requests
import streamlit as st


EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_HOST_EMBED = "http://localhost:11434/api"
OLLAMA_EMBEDDING_ENDPOINT = f"{OLLAMA_HOST_EMBED}/embeddings"

def get_embedding(text: str):
    """
    Calls the Ollama embedding endpoint to get an embedding vector for the text.
    """
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    try:
        response = requests.post(url=OLLAMA_EMBEDDING_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if embedding is None:
            st.error("Embedding model did not return an embedding.")
            return None
        return embedding
    except Exception as e:
        st.error(f"Error obtaining embedding: {e}")
        return None



