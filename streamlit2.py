import streamlit as st
import requests
import chromadb
import uuid
import time

# ------------- Configuration -------------
# Path to your persistent ChromaDB database directory
PERSISTENT_DB_PATH = "./chroma_data"  # Adjust as needed
# Name of the collection to use (it must already exist in your persistent DB)
CHROMA_COLLECTION = "my_collection"

# The model names as configured in your Ollama setup
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.1:latest"

# Base URL for your locally hosted Ollama endpoints.
# Adjust the host/port if necessary.
OLLAMA_HOST_EMBED = "http://localhost:11434/api"
OLLAMA_HOST = "http://localhost:11434/v1/completions"

# Endpoints for embedding and LLM (adjust the paths as required)
OLLAMA_EMBEDDING_ENDPOINT = f"{OLLAMA_HOST_EMBED}/embeddings"
OLLAMA_LLM_ENDPOINT = f"{OLLAMA_HOST}/llm"
# -----------------------------------------

def get_embedding(prompt: str):
    """
    Calls the Ollama embedding endpoint to get an embedding vector for the prompt.
    Expects a JSON response with an "embedding" field.
    """
    payload = {"model": EMBEDDING_MODEL, "prompt": prompt}
    try:
        response = requests.post(OLLAMA_EMBEDDING_ENDPOINT, json=payload)
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

def add_embedding_to_db(prompt: str, embedding):
    """
    Connects to the persistent ChromaDB database and adds the prompt and its embedding.
    """
    try:
        client = chromadb.PersistentClient(path=PERSISTENT_DB_PATH)
        # Retrieve the existing collection.
        collection = client.get_collection(name=CHROMA_COLLECTION)
        # Generate a unique ID for this prompt
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[prompt],
            embeddings=[embedding],
            metadatas=[{"source": "user"}],
            ids=[doc_id]
        )
        st.success("Prompt and embedding successfully added to the database.")
    except Exception as e:
        st.error(f"Error adding embedding to the database: {e}")

def retrieve_context(embedding, n_results=3):
    """
    Performs a similarity search in the persistent ChromaDB using the query embedding.
    Returns a list of top matching documents to be used as context.
    """
    try:
        client = chromadb.PersistentClient(path=PERSISTENT_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "ids", "distances"]
        )
        # results["documents"] is a list of lists (one list per query)
        context_docs = results["documents"][0]
        return context_docs
    except Exception as e:
        st.error(f"Error retrieving context from ChromaDB: {e}")
        return []

def build_augmented_prompt(user_prompt: str, context_docs: list) -> str:
    """
    Combines the retrieved context and the user prompt into a single augmented prompt.
    """
    if context_docs:
        context_text = "\n".join(context_docs)
        augmented_prompt = (
            f"Use the following context to help answer the query.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Query: {user_prompt}"
        )
    else:
        # If no context is retrieved, fall back to the original user prompt.
        augmented_prompt = user_prompt
    return augmented_prompt

def stream_llm_response(prompt: str):
    """
    Calls the Ollama LLM endpoint with stream=True and yields text chunks as they arrive.
    Adjust the streaming handling as needed depending on your API’s response format.
    """
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": True}
    try:
        with requests.post(OLLAMA_LLM_ENDPOINT, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    # Assuming each line is a text chunk; adjust if your API returns JSON.
                    yield line.decode("utf-8")
    except Exception as e:
        yield f"\n[Error streaming LLM response: {e}]"

def main():
    st.title("Local LLM Chat with RAG via ChromaDB")
    st.markdown(
        """
   You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
        """
    )
    
    prompt = st.text_input("Enter your prompt:")
    if st.button("Submit") and prompt:
        st.info("Processing your prompt...")
        
        # 1. Get the embedding via the Ollama embedding model.
        embedding = get_embedding(prompt)
        if not embedding:
            st.error("Failed to obtain embedding. Aborting further processing.")
            return
        
        # 2. Add the prompt and its embedding to the persistent ChromaDB.
        add_embedding_to_db(prompt, embedding)
        
        # 3. Retrieve context from ChromaDB based on the query embedding.
        context_docs = retrieve_context(embedding, n_results=3)
        if context_docs:
            st.markdown("#### Retrieved Context:")
            for idx, doc in enumerate(context_docs, start=1):
                st.markdown(f"**Doc {idx}:** {doc}")
        else:
            st.warning("No similar context found in the database.")
        
        # 4. Build the augmented prompt using the retrieved context.
        augmented_prompt = build_augmented_prompt(prompt, context_docs)
        
        # 5. Send the augmented prompt to the LLM and stream back the response.
        st.markdown("### LLM Response:")
        response_placeholder = st.empty()  # Container to update with streamed text
        full_response = ""
        for chunk in stream_llm_response(augmented_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response)
            # Small delay to help with UI updates (optional)
            time.sleep(0.1)
        st.success("Response complete.")

if __name__ == "__main__":
    main()
