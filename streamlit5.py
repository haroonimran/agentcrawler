import streamlit as st
import requests
import chromadb
import uuid
import time
import json
from bs4 import BeautifulSoup

# ------------- Configuration -------------
PERSISTENT_DB_PATH = "./chroma_data"  # Adjust as needed
CHROMA_COLLECTION = "my_collection"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-R1:7b"

RESET_COLLECTION = False

# Ollama endpoints (embedding vs. completions)
OLLAMA_HOST_EMBED = "http://localhost:11434/api"
OLLAMA_HOST = "http://localhost:11434/v1/completions"

OLLAMA_EMBEDDING_ENDPOINT = f"{OLLAMA_HOST_EMBED}/embeddings"
OLLAMA_LLM_ENDPOINT = f"{OLLAMA_HOST}"
# -----------------------------------------


def get_chroma_collection():
    """
    Returns a persistent ChromaDB collection. 
    No dimension checks or auto-deletion unless RESET_COLLECTION is True.
    """
    client = chromadb.PersistentClient(path=PERSISTENT_DB_PATH)
    if RESET_COLLECTION:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            st.info(f"Collection '{CHROMA_COLLECTION}' reset (forced deletion).")
        except Exception:
            pass
        return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)

    try:
        return client.get_collection(CHROMA_COLLECTION)
    except:
        st.info(f"Collection '{CHROMA_COLLECTION}' not found. Creating a new one...")
        return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)


def get_embedding(text: str):
    """
    Calls the Ollama embedding endpoint to get an embedding vector for the text.
    """
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
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


def add_embedding_to_db(document: str, embedding, source: str = "user", extra_metadata: dict = None):
    """
    Adds a document + embedding to the ChromaDB collection with optional metadata.
    """
    try:
        collection = get_chroma_collection()
        doc_id = str(uuid.uuid4())
        metadata = {"source": source}
        if extra_metadata:
            metadata.update(extra_metadata)

        collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        st.success(f"Document from source '{source}' added to the database.")
    except Exception as e:
        st.error(f"Error adding document to the database: {e}")


def retrieve_context(embedding, n_results=3):
    """
    Performs a similarity search in ChromaDB using the query embedding.
    Returns the top matching documents.
    """
    try:
        collection = get_chroma_collection()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # no "ids"
        )
        if results and "documents" in results and len(results["documents"]) > 0:
            return results["documents"][0]
        return []
    except Exception as e:
        st.error(f"Error retrieving context from ChromaDB: {e}")
        return []


def build_augmented_prompt(user_prompt: str, context_docs: list) -> str:
    """
    Combines retrieved context with the user prompt into an augmented prompt.
    """
    if context_docs:
        context_text = "\n".join(context_docs)
        return (
            f"Use the following context to help answer the query.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Query: {user_prompt}"
        )
    return user_prompt


def stream_llm_response(prompt: str):
    """
    Calls the Ollama LLM endpoint with stream=True.
    The streaming output lines have the form: 
       data: {...JSON...}
    or 
       data: [DONE]

    We parse out the JSON, and then yield choices[0]["text"] to display 
    a readable incremental response.
    """
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True
    }
    try:
        with requests.post(OLLAMA_LLM_ENDPOINT, json=payload, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")

                # We expect lines like "data: {...}" or "data: [DONE]"
                if not line_str.startswith("data: "):
                    # If there's something else, skip or yield raw
                    continue

                # Remove the "data: " prefix to parse the JSON
                content = line_str[len("data: "):].strip()

                # If it's "[DONE]", we stop
                if content == "[DONE]":
                    break

                # Attempt to parse JSON
                try:
                    data = json.loads(content)
                    # The text is in data["choices"][0]["text"]
                    choices = data.get("choices", [])
                    if choices and "text" in choices[0]:
                        yield choices[0]["text"]
                except json.JSONDecodeError:
                    # Not valid JSON, skip or yield raw
                    continue

    except Exception as e:
        yield f"\n[Error streaming LLM response: {e}]"


def simple_crawl(url: str) -> str:
    """
    A simple function to fetch and parse webpage text.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        st.error(f"Error crawling {url}: {e}")
        return ""


def crawl_and_embed(url: str):
    """
    Crawls the URL, embeds the text, and stores it in ChromaDB.
    """
    st.info(f"Crawling URL: {url}")
    text = simple_crawl(url)
    if not text:
        st.error("No content extracted from the URL.")
        return

    embedding = get_embedding(text)
    if not embedding:
        st.error("Failed to obtain embedding for crawled content.")
        return

    add_embedding_to_db(text, embedding, source="crawl", extra_metadata={"url": url})
    st.success("Website embedding complete. Enter your query now.")


def main():
    st.title("Local LLM Chat with RAG (Human-Readable Streaming)")

    # Sidebar web crawler
    st.sidebar.header("Web Crawler")
    crawl_url = st.sidebar.text_input("Enter URL to crawl:")
    if st.sidebar.button("Crawl URL") and crawl_url:
        crawl_and_embed(crawl_url)

    # Main chat interface
    st.header("Ask the LLM (RAG)")
    prompt = st.text_input("Enter your query or prompt:")
    if st.button("Submit Query") and prompt:
        st.info("Processing your prompt...")
        embedding = get_embedding(prompt)
        if not embedding:
            st.error("Failed to obtain embedding. Aborting.")
            return

        add_embedding_to_db(prompt, embedding, source="user")
        context_docs = retrieve_context(embedding, n_results=3)
        if context_docs:
            st.markdown("#### Retrieved Context:")
            for idx, doc in enumerate(context_docs, 1):
                st.markdown(f"**Doc {idx}:** {doc}")
        else:
            st.warning("No similar context found.")

        augmented_prompt = build_augmented_prompt(prompt, context_docs)
        st.markdown("### LLM Response:")
        response_placeholder = st.empty()
        full_response = ""

        # Stream each text chunk in real time
        for chunk in stream_llm_response(augmented_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response)
            time.sleep(0.05)

        st.success("Response complete.")

if __name__ == "__main__":
    main()
