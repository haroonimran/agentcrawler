

import uuid
import chromadb
import streamlit as st

# ------------- Configuration -------------
PERSISTENT_DB_PATH = "./chroma_data"  # Adjust as needed
CHROMA_COLLECTION = "my_collection1"
# Currently the variable RESET_COLLECTION is false by default, and stays so throughout.
RESET_COLLECTION = False

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
            st.info(f"Collection '{CHROMA_COLLECTION}' reset (forced deletion) Failed due to exception.")
        return client.create_collection(name=CHROMA_COLLECTION, embedding_function=None)

    try:
        return client.get_collection(CHROMA_COLLECTION)
    except:
        st.info(f"Collection '{CHROMA_COLLECTION}' not found. Creating a new one...")
        return client.create_collection(name=CHROMA_COLLECTION, embedding_function=None)

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
        # For user prompts, we show an immediate success message.
        if source == "user":
            st.success("User prompt added to the database.")
    except Exception as e:
        st.error(f"Error adding document to the database: {e}")