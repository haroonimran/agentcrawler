
import streamlit as st

#imports for user defined functions
from embed import get_embedding
from insertdata import add_embedding_to_db


# Parameter to control chunk size (number of characters per chunk)
CHUNK_SIZE = 2000  # Adjust as needed

def process_chunks(text: str, source_url: str):
    """
    Splits the text into chunks, obtains an embedding for each chunk,
    and adds them to the ChromaDB collection with metadata.
    """
    chunks = chunk_text(text, CHUNK_SIZE)
    if not chunks:
        st.error(f"Failed to split content from {source_url} into chunks.")
        return

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if not embedding:
            st.error(f"Failed to obtain embedding for chunk {idx+1} from {source_url}.")
            continue
        add_embedding_to_db(
            chunk,
            embedding,
            source="crawl",
            extra_metadata={"url": source_url, "chunk_index": idx+1, "total_chunks": len(chunks)}
        )


def chunk_text(text: str, chunk_size: int) -> list:
    """
    Splits the given text into a list of chunks of size 'chunk_size' (in characters).
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
