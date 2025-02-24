import requests
import chromadb
import uuid
import json
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from get_ollama_models import model_selection


# ------------- Configuration -------------
PERSISTENT_DB_PATH = "./chroma_data"  # Adjust as needed
CHROMA_COLLECTION = "my_collection1"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = model_selection()

# Currently the variable RESET_COLLECTION is false by default, and stays so throughout.
RESET_COLLECTION = False

# Parameter to control chunk size (number of characters per chunk)
CHUNK_SIZE = 2000  # Adjust as needed

# Static instructions to prefix each user prompt
STATIC_PROMPT = (
    "Background: You are an expert reviewing available documents and answering questions about A.R.Rahman."
   # "Only respond to questions that are about Pydantic by searching through the documents provided to you in the context."
    #"If you dont know the answer to a question, be honest and admit that you dont know." 
    #"Do not attempt to answer questions when you are unable to derive a clear contextual understanding based on the user prompt and documents."
)

# Ollama endpoints (embedding vs. completions)
OLLAMA_HOST_EMBED = "http://localhost:11434/api"
OLLAMA_HOST = "http://localhost:11434/v1/completions"

OLLAMA_EMBEDDING_ENDPOINT = f"{OLLAMA_HOST_EMBED}/embeddings"
OLLAMA_LLM_ENDPOINT = f"{OLLAMA_HOST}"
# -----------------------------------------





# When SLC.1 from streamlitchat.py detects the "Crawl URL" button is pressed:
def crawl_and_embed(url: str,keyword_for_supplemental_urls: str):
    """
    Crawls the specified URL, embeds its content, and then crawls and embeds
    the content of every link on that page containing the supplemetal links keyword specified by the user on the Streamlit fron end.
    """
    st.info(f"Crawling main URL: {url}")
    
    # Crawl and embed the main page
    main_text = simple_crawl(url)
    if main_text:
        process_chunks(main_text, url)
    else:
        st.error("No content extracted from the main URL.")
    
    # Extract links containing the keyword and crawl them
    filtered_links = get_filtered_links(url, keyword_for_supplemental_urls)
    # If the user leaves the keyword blank display a messge, and  print the st.success() message.
    if filtered_links == None:
        st.info("No keyword provided -- No additional URLs crawled")
    # If a keyword was provided by the user, crawl and embed the additional URLs.
    else:    
        st.info(f"Found {len(filtered_links)} links containing '{keyword_for_supplemental_urls}'.")
        
        for link in filtered_links:
            st.info(f"Crawling filtered link: {link}")
            link_text = simple_crawl(link)
            if link_text:
                process_chunks(link_text, link)
            else:
                st.error(f"No content extracted from {link}.")
    
    st.success("Website embedding complete. Enter your query now.")

# Crawl and Embed Additional links based on a keyword specified by the user via the front-end.
def get_filtered_links(url: str, keyword_for_supplemental_urls: str) -> list:
    """
    Extracts and returns a list of absolute URLs from the page at 'url'
    that contain the specified keyword.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        if keyword_for_supplemental_urls == "":
            return None
        else:
            for a_tag in soup.find_all("a", href=True):
                link = urljoin(url, a_tag["href"])
                if keyword_for_supplemental_urls.lower() in link.lower():
                    links.append(link)
            return links
    except Exception as e:
        st.error(f"Error retrieving links from {url}: {e}")
        return []


""" Haroon Imran 15-Feb-2025 : Need to enhance this to crawl all urls using a sitemap"""
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


def retrieve_context(embedding, n_results=5):
    """
    Performs a similarity search in ChromaDB using the query embedding.
    Returns the top matching documents.
    """
    try:
        collection = get_chroma_collection()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # No "ids"
        )
        if results and "documents" in results and len(results["documents"]) > 0:
            return results["documents"][0]
        return []
    except Exception as e:
        st.error(f"Error retrieving context from ChromaDB: {e}")
        return []


def build_augmented_prompt(user_prompt: str, context_docs: list) -> str:
    """
    Combines the static instructions, retrieved context, and the user prompt into an augmented prompt.
    """
    # Start with static instructions
    prompt = STATIC_PROMPT + "\n\n"
    try:
        if context_docs:
            context_text = "\n".join(context_docs)
            prompt += f"Context:\n{context_text}\n\n"
            prompt += f"Query: {user_prompt}"
        
            with st.container(border=True,height=200):
                st.success(prompt)
            return prompt
        raise Exception("A retreived context does not exist for your Agent")
    except Exception as e:
         st.error(f"Retreived Context is missing.: {e}")
   


def stream_llm_response(prompt: str):
    """
    Calls the Ollama LLM endpoint with stream=True.
    Expected streaming lines are in the form: 
      data: {...JSON...}
    or 
      data: [DONE]

    This function parses out the JSON and yields choices[0]["text"] for a readable response.
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
                if not line_str.startswith("data: "):
                    continue
                content = line_str[len("data: "):].strip()
                if content == "[DONE]":
                    break
                try:
                    data = json.loads(content)
                    choices = data.get("choices", [])
                    if choices and "text" in choices[0]:
                        yield choices[0]["text"]
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield f"\n[Error streaming LLM response: {e}]"





def chunk_text(text: str, chunk_size: int) -> list:
    """
    Splits the given text into a list of chunks of size 'chunk_size' (in characters).
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

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





