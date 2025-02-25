import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# imports for user defined functions
from get_ollama_models import model_selection
from insertdata import get_chroma_collection
from chunker import process_chunks


LLM_MODEL = model_selection()

# Static instructions to prefix each user prompt
STATIC_PROMPT = (
    "Background: You are an expert reviewing available documents and answering questions about A.R.Rahman."
   # "Only respond to questions that are about Pydantic by searching through the documents provided to you in the context."
    #"If you dont know the answer to a question, be honest and admit that you dont know." 
    #"Do not attempt to answer questions when you are unable to derive a clear contextual understanding based on the user prompt and documents."
)
# Ollama endpoints (embedding vs. completions)
OLLAMA_HOST = "http://localhost:11434/v1/completions"
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
