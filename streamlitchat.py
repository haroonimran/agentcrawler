# Library Imports
import streamlit as st
import time

# Imports for user defined functions
from ragmain import crawl_and_embed,build_augmented_prompt,stream_llm_response,retrieve_context
from insertdata import add_embedding_to_db
from embed import get_embedding

# Haroon Imran 24-Feb-2025:
# streamlit.py is where it all starts. Execute this file in the terminal to run the application.
# command:  >>            streamlit run /home/imran/llmbox/agentcrawler/streamlitchat.py


def main():
    st.title("Local LLM Chat with RAG (Chunked Data & Human-Readable Streaming)")

    # SLC.1 Sidebar: Web Crawler
    st.sidebar.header("Web Crawler")
    crawl_url = st.sidebar.text_input("Enter URL to crawl:")
    keyword = st.sidebar.text_input("Enter keyword filter for additinal URLs")
    if st.sidebar.button("Crawl URL"):
        if crawl_url:
            crawl_and_embed(crawl_url,keyword)
        else:
            st.sidebar.error("Error! URL to be Crawled not entered! Enter URL and try again.")


    # Main: Chat Interface
    st.header("Ask the LLM (RAG)")
    prompt = st.text_input("Enter your query or prompt:")
    if st.button("Submit Query") and prompt:
        st.info("Processing your prompt...")
        embedding = get_embedding(prompt)
        if not embedding:
            st.error("Failed to obtain embedding. Aborting.")
            return

        add_embedding_to_db(prompt, embedding, source="user")
        context_docs = retrieve_context(embedding, n_results=30)
        if context_docs:
            st.markdown("#### Retrieved Context:")
            for idx, doc in enumerate(context_docs, 1):
                with st.container(border=True,height=200):
                    st.text(f"**Doc {idx}:** {doc}")
        else:
            st.warning("No similar context found.")

        # Build augmented prompt with static instructions
        augmented_prompt = build_augmented_prompt(prompt, context_docs)
        with st.container(border=True,height=200):
            st.text(augmented_prompt)
        
        st.markdown("### LLM Response:")
        response_placeholder = st.empty()
        full_response = ""

        for chunk in stream_llm_response(augmented_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response)
            time.sleep(0.05)

        st.success("Response complete.")

    

if __name__ == "__main__":
    main()
