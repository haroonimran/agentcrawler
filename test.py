from __future__ import annotations
import asyncio
import streamlit as st
import httpx
from typing import Literal, TypedDict
import chromadb
import logfire

# (If you still need these imports for other reasons, keep them)
# from chromadb import Client
# from openai import AsyncOpenAI
# from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.messages import (...)
# from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

from dotenv import load_dotenv
load_dotenv()

# If you still want a persistent Chroma client, you can keep it:
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Suppress logfire warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """
    Format of messages in st.session_state.messages, or in the UI.
    Each message has a role ('user' or 'assistant'), a timestamp, and the content.
    """
    role: Literal['user', 'assistant', 'system']
    timestamp: str
    content: str

def display_message(chat_msg: ChatMessage):
    """
    Render a ChatMessage in Streamlit's UI.
    """
    if chat_msg["role"] == "system":
        with st.chat_message("system"):
            st.markdown(f"**System**: {chat_msg['content']}")
    elif chat_msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(chat_msg["content"])
    else:
        # Assistant
        with st.chat_message("assistant"):
            st.markdown(chat_msg["content"])

async def ollama_stream(prompt: str, model_name: str = "llama3.1:latest"):
    """
    Call Ollama's /generate endpoint in streaming mode.
    Yields partial text chunks as they arrive.
    """
    url = "http://localhost:11434/v1/completions" # default Ollama port & endpoint
    payload = {
        "prompt": prompt,
        "model": model_name,
        "stream": True, # to get chunked output
        # "stop": ["###"], # example of a stop token if you want
        # any other Ollama parameters you might need ...
    }

    async with httpx.AsyncClient() as client:
        # POST to /generate with JSON
        async with client.stream("POST", url, json=payload) as response:
            # Check if success
            response.raise_for_status()

            # Ollama streams partial tokens line by line
            async for line in response.aiter_lines():
                # Each line is typically a JSON object or partial chunk
                # Ollama's output lines often look like:
                # {"type":"token","data":"Hello"} 
                # {"type":"done","data":""}
                # ...
                # We need to parse only when type=token to get the text
                if not line.strip():
                    continue
                try:
                    data = httpx.JsonDecodeError
                    data = httpx._models.json.loads(line)
                    if data.get("type") == "token":
                        yield data["data"]
                except Exception:
                    # If parsing fails, skip this chunk
                    pass

async def main():
    st.title("Locally Hosted Ollama Chat")
    st.write("Ask any question about pydantic-ai, or anything else—powered by Ollama locally.")

    # Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for msg in st.session_state.messages:
        display_message(msg)

    # A Chat input box
    user_input = st.chat_input("What do you want to ask?")

    # If user types something, handle it
    if user_input:
        # 1) Save user message
        st.session_state.messages.append({
            "role": "user",
            "timestamp": "",
            "content": user_input
        })
        # Display user message right away
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) Now we want to stream the assistant's response from Ollama
        # We'll display partial text as it arrives.
        partial_text = ""
        message_placeholder = st.empty() # for partial tokens

        with st.chat_message("assistant"):
            # Stream tokens from Ollama
            async for chunk in ollama_stream(prompt=user_input, model_name="llama3.1:latest"):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

        # 3) Once done streaming, store the final assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "timestamp": "",
            "content": partial_text
        })

if __name__ == "__main__":
    asyncio.run(main())