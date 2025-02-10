from __future__ import annotations
import asyncio
import streamlit as st
import httpx
import json
from typing import Literal, TypedDict
import chromadb
import logfire

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
    Call Ollama's /v1/completions endpoint with "stream": True.
    Yields partial text tokens as they arrive.
    """
    url = "http://localhost:11434/v1/completions"
    payload = {
        "prompt": prompt,
        "model": model_name,
        "stream": True,  # ask Ollama to stream partial responses
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # DEBUG: print raw line
                print("RAW line:", repr(line))

                # Ollama sends lines like:
                #   data: {"id":"cmpl-...","object":"text_completion","choices":[{"text":"How"}]}
                # or the final line: data: [DONE]
                # So we remove the "data: " prefix.
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()

                # If it's the "[DONE]" line, we stop streaming
                if line == "[DONE]":
                    break

                # Otherwise, parse the JSON
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print("JSON parse error:", e)
                    print("Offending line:", line)
                    continue

                # The partial text typically appears in data["choices"][0]["text"]
                choices = data.get("choices", [])
                if not choices:
                    continue

                text_chunk = choices[0].get("text", "")
                # yield this partial text
                yield text_chunk

async def main():
    st.title("Locally Hosted Ollama Chat")
    st.write("Ask any question and see partial streaming with Ollama locally.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for msg in st.session_state.messages:
        display_message(msg)

    # A Chat input box
    user_input = st.chat_input("What do you want to ask?")

    # If user typed something
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

        # 2) Stream the assistant's response
        partial_text = ""
        message_placeholder = st.empty()  # for partial tokens

        with st.chat_message("assistant"):
            async for chunk in ollama_stream(prompt=user_input, model_name="llama3.1:latest"):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

        # 3) Once done, store the final assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "timestamp": "",
            "content": partial_text
        })

if __name__ == "__main__":
    asyncio.run(main())
