from __future__ import annotations
import asyncio
import json
import time
from typing import Literal, TypedDict, Optional, List, Any, AsyncGenerator

import streamlit as st
import logfire
import chromadb
import httpx

# pydantic_ai + pydantic_ai_expert
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# Load env vars if needed
from dotenv import load_dotenv
load_dotenv()

# Suppress logfire warnings (optional)
logfire.configure(send_to_logfire='never')

# -----------------------------------------------------------------------------
# 1) OUR CUSTOM OLLAMA CLIENT THAT MIMICS OPENAI'S CHUNKED STREAM
# -----------------------------------------------------------------------------

class OllamaClientMimicOpenAI:
    """
    Custom client that:
      - Calls Ollama's /v1/completions endpoint with "stream": True
      - Parses each SSE line (prefixed with "data: ")
      - Converts it to an OpenAI-like chunk:
          {"id":..., "object":"chat.completion.chunk", "choices":[{"delta":{"content":"partial"}}], ...}
      - Yields that JSON string to pydantic_ai_expert so `stream_text(delta=True)` works.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1:latest"):
        self.base_url = base_url
        self.model_name = model_name

    async def achat_stream(self,
                           messages: List[Any],
                           settings: Optional[Any] = None,
                           **kwargs) -> AsyncGenerator[str, None]:
        """
        pydantic_ai_expert.run_stream(...) calls .achat_stream(...) for partial outputs.

        We must:
          1) Convert the list of pydantic_ai "ChatMessage"s to a single prompt.
          2) Send that prompt to Ollama (POST /v1/completions, stream=True).
          3) For each chunk from Ollama, produce a JSON that looks like an OpenAI chunk,
             containing a "delta":{"content": "..."} so pydantic_ai_expert can parse partial tokens.
        """
        # 1) Build a single prompt from system/user/assistant messages
        prompt_fragments = []
        for m in messages:
            if m.role == "system":
                prompt_fragments.append(f"[system]\n{m.content}")
            elif m.role == "user":
                prompt_fragments.append(f"[user]\n{m.content}")
            elif m.role == "assistant":
                prompt_fragments.append(f"[assistant]\n{m.content}")

        full_prompt = "\n\n".join(prompt_fragments)

        # 2) Construct the Ollama request
        url = f"{self.base_url}/v1/completions"
        payload = {
            "prompt": full_prompt,
            "model": self.model_name,
            "stream": True
        }

        # 3) Make streaming POST request to Ollama
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                # 4) Stream lines like:
                #    data: {"choices":[{"text":"Hello"}],"model":"..."...}
                #    data: {"choices":[{"text":" world"}],...}
                #    data: [DONE]
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    if line == "[DONE]":
                        break

                    try:
                        chunk_json = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # We expect something like chunk_json["choices"][0]["text"] = "some partial text"
                    choices = chunk_json.get("choices", [])
                    if not choices:
                        continue

                    # The partial text from Ollama
                    text_chunk = choices[0].get("text", "")
                    if not text_chunk:
                        continue

                    # 5) Transform that into an OpenAI "chat.completion.chunk" shape
                    #    so pydantic_ai_expert sees "delta":{"content": "..."}.
                    #    We'll create a minimal chunk with finish_reason=None:
                    openai_like_chunk = {
                        "id": "ollama-chunk",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [
                            {
                                "delta": {"content": text_chunk},
                                "index": 0,
                                "finish_reason": None
                            }
                        ]
                    }

                    # 6) pydantic_ai_expert expects each chunk as a JSON string line
                    #    We'll yield that line. It sees it, extracts the 'delta' content.
                    yield json.dumps(openai_like_chunk)

    async def achat(self,
                    messages: List[Any],
                    settings: Optional[Any] = None,
                    **kwargs) -> str:
        """
        Non-stream fallback if streaming isn't used. We just concatenate all partial chunks.
        """
        full_text = []
        async for chunk_str in self.achat_stream(messages, settings=settings, **kwargs):
            # chunk_str is an OpenAI-like JSON chunk with "delta":{"content":"..."}
            # We parse it:
            try:
                data = json.loads(chunk_str)
                content = data["choices"][0]["delta"].get("content", "")
                full_text.append(content)
            except:
                pass
        return "".join(full_text)

# -----------------------------------------------------------------------------
# 2) YOUR ORIGINAL CODE (MINOR MODS TO USE THE CUSTOM CLIENT)
# -----------------------------------------------------------------------------

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, etc.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Same agent logic: pydantic_ai_expert.run_stream(...) uses our custom client for streaming.
    """
    deps = PydanticAIDeps(
        client=chromadb.PersistentClient(path="./chroma_data"),
        openai_client=ollama_client_mimic  # <--- Our custom client
    )

    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],
    ) as result:

        partial_text = ""
        message_placeholder = st.empty()

        # .stream_text(delta=True) reads each chunk expecting "delta":{"content": "..."}
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Add new messages from this run, excluding user-prompt
        filtered_messages = [
            msg for msg in result.new_messages()
            if not (hasattr(msg, 'parts') and
                    any(part.part_kind == 'user-prompt' for part in msg.parts))
        ]
        st.session_state.messages.extend(filtered_messages)

        # Store final response
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("Pydantic AI Agentic RAG (with Ollama Streaming)")
    st.write("Ask a question and watch partial text streaming from Ollama in real time.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("What do you want to ask?")
    if user_input:
        # Append user request
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Show partial assistant response
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # 3) CREATE OUR CUSTOM OLLAMA CLIENT & OPTIONAL MODEL
    # -----------------------------------------------------------------------------
    ollama_client_mimic = OllamaClientMimicOpenAI(
        base_url='http://localhost:11434',
        model_name='llama3.1:latest'  # must match what 'ollama list' shows
    )

    # The pydantic_ai_expert flow doesn't necessarily require an OpenAIModel instance,
    # as it directly uses openai_client from PydanticAIDeps. But we can define one:
    model = OpenAIModel(
        model_name="llama3.1:latest",
        openai_client=ollama_client_mimic
    )

    asyncio.run(main())
