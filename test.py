import asyncio
import chromadb
import streamlit as st
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

# Initialize clients and model
client = chromadb.PersistentClient(path="./chroma_data")
local_client = AsyncOpenAI(base_url='http://localhost:11434/v1', api_key="na")
model = OpenAIModel(model_name="llama3.1:latest", openai_client=local_client)

async def run_agent_with_streaming(user_input: str):
    deps = PydanticAIDeps(client=client, openai_client=local_client)
    message_placeholder = st.empty()
    full_response = ""

    try:
        st.write("Starting stream...")
        async with pydantic_ai_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],
        ) as result:
            st.write("Stream opened successfully.")
            async for chunk in result.stream_text(delta=True):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                st.write(f"Received chunk: {chunk}")  # Debug output
                await asyncio.sleep(0.01)

        st.write("Stream completed.")
        message_placeholder.markdown(full_response)
        return full_response, result.new_messages()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, []

async def main():
    st.title("Pydantic AI Agentic RAG")
    st.write("Ask any question about Pydantic AI, the hidden truths of the beauty of this framework lie within.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("What questions do you have about Pydantic AI?")

    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            st.write("Generating response...")
            full_response, new_messages = await run_agent_with_streaming(user_input)
            if full_response:
                filtered_messages = [msg for msg in new_messages 
                                     if not (hasattr(msg, 'parts') and 
                                             any(part.part_kind == 'user-prompt' for part in msg.parts))]
                st.session_state.messages.extend(filtered_messages)
                st.session_state.messages.append(
                    ModelResponse(parts=[TextPart(content=full_response)])
                )
            else:
                st.error("Failed to generate a response.")

def display_message_part(part):
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

if __name__ == "__main__":
    asyncio.run(main())