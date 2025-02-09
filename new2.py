import os
import asyncio
import json
from pydantic import BaseModel
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import chromadb
from chromadb.utils import embedding_functions
import requests

class Blog(BaseModel):
    title: str
    date: str

class OllamaEmbedder:
    def __init__(self, model_name="nomic-embed-text:latest"):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            try:
                response = requests.post("http://localhost:11434/api/embed", json={
                    "model": self.model_name,
                    "prompt": text
                })
                response.raise_for_status()
                data = response.json()
                embeddings.append(data['embedding'])
            except Exception as e:
                print(f"Error in embedding: {str(e)}")
                embeddings.append([0.0] * 4096)  # Fallback embedding
        return embeddings

def query_ollama(prompt, context):
    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "deepseek-R1:7b",
            "messages": [
                {"role": "system", "content": f"Use this context to inform your responses: {context}"},
                {"role": "user", "content": prompt}
            ]
        })
        response.raise_for_status()
        data = response.json()
        return data['message']['content']
    except requests.RequestException as e:
        return f"Error querying Ollama: {str(e)}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"Error processing Ollama response: {str(e)}"

async def main():
    llm_strategy = LLMExtractionStrategy(
        provider="ollama/deepseek-R1:7b", 
        api_token="none",
        schema=Blog.model_json_schema(),            
        extraction_type="schema",
        instruction="Extract all blog posts objects with blog title and date from the content.",
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 800}
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    browser_cfg = BrowserConfig(headless=True)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = OllamaEmbedder(model_name="nomic-embed-text:latest")
    collection = client.get_or_create_collection(
        "blog_posts",
        embedding_function=embedding_function
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://fahdmirza.com",
            config=crawl_config
        )

        if result.success:
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)
            llm_strategy.show_usage()

            # Add extracted data to ChromaDB
            for i, blog_post in enumerate(data):
                collection.add(
                    documents=[json.dumps(blog_post)],
                    metadatas=[{"source": "https://fahdmirza.com"}],
                    ids=[f"blog_{i}"]
                )
            print("Data added to ChromaDB successfully.")

            # Query the database
            while True:
                user_query = input("Enter your question about the blog posts (or 'quit' to exit): ")
                if user_query.lower() == 'quit':
                    break

                results = collection.query(
                    query_texts=[user_query],
                    n_results=1
                )

                context = results['documents'][0][0] if results['documents'] else "No relevant context found."
                response = query_ollama(user_query, context)
                print("Assistant:", response)

        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())
