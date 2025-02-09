import asyncio
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import aiohttp

class BlogPost(BaseModel):
    title: str
    content: str
    date: str

class RAGPipeline:
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.embed_model = "nomic-embed-text"
        self.llm_model = "llama3.2"
        
        # Configure ChromaDB with Ollama embeddings [8][19]
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = OllamaEmbeddingFunction(
            url=f"{self.ollama_host}/api/embeddings",
            model_name=self.embed_model
        )
        self.collection = self.client.get_or_create_collection(
            name="web_content",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    async def crawl_website(self, url: str):
        """Crawl and extract content using Crawl4AI [5][19]"""
        strategy = LLMExtractionStrategy(
            provider="ollama",
            schema=BlogPost.model_json_schema(),
            extraction_type="schema",
            instruction="Extract blog posts with title, content, and date",
            chunk_token_threshold=4000,
            input_format="html"
        )

        crawler = AsyncWebCrawler(config=BrowserConfig(headless=True))
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                extraction_strategy=strategy,
                cache_mode="bypass"
            )
        )
        
        if not result.success:
            raise Exception(f"Crawling failed: {result.error_message}")
            
        return json.loads(result.extracted_content)

    async def store_embeddings(self, documents: list):
        """Store crawled content in ChromaDB [2][8]"""
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [{
            "source": doc.get('url', ''),
            "title": doc['title'],
            "date": doc['date']
        } for doc in documents]
        
        documents = [f"{doc['title']}\n{doc['content']}" for doc in documents]
        
        await self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    async def query_llm(self, question: str, context: str):
        """Query Ollama with RAG context [3][14]"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": f"Context:\n{context}"},
                        {"role": "user", "content": question}
                    ],
                    "stream": False
                }
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def rag_query(self, question: str, n_results=3):
        """Full RAG pipeline execution [5][10]"""
        # Retrieve relevant documents
        results = await self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Combine context from top matches
        context = "\n\n".join(results['documents'][0])
        
        # Generate LLM response
        response = await self.query_llm(question, context)
        return response['message']['content']

async def main():
    rag = RAGPipeline()
    
    # Example usage:
    # 1. Crawl and index content
    content = await rag.crawl_website("https://https://ai.pydantic.dev/")
    await rag.store_embeddings(content)
    
    # 2. Query the RAG system
    response = await rag.rag_query("What's the latest post about AI safety?")
    print("RAG Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
