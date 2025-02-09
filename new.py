import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class Blog(BaseModel):
    title: str
    date: str
	
async def main():
    llm_strategy = LLMExtractionStrategy(
        provider="ollama/deepseek-r1:7b", 
        api_token="none",
        schema=Blog.schema_json(),            
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

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://fahdmirza.com",
            config=crawl_config
        )

        if result.success:
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)
            llm_strategy.show_usage()
        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())