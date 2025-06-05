import json
import asyncio
import logging
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import CssExtractionStrategy
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def crawl_and_extract(url: str, html_content: Optional[str], semaphore: asyncio.Semaphore) -> str:
    """
    Use Crawl4AI to extract structured data (headline and summary) from a URL or its HTML content.
    
    Args:
        url: The URL to process.
        html_content: Pre-fetched HTML content (if available).
        semaphore: asyncio.Semaphore to limit concurrent Crawl4AI tasks.
    
    Returns:
        JSON string of extracted data.
    """
    async with semaphore:
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                # Define extraction strategy with robust CSS selectors
                extraction_strategy = CssExtractionStrategy(
                    schema={
                        "headline": "h1, h2, .headline, .title, [itemprop='headline']",
                        "summary": "p:first-of-type, .summary, .lead, [itemprop='description'], meta[name='description']::attr(content)"
                    }
                )
                logger.info(f"Starting extraction for {url}")

                # Use existing HTML if available, otherwise fetch
                if html_content:
                    result = await crawler.arun(
                        url=url,
                        html_content=html_content,
                        extraction_strategy=extraction_strategy,
                        bypass_cache=True
                    )
                else:
                    result = await crawler.arun(
                        url=url,
                        extraction_strategy=extraction_strategy
                    )

                # Handle multiple matches by taking the first non-empty result
                extracted_data = {}
                for key, value in result.extracted_content.items():
                    if isinstance(value, list) and value:
                        extracted_data[key] = value[0]
                    elif value:
                        extracted_data[key] = value
                    else:
                        extracted_data[key] = ""
                
                extracted_data = extracted_data or {"headline": "", "summary": ""}
                logger.info(f"Extracted data for {url}: {extracted_data}")
                return json.dumps(extracted_data)
        except Exception as e:
            logger.error(f"Crawl4AI extraction failed for {url}: {str(e)}")
            raise Exception(f"Crawl4AI extraction failed for {url}: {str(e)}")
