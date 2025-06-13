import json
import asyncio
import logging
from crawl4ai import AsyncWebCrawler
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def crawl_and_extract(url: str, html_content: Optional[str], semaphore: asyncio.Semaphore) -> str:
    """
    Use Crawl4AI to extract structured data from a URL or its HTML content.
    Updated for current crawl4ai version.
    
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
                logger.info(f"Starting extraction for {url}")

                # Use existing HTML if available, otherwise fetch
                if html_content:
                    result = await crawler.arun(
                        url=url,
                        html_content=html_content,
                        bypass_cache=True
                    )
                else:
                    result = await crawler.arun(
                        url=url,
                        bypass_cache=True
                    )

                if not result.success:
                    logger.error(f"Crawl4AI failed for {url}: {result.error_message}")
                    raise Exception(f"Crawl4AI failed for {url}: {result.error_message}")

                # Extract basic structured data from the result
                extracted_data = {
                    "headline": result.metadata.get("title", "") if result.metadata else "",
                    "summary": result.cleaned_html[:500] + "..." if result.cleaned_html else ""
                }
                
                # Try to extract more structured data if available
                if hasattr(result, 'extracted_content') and result.extracted_content:
                    extracted_data.update(result.extracted_content)
                
                logger.info(f"Extracted data for {url}: {extracted_data}")
                return json.dumps(extracted_data)
                
        except Exception as e:
            logger.error(f"Crawl4AI extraction failed for {url}: {str(e)}")
            raise Exception(f"Crawl4AI extraction failed for {url}: {str(e)}")