#this file was created to test the functionality of the web scraper and lated to ad into the history model for scraping links
# This script is a web scraper that extracts educational content from various websites,
# summarizes it using the Groq API, and stores the results in a MongoDB database.

import asyncio
import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import aiohttp
import base64
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from crawl4ai import AsyncWebCrawler
from groq import Groq

# --------------------------
# Configuration & Constants
# --------------------------

# Hard-coded MongoDB URI (as requested)
MONGODB_URI = "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin"
DB_NAME = "test"
COLLECTION_NAME = "webdata"

# GROQ API key from system environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# List of websites to crawl
WEBSITES = [
    "https://quotes.toscrape.com/page/1/",
    "https://www.ndtv.com/latest",
    "https://www.wikipedia.org/",
    "https://www.reddit.com/r/technology/",
    "https://www.hindustantimes.com/india-news",
    "https://economictimes.indiatimes.com/tech",
    "https://in.indeed.com/jobs?q=machine+learning&l=India",
    "https://www.amazon.in/s?k=headphones",
    "https://www.moneycontrol.com/news/",
    "https://www.zomato.com/india"
]

# Refined CSS selectors in descending priority
PRIORITY_SELECTORS = [
    "article.post", "article", "div.post-content", "div.entry-content",
    "section.article-body", "div.main-content"
]

FALLBACK_SELECTOR = "body"

# Other constants
CONCURRENT_TASKS = 5
PAGE_TIMEOUT_MS = 180_000  # 180 seconds
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
LLM_THREADPOOL_WORKERS = 2

# Regex pattern to detect invalid summaries
INVALID_SUMMARY_PATTERN = re.compile(
    r"(?i)(unfortunately.*no\s*(data|text|content)|no\s*(data|text|content).*(given|available)?|nothing\s*available|no\s*data)"
)

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------
# Pydantic Model Definition
# --------------------------
class EducationalContent(BaseModel):
    title: str = Field(..., description="Title of the content")
    description: str = Field(..., description="Summary of the content")
    content_type: str = Field(..., description="Type (e.g., Article, Tutorial)")
    key_details: List[str] = Field(..., description="Details (e.g., author, tags)")
    image: Optional[str] = Field(None, description="Base64 encoded relevant image")
    source_url: str = Field(..., description="URL of the scraped page")
    crawl_timestamp: float = Field(..., description="Unix timestamp of crawl")

    class Config:
        extra = "forbid"


# --------------------------
# Database Manager (Async)
# --------------------------
class DatabaseManager:
    def __init__(self, connection_string: str, db_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[COLLECTION_NAME]

    async def url_exists(self, url: str) -> bool:
        doc = await self.collection.find_one({"source_url": url})
        return doc is not None

    async def insert_content(self, content: dict):
        await self.collection.insert_one(content)

    def close(self):
        self.client.close()


# --------------------------
# Summarizer (Groq) Wrapper
# --------------------------
class Summarizer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set in environment")
        self.client = Groq(api_key=api_key)
        self.executor = ThreadPoolExecutor(max_workers=LLM_THREADPOOL_WORKERS)
        # Limit simultaneous Groq calls
        self.semaphore = asyncio.Semaphore(2)

    async def summarize_text(self, text: str) -> Optional[str]:
        """
        Summarize raw text content. Returns None if text is too short or on error.
        """
        if not text or len(text.strip()) < 100:
            return None

        prompt = (
            "Summarize this into a 50–100 word description focusing on educational "
            f"value, key topics, programming languages, and intended audience:\n\n{text[:2000]}"
        )

        async with self.semaphore:
            loop = asyncio.get_event_loop()

            def run_sync():
                completion = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                    max_tokens=150,
                    top_p=1,
                    stream=False
                )
                return completion.choices[0].message.content.strip()

            try:
                summary = await loop.run_in_executor(self.executor, run_sync)
                return summary if summary else None
            except Exception as e:
                logger.error(f"Groq summarization error: {e}")
                return None

    async def summarize_site(self, url: str) -> Optional[str]:
        """
        If no content extracted, ask LLM to summarize what it knows about the site.
        If LLM has no relevant info, returns "no data".
        """
        prompt = (
            f"Based on your knowledge (and online information if available), "
            f"summarize this website: {url}. If you have no relevant information, "
            f"simply respond with 'no data'."
        )

        async with self.semaphore:
            loop = asyncio.get_event_loop()

            def run_sync():
                completion = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                    max_tokens=100,
                    top_p=1,
                    stream=False
                )
                return completion.choices[0].message.content.strip()

            try:
                summary = await loop.run_in_executor(self.executor, run_sync)
                return summary if summary else None
            except Exception as e:
                logger.error(f"Groq site-summary error: {e}")
                return None


# --------------------------
# Image Fetcher
# --------------------------
class ImageFetcher:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def download_image(self, image_url: str) -> Optional[str]:
        # Skip URLs with irrelevant keywords
        irrelevant = ["ad", "advertisement", "banner", "logo", "social", "footer", "sponsor", "promo"]
        if any(k in image_url.lower() for k in irrelevant):
            logger.info(f"Skipping irrelevant image: {image_url}")
            return None

        try:
            async with self.session.get(image_url, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if len(data) <= MAX_IMAGE_SIZE_BYTES:
                        return base64.b64encode(data).decode()
                    else:
                        logger.info(f"Skipping large image (>5MB): {image_url}")
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")

        return None


# --------------------------
# HTML Extraction Utilities
# --------------------------
def select_primary_element(soup: BeautifulSoup) -> Optional[Tag]:
    # 1. Find <script type="application/ld+json"> with Article schema
    ld_json = soup.find("script", {"type": "application/ld+json"})
    if ld_json and soup.body:
        return soup.body

    # 2. Check OpenGraph title, locate ancestor
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        title_text = og_title["content"].strip()
        for ancestor in soup.find_all(["article", "main"]):
            if title_text in ancestor.get_text():
                return ancestor

    # 3. Priority selectors
    for sel in PRIORITY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            return el

    # 4. Fallback to body
    return soup.body


def extract_title(element: Tag, soup: BeautifulSoup) -> Optional[str]:
    # 1. <h1>
    h1 = element.select_one("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # 2. OpenGraph meta
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # 3. <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None


# --------------------------
# Extractor: Parses HTML, Summarizes, Downloads Image
# --------------------------
class Extractor:
    def __init__(self, summarizer: Summarizer, image_fetcher: ImageFetcher):
        self.summarizer = summarizer
        self.image_fetcher = image_fetcher

    async def extract_data(self, html: str, url: str) -> Optional[dict]:
        soup = BeautifulSoup(html, "html.parser")
        primary = select_primary_element(soup)

        # If no primary element, ask LLM for site summary
        if not primary:
            logger.warning(f"No primary element found for {url}. Requesting site summary...")
            summary = await self.summarizer.summarize_site(url)
            if summary and not INVALID_SUMMARY_PATTERN.search(summary):
                data = {
                    "title": f"Site overview: {url}",
                    "description": summary,
                    "content_type": "Site Overview",
                    "key_details": ["No direct page content"],
                    "image": None,
                    "source_url": url,
                    "crawl_timestamp": asyncio.get_event_loop().time()
                }
                try:
                    content_model = EducationalContent(**data)
                    return content_model.model_dump()
                except Exception as e:
                    logger.error(f"Pydantic validation failed for site summary {url}: {e}")
            else:
                logger.warning(f"Site summary invalid or 'no data' for {url}. Skipping.")
            return None

        # Title extraction
        title = extract_title(primary, soup)
        if not title:
            logger.warning(f"No valid title for {url}. Requesting site summary...")
            summary = await self.summarizer.summarize_site(url)
            if summary and not INVALID_SUMMARY_PATTERN.search(summary):
                data = {
                    "title": f"Site overview: {url}",
                    "description": summary,
                    "content_type": "Site Overview",
                    "key_details": ["No direct page content"],
                    "image": None,
                    "source_url": url,
                    "crawl_timestamp": asyncio.get_event_loop().time()
                }
                try:
                    content_model = EducationalContent(**data)
                    return content_model.dict()
                except Exception as e:
                    logger.error(f"Pydantic validation failed for site summary {url}: {e}")
            else:
                logger.warning(f"Site summary invalid or 'no data' for {url}. Skipping.")
            return None

        # Content type
        type_elem = primary.select_one(".category, .post-category, .tag, meta[name='keywords']")
        content_type = type_elem.get_text(strip=True) if type_elem and type_elem.get_text(strip=True) else "Article"

        # Key details (author, tags, etc.)
        details_elems = primary.select(".author, .post-meta, .tags, .difficulty, meta[name='keywords']")
        key_details = [d.get_text(strip=True) for d in details_elems if d.get_text(strip=True)]
        if not key_details:
            key_details = ["No details"]

        # Raw text and summarization
        raw_text = primary.get_text(separator=" ", strip=True)
        summary = await self.summarizer.summarize_text(raw_text)
        if not summary:
            logger.warning(f"No valid summary from content for {url}. Requesting site summary...")
            summary = await self.summarizer.summarize_site(url)
            if not summary or INVALID_SUMMARY_PATTERN.search(summary):
                logger.warning(f"Site summary invalid or 'no data' for {url}. Skipping.")
                return None

        # If summary matches invalid pattern, skip
        if INVALID_SUMMARY_PATTERN.search(summary):
            logger.warning(f"Summary pattern matched invalid for {url}. Skipping.")
            return None

        # Image extraction
        img_elem = primary.select_one("img")
        image_b64 = None
        if img_elem and img_elem.get("src"):
            full_url = urljoin(url, img_elem["src"])
            image_b64 = await self.image_fetcher.download_image(full_url)

        # Assemble data
        data = {
            "title": title,
            "description": summary,
            "content_type": content_type,
            "key_details": key_details,
            "image": image_b64,
            "source_url": url,
            "crawl_timestamp": asyncio.get_event_loop().time()
        }

        # Validate via Pydantic
        try:
            content_model = EducationalContent(**data)
            return content_model.dict()
        except Exception as e:
            logger.error(f"Pydantic validation failed for {url}: {e}")
            return None


# --------------------------
# Crawler: Uses AsyncWebCrawler to Fetch HTML
# --------------------------
class Crawler:
    def __init__(self, db_manager: DatabaseManager, extractor: Extractor, session: aiohttp.ClientSession):
        self.db_manager = db_manager
        self.extractor = extractor
        self.session = session

    async def crawl_and_store(self, url: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            logger.info(f"Starting crawl: {url}")
            # Check duplicate
            if await self.db_manager.url_exists(url):
                logger.info(f"Skipping {url}: already exists")
                return

            crawler_config = {
                "browser_type": "chromium",
                "headless": True,
                "user_agent": USER_AGENT,
                "page_timeout": PAGE_TIMEOUT_MS
            }

            try:
                async with AsyncWebCrawler(verbose=True, **crawler_config) as crawler:
                    result = await crawler.arun(
                        url=url,
                        max_pages_to_crawl=1,
                        bypass_cache=True,
                        page_timeout=PAGE_TIMEOUT_MS
                    )

                if not result.success:
                    logger.error(f"Crawl failed for {url}: {result.error_message}")
                    return

                logger.info(f"Crawl succeeded for {url}, extracting...")
                extracted = await self.extractor.extract_data(result.html, url)
                if not extracted:
                    logger.info(f"No valid data extracted or site summary skipped for {url}")
                    return

                await self.db_manager.insert_content(extracted)
                logger.info(f"Inserted into DB: {extracted['title']} ({extracted['content_type']})")

            except Exception as e:
                logger.error(f"Exception during crawl of {url}: {e}")


# --------------------------
# Main Entry Point
# --------------------------
async def main():
    # Validate environment
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in environment")
        return

    # Initialize components
    db_manager = DatabaseManager(MONGODB_URI, DB_NAME)
    summarizer = Summarizer(api_key=GROQ_API_KEY)

    # Shared aiohttp session
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        image_fetcher = ImageFetcher(session)
        extractor = Extractor(summarizer, image_fetcher)
        crawler = Crawler(db_manager, extractor, session)

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
        tasks = [crawler.crawl_and_store(url, semaphore) for url in WEBSITES]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Close DB connection
    db_manager.close()
    logger.info(f"Scraping completed. Data stored in {DB_NAME}.{COLLECTION_NAME}")


if __name__ == "__main__":
    asyncio.run(main())
