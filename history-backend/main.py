import sqlite3
import asyncio
import httpx
import json
import os
import logging
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup, Tag
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONCURRENT_TASKS = 5
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
LLM_THREADPOOL_WORKERS = 2
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

# Enhanced CSS selectors
PRIORITY_SELECTORS = [
    "article.post", "article", "div.post-content", "div.entry-content",
    "section.article-body", "div.main-content", ".content", ".article-content"
]

# Invalid summary pattern
INVALID_SUMMARY_PATTERN = re.compile(
    r"(?i)(unfortunately.*no\s*(data|text|content)|no\s*(data|text|content).*(given|available)?|nothing\s*available|no\s*data)"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static directory not found")

# Your original Pydantic model
class HistoryItem(BaseModel):
    url: str
    title: str
    lastVisitTime: float

# Enhanced content model
class EducationalContent(BaseModel):
    title: str
    description: str
    content_type: str
    key_details: List[str]
    image: Optional[str] = None
    source_url: str
    crawl_timestamp: float

# Your original database initialization
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visited_sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            last_visit_time DATETIME,
            html_content TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            extracted_json TEXT,
            extraction_timestamp DATETIME,
            FOREIGN KEY (url) REFERENCES visited_sites(url)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Groq Summarizer
class Summarizer:
    def __init__(self, api_key: str):
        if not api_key:
            logger.warning("GROQ_API_KEY not found, AI features will be limited")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
        self.executor = ThreadPoolExecutor(max_workers=LLM_THREADPOOL_WORKERS)
        self.semaphore = asyncio.Semaphore(2)

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize raw text content"""
        if not self.client or not text or len(text.strip()) < 100:
            return self._generate_fallback_summary(text)

        prompt = (
            "Summarize this web content into 2-3 sentences focusing on the main topic, "
            f"key insights, and educational value:\n\n{text[:2000]}"
        )

        async with self.semaphore:
            loop = asyncio.get_event_loop()

            def run_sync():
                completion = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
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
                return self._generate_fallback_summary(text)

    def _generate_fallback_summary(self, text: str) -> str:
        """Generate basic summary when AI unavailable"""
        if not text:
            return "No content available for summarization."
        
        # Extract first meaningful sentences
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if meaningful_sentences:
            summary = '. '.join(meaningful_sentences[:2])
            return summary[:200] + "..." if len(summary) > 200 else summary
        
        return f"Content extracted from webpage. Contains {len(text)} characters of text."

    async def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        if not self.client:
            return self._extract_basic_keywords(text)

        prompt = f"Extract 3-5 key topics/keywords from this content. Return only keywords separated by commas:\n\n{text[:1000]}"

        async with self.semaphore:
            loop = asyncio.get_event_loop()

            def run_keyword_extraction():
                try:
                    completion = self.client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=100,
                        stream=False
                    )
                    result = completion.choices[0].message.content.strip()
                    keywords = [k.strip() for k in result.split(',') if k.strip()]
                    return keywords[:5]
                except Exception:
                    return self._extract_basic_keywords(text)

            try:
                return await loop.run_in_executor(self.executor, run_keyword_extraction)
            except Exception:
                return self._extract_basic_keywords(text)

    def _extract_basic_keywords(self, text: str) -> List[str]:
        """Basic keyword extraction without AI"""
        if not text:
            return ["General Content"]
        
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'were', 'what', 'when', 'where'}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 most frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word.capitalize() for word, _ in top_words] if top_words else ["General Content"]

# Image fetcher
class ImageFetcher:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def download_image(self, image_url: str) -> Optional[str]:
        # Skip URLs with irrelevant keywords
        irrelevant = ["ad", "advertisement", "banner", "logo", "social", "footer", "sponsor", "promo"]
        if any(k in image_url.lower() for k in irrelevant):
            return None

        try:
            async with self.session.get(image_url, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if len(data) <= MAX_IMAGE_SIZE_BYTES:
                        return base64.b64encode(data).decode()
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")

        return None

# Content extraction functions
def select_primary_element(soup: BeautifulSoup) -> Optional[Tag]:
    """Select the main content element"""
    # Check OpenGraph title, locate ancestor
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        title_text = og_title["content"].strip()
        for ancestor in soup.find_all(["article", "main"]):
            if title_text in ancestor.get_text():
                return ancestor

    # Priority selectors
    for sel in PRIORITY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            return el

    # Fallback to body
    return soup.body

def extract_title(element: Tag, soup: BeautifulSoup) -> Optional[str]:
    """Extract page title"""
    # Try h1 first
    h1 = element.select_one("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # OpenGraph meta
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # HTML title
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None

def determine_content_type(soup: BeautifulSoup, url: str) -> str:
    """Determine content type based on URL and page structure"""
    url_lower = url.lower()
    
    # Check URL patterns
    if any(pattern in url_lower for pattern in ['blog', 'article', 'post']):
        return 'Article'
    elif any(pattern in url_lower for pattern in ['tutorial', 'guide', 'how-to']):
        return 'Tutorial'
    elif any(pattern in url_lower for pattern in ['documentation', 'docs', 'api']):
        return 'Documentation'
    elif any(pattern in url_lower for pattern in ['news', 'press']):
        return 'News'
    elif any(pattern in url_lower for pattern in ['course', 'lesson', 'learn']):
        return 'Educational'
    
    return 'Web Content'

# Enhanced extractor class
class Extractor:
    def __init__(self, summarizer: Summarizer, image_fetcher: ImageFetcher):
        self.summarizer = summarizer
        self.image_fetcher = image_fetcher

    async def extract_data(self, html: str, url: str) -> Optional[dict]:
        """Extract structured content from HTML"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            primary = select_primary_element(soup)
            if not primary:
                logger.warning(f"No primary element found for {url}")
                return None

            # Extract title
            title = extract_title(primary, soup)
            if not title:
                logger.warning(f"No valid title for {url}")
                return None

            # Extract main text content
            raw_text = primary.get_text(separator=" ", strip=True)
            if len(raw_text) < 100:
                logger.warning(f"Content too short for {url}")
                return None

            # Generate AI summary and keywords
            summary = await self.summarizer.summarize_text(raw_text)
            if not summary or INVALID_SUMMARY_PATTERN.search(summary):
                logger.warning(f"No valid summary generated for {url}")
                return None

            keywords = await self.summarizer.extract_keywords(raw_text)
            
            # Determine content type
            content_type = determine_content_type(soup, url)
            
            # Extract image
            image_b64 = None
            img_elem = primary.select_one("img")
            if img_elem and img_elem.get("src"):
                full_url = urljoin(url, img_elem["src"])
                image_b64 = await self.image_fetcher.download_image(full_url)

            # Assemble data
            data = {
                "title": title,
                "description": summary,
                "content_type": content_type,
                "key_details": keywords,
                "image": image_b64,
                "source_url": url,
                "crawl_timestamp": datetime.now().timestamp()
            }

            # Validate with Pydantic
            content_model = EducationalContent(**data)
            return content_model.model_dump()

        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None

# Initialize components
summarizer = Summarizer(GROQ_API_KEY)

# Your original working async HTML fetcher
async def fetch_html(url: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            async with httpx.AsyncClient(timeout=10.0, headers={'User-Agent': USER_AGENT}) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.text
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return ""

# Your original working POST endpoint
@app.post("/api/ingest")
async def ingest_history(items: List[HistoryItem]):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    semaphore = asyncio.Semaphore(5)
    tasks = []
    new_urls = []

    for item in items:
        try:
            # Check if URL already exists
            cursor.execute("SELECT id FROM visited_sites WHERE url = ?", (item.url,))
            if cursor.fetchone():
                logger.info(f"Skipping duplicate URL: {item.url}")
                continue
            
            # Insert new record
            cursor.execute(
                "INSERT INTO visited_sites (url, title, last_visit_time) VALUES (?, ?, ?)",
                (item.url, item.title, datetime.fromtimestamp(item.lastVisitTime / 1000.0))
            )
            new_urls.append(item.url)
            tasks.append(fetch_html(item.url, semaphore))
            
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate URL skipped: {item.url}")
            continue

    conn.commit()
    conn.close()

    # Fetch HTML content asynchronously
    if tasks:
        html_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        
        for url, html in zip(new_urls, html_results):
            if isinstance(html, str) and html:
                cursor.execute(
                    "UPDATE visited_sites SET html_content = ? WHERE url = ?",
                    (html, url)
                )
                logger.info(f"Stored HTML for {url}")
        
        conn.commit()
        conn.close()

    return {"status": "success", "processed": len(new_urls)}

# Enhanced extraction endpoint
@app.post("/api/extract")
async def extract_data_endpoint(url: Optional[str] = None):
    logger.info(f"Starting extraction for {'specific URL: ' + url if url else 'all URLs'}")
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    
    # Fetch URLs to process
    if url:
        cursor.execute("SELECT url, html_content FROM visited_sites WHERE url = ? AND html_content IS NOT NULL", (url,))
        results = [cursor.fetchone()]
        if not results[0]:
            conn.close()
            raise HTTPException(status_code=404, detail="URL not found or no HTML content")
    else:
        cursor.execute("SELECT url, html_content FROM visited_sites WHERE html_content IS NOT NULL")
        results = cursor.fetchall()

    conn.close()

    if not results:
        return {"status": "success", "message": "No URLs with HTML content to process", "results": []}

    # Process URLs with enhanced extraction
    extracted_results = []
    semaphore = asyncio.Semaphore(3)
    
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        image_fetcher = ImageFetcher(session)
        extractor = Extractor(summarizer, image_fetcher)
        
        tasks = []
        valid_urls = []
        
        for row in results:
            url_val, html_content = row
            if html_content and url_val:
                tasks.append(extractor.extract_data(html_content, url_val))
                valid_urls.append(url_val)

        if not tasks:
            return {"status": "success", "message": "No valid URLs to process", "results": []}

        # Run extraction tasks
        extraction_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        
        for url_val, result in zip(valid_urls, extraction_results):
            if isinstance(result, dict) and result:
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO extracted_data (url, extracted_json, extraction_timestamp) VALUES (?, ?, ?)",
                        (url_val, json.dumps(result), datetime.now())
                    )
                    extracted_results.append({"url": url_val, "extracted_data": result})
                    logger.info(f"Successfully extracted data for {url_val}: {result.get('title', 'No title')}")
                except Exception as e:
                    logger.error(f"Failed to store extracted data for {url_val}: {e}")
                    extracted_results.append({"url": url_val, "error": str(e)})
            else:
                error_msg = str(result) if isinstance(result, Exception) else "No data extracted"
                logger.error(f"Extraction failed for {url_val}: {error_msg}")
                extracted_results.append({"url": url_val, "error": error_msg})
        
        conn.commit()
        conn.close()

    logger.info(f"Extraction completed. Processed {len(extracted_results)} URLs")
    return {"status": "success", "results": extracted_results}

# Keep all your original endpoints
@app.get("/api/urls")
async def get_urls():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.id, v.url, v.title, v.last_visit_time, e.extracted_json
        FROM visited_sites v
        LEFT JOIN extracted_data e ON v.url = e.url
        ORDER BY v.last_visit_time DESC
    """)
    results = []
    for row in cursor.fetchall():
        try:
            extracted_data = json.loads(row[4]) if row[4] else None
            results.append({
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "last_visit_time": row[3],
                "extracted_data": extracted_data
            })
        except json.JSONDecodeError:
            results.append({
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "last_visit_time": row[3],
                "extracted_data": None
            })
    conn.close()
    return results

@app.get("/api/html/{url:path}")
async def get_html_by_url(url: str):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT html_content FROM visited_sites WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    if not result or not result[0]:
        raise HTTPException(status_code=404, detail="HTML not found for URL")
    return {"url": url, "html_content": result[0]}

@app.get("/api/html/id/{id}")
async def get_html_by_id(id: int):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT url, html_content FROM visited_sites WHERE id = ?", (id,))
    result = cursor.fetchone()
    conn.close()
    if not result or not result[1]:
        raise HTTPException(status_code=404, detail="HTML not found for ID")
    return {"url": result[0], "html_content": result[1]}

# New endpoints for frontend
@app.get("/")
async def root():
    return {"message": "MindCanvas Backend Running", "version": "2.0", "groq_enabled": bool(GROQ_API_KEY)}

@app.get("/api/content")
async def get_all_content():
    """Get all extracted content for knowledge graph"""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT v.url, v.title, e.extracted_json, e.extraction_timestamp
        FROM visited_sites v
        JOIN extracted_data e ON v.url = e.url
        ORDER BY e.extraction_timestamp DESC
    """)
    
    results = []
    for row in cursor.fetchall():
        try:
            extracted_data = json.loads(row[2]) if row[2] else None
            if extracted_data:
                results.append({
                    "id": len(results) + 1,
                    "url": row[0],
                    "title": extracted_data.get("title", row[1]),
                    "description": extracted_data.get("description", ""),
                    "content_type": extracted_data.get("content_type", "Web Content"),
                    "key_details": extracted_data.get("key_details", []),
                    "has_image": bool(extracted_data.get("image")),
                    "extraction_timestamp": row[3]
                })
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing extracted data: {e}")
            continue
    
    conn.close()
    return {"content": results, "total": len(results)}

@app.get("/api/stats")
async def get_stats():
    """Get processing statistics"""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    
    # Get total URLs
    cursor.execute("SELECT COUNT(*) FROM visited_sites")
    total_urls = cursor.fetchone()[0]
    
    # Get extracted content count
    cursor.execute("SELECT COUNT(*) FROM extracted_data")
    total_extracted = cursor.fetchone()[0]
    
    # Get URLs with HTML
    cursor.execute("SELECT COUNT(*) FROM visited_sites WHERE html_content IS NOT NULL")
    with_html = cursor.fetchone()[0]
    
    # Get content type distribution
    cursor.execute("SELECT extracted_json FROM extracted_data")
    content_types = {}
    for row in cursor.fetchall():
        try:
            data = json.loads(row[0])
            content_type = data.get("content_type", "Unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
        except (json.JSONDecodeError, KeyError):
            continue
    
    conn.close()
    
    return {
        "status_counts": {
            "pending": total_urls - total_extracted,
            "completed": total_extracted,
            "failed": 0  # We don't track failures separately yet
        },
        "content_types": content_types,
        "total_extracted": total_extracted
    }

@app.delete("/api/reset")
async def reset_database():
    """Reset database (for development)"""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM extracted_data")
    cursor.execute("DELETE FROM visited_sites")
    
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Database reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")