import sqlite3
import asyncio
import httpx
import json
import os
import logging
import re
import hashlib
import math
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bs4 import BeautifulSoup, Tag
from openai import OpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
#OpenAI Error lib
from openai import OpenAIError

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONCURRENT_TASKS = 5
BATCH_SIZE = 20  # Using your optimized batch size
CONCURRENCY = 3  # Parallel API requests
MAX_OUTPUT_TOKENS = 2048  # Reduced for summaries
MAX_CONTENT_LENGTH = 1500

# URLs to EXCLUDE
EXCLUDED_DOMAINS = {
    'google.com', 'google.co.in', 'bing.com', 'duckduckgo.com', 'yahoo.com',
    'facebook.com', 'twitter.com', 'x.com', 'instagram.com', 'linkedin.com',
    'tiktok.com', 'reddit.com', 'pinterest.com'
}

EXCLUDED_PATTERNS = [
    r'/search\?', r'/results\?', r'/feed', r'/login', r'/signup', 
    r'/cart', r'/checkout', r'/admin', r'\.pdf$', r'\.(jpg|png|gif|mp4)$'
]

CONTENT_SELECTORS = [
    'article', 'main', '.content', '.post-content', '.entry-content',
    '.article-content', '[role="main"]'
]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static directory not found")

# Models
class HistoryItem(BaseModel):
    url: str
    title: str
    lastVisitTime: float

# Database schema
def init_db():
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            content_type TEXT NOT NULL,
            key_topics TEXT NOT NULL,
            visit_timestamp DATETIME,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            processing_method TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS url_cache (
            url_hash TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            content_type TEXT,
            key_topics TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

init_db()

# URL filtering functions
def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        if domain in EXCLUDED_DOMAINS:
            return False
        
        for pattern in EXCLUDED_PATTERNS:
            if re.search(pattern, url.lower()):
                return False
        
        return parsed.scheme in ['http', 'https'] and '.' in domain
        
    except Exception:
        return False

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def clean_title(title: str) -> str:
    if not title:
        return "Untitled"
    
    suffixes = [' - Google Search', ' - Bing', ' | Facebook', ' | Twitter']
    for suffix in suffixes:
        if title.endswith(suffix):
            title = title[:-len(suffix)]
    
    return title.strip()[:100] or "Untitled"

# Content extractor
class FastExtractor:
    def extract_clean_text(self, html: str, url: str, title: str) -> Optional[Dict[str, str]]:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                if hasattr(tag, 'decompose'):
                    tag.decompose()
            
            # Find main content
            main_content = None
            for selector in CONTENT_SELECTORS:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 200:
                    main_content = element
                    break
            
            if not main_content:
                main_content = soup.body
            
            if not main_content:
                return None
            
            content_text = main_content.get_text(separator=' ', strip=True)
            content_text = re.sub(r'\s+', ' ', content_text)
            
            if len(content_text) < 100:
                return None
            
            return {
                'url': url,
                'title': clean_title(title),
                'content': content_text[:MAX_CONTENT_LENGTH],
                'domain': urlparse(url).netloc
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed for {url}: {e}")
            return None

# OpenAI Batch API Processor using your working code
class OpenAIBatchProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            logger.warning("OPENAI_API_KEY not found, using pattern-based processing only")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    async def _call_batch(self, prompts: List[str]) -> List[str]:
        """Send one batch with retry - using your working code"""
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(OpenAIError),
        ):
            with attempt:
                response = await self.client.responses.acreate(
                    model="gpt-4.1-nano",  # Using your working model
                    input=[{"text": p} for p in prompts],
                    text={"format": {"type": "text"}},
                    reasoning={}, 
                    tools=[],
                    temperature=1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    top_p=1,
                    store=True,
                )
                return [c.text.strip() for c in response.data]

    async def batch_run(self, prompts: List[str]) -> List[str]:
        """Batch processing with concurrency control - using your working code"""
        if not self.client:
            return ["Pattern-based processing" for _ in prompts]
        
        sem = asyncio.Semaphore(CONCURRENCY)

        async def worker(chunk: List[str]) -> List[str]:
            async with sem:
                logger.info(f"→ Sending batch of {len(chunk)} prompts to gpt-4.1-nano")
                return await self._call_batch(chunk)

        # Create tasks for each chunk
        tasks = []
        total = len(prompts)
        for i in range(0, total, BATCH_SIZE):
            chunk = prompts[i : i + BATCH_SIZE]
            tasks.append(asyncio.create_task(worker(chunk)))

        # Gather results
        batches = await asyncio.gather(*tasks)
        # Flatten results
        return [resp for batch in batches for resp in batch]

    async def process_content_batch(self, content_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process content items using batch API"""
        
        if not self.client or not content_items:
            return [self._pattern_process(item) for item in content_items]
        
        try:
            # Create prompts for batch processing
            prompts = []
            for item in content_items:
                prompt = f"""Analyze this webpage content and respond with JSON only:

URL: {item['url']}
Title: {item['title']}
Content: {item['content']}

Respond with JSON:
{{
    "title": "cleaned title (max 80 chars)",
    "summary": "2-3 sentence summary focusing on main value and insights",
    "content_type": "Article|Tutorial|Documentation|News|Tool|Reference|Blog",
    "key_topics": ["topic1", "topic2", "topic3"]
}}

JSON only, no explanation:"""
                prompts.append(prompt)
            
            # Process all prompts in batches
            responses = await self.batch_run(prompts)
            
            # Parse responses
            results = []
            for i, (item, response) in enumerate(zip(content_items, responses)):
                try:
                    # Clean and parse JSON response
                    json_str = response.strip()
                    if '```json' in json_str:
                        json_str = json_str.split('```json')[1].split('```')[0]
                    elif '```' in json_str:
                        json_str = json_str.split('```')[1].split('```')[0]
                    elif '{' in json_str:
                        start = json_str.find('{')
                        end = json_str.rfind('}') + 1
                        json_str = json_str[start:end]
                    
                    parsed = json.loads(json_str)
                    
                    results.append({
                        'url': item['url'],
                        'title': parsed.get('title', item['title'])[:100],
                        'summary': parsed.get('summary', '')[:500],
                        'content_type': parsed.get('content_type', 'Web Content'),
                        'key_topics': parsed.get('key_topics', ['General'])[:4],
                        'processing_method': 'openai_batch'
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to parse response for {item['url']}: {e}")
                    results.append(self._pattern_process(item))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [self._pattern_process(item) for item in content_items]

    def _pattern_process(self, item: Dict[str, str]) -> Dict[str, Any]:
        """Fast pattern-based processing fallback"""
        url = item['url'].lower()
        title = item['title'].lower()
        content = item['content'].lower()
        
        # Content type detection
        content_type = 'Web Content'
        if any(word in url for word in ['tutorial', 'guide', 'how-to']):
            content_type = 'Tutorial'
        elif any(word in url for word in ['docs', 'documentation', 'api']):
            content_type = 'Documentation'
        elif any(word in url for word in ['news', 'blog', 'article']):
            content_type = 'Article'
        elif any(word in title for word in ['tutorial', 'how to', 'guide']):
            content_type = 'Tutorial'
        
        # Extract topics
        topics = []
        tech_terms = ['python', 'javascript', 'react', 'api', 'ai', 'machine learning']
        for term in tech_terms:
            if term in content:
                topics.append(term.title())
        
        # Domain-based topics
        domain = item['domain'].lower()
        if 'github.com' in domain:
            topics.append('Programming')
        elif 'stackoverflow.com' in domain:
            topics.append('Q&A')
        elif 'medium.com' in domain:
            topics.append('Articles')
        
        if not topics:
            topics = ['General']
        
        # Create summary
        sentences = re.split(r'[.!?]+', item['content'])
        good_sentences = [s.strip() for s in sentences if 30 <= len(s.strip()) <= 200]
        summary = '. '.join(good_sentences[:2]) if good_sentences else f"Content from {item['domain']}"
        
        return {
            'url': item['url'],
            'title': item['title'],
            'summary': summary[:400],
            'content_type': content_type,
            'key_topics': topics[:4],
            'processing_method': 'pattern'
        }

# HTML fetcher
async def fetch_html_fast(url: str, semaphore: asyncio.Semaphore) -> Optional[str]:
    async with semaphore:
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; MindCanvas/1.0)'},
                follow_redirects=True
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

# Initialize components
extractor = FastExtractor()
ai_processor = OpenAIBatchProcessor(OPENAI_API_KEY)

# Main processing pipeline
async def process_urls_efficiently(items: List[HistoryItem]) -> Dict[str, int]:
    """Ultra-efficient URL processing with OpenAI Batch API"""
    
    # Filter valid URLs
    valid_items = [item for item in items if is_valid_url(item.url)]
    logger.info(f"Filtered {len(valid_items)}/{len(items)} valid URLs")
    
    # Check existing and cache
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    new_items = []
    cached_count = 0
    
    for item in valid_items:
        # Check if already processed
        cursor.execute("SELECT id FROM processed_content WHERE url = ?", (item.url,))
        if cursor.fetchone():
            continue
        
        # Check cache
        url_hash = get_url_hash(item.url)
        cursor.execute("SELECT title, summary, content_type, key_topics FROM url_cache WHERE url_hash = ?", (url_hash,))
        cached = cursor.fetchone()
        
        if cached:
            cursor.execute("""
                INSERT INTO processed_content 
                (url, title, summary, content_type, key_topics, visit_timestamp, processing_method)
                VALUES (?, ?, ?, ?, ?, ?, 'cached')
            """, (
                item.url, cached[0], cached[1], cached[2], cached[3],
                datetime.fromtimestamp(item.lastVisitTime / 1000.0)
            ))
            cached_count += 1
        else:
            new_items.append(item)
    
    conn.commit()
    conn.close()
    
    if not new_items:
        logger.info(f"All URLs already processed or cached")
        return {"processed": cached_count, "total": len(items), "cached": cached_count}
    
    logger.info(f"Processing {len(new_items)} new URLs with OpenAI Batch API")
    
    # Fetch HTML content
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    html_tasks = [fetch_html_fast(item.url, semaphore) for item in new_items]
    html_results = await asyncio.gather(*html_tasks, return_exceptions=True)
    
    # Extract content
    content_items = []
    for item, html in zip(new_items, html_results):
        if isinstance(html, str) and html:
            extracted = extractor.extract_clean_text(html, item.url, item.title)
            if extracted:
                extracted['visit_time'] = item.lastVisitTime
                content_items.append(extracted)
    
    logger.info(f"Extracted content from {len(content_items)} URLs")
    
    if not content_items:
        return {"processed": cached_count, "total": len(items), "cached": cached_count}
    
    # Process with OpenAI Batch API
    logger.info(f"Processing {len(content_items)} URLs with gpt-4.1-nano batch API")
    processed_items = await ai_processor.process_content_batch(content_items)
    
    # Store results
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    stored_count = 0
    for original_item, processed_item in zip(content_items, processed_items):
        try:
            # Store in main table
            cursor.execute("""
                INSERT INTO processed_content 
                (url, title, summary, content_type, key_topics, visit_timestamp, processing_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                processed_item['url'],
                processed_item['title'],
                processed_item['summary'],
                processed_item['content_type'],
                json.dumps(processed_item['key_topics']),
                datetime.fromtimestamp(original_item['visit_time'] / 1000.0),
                processed_item['processing_method']
            ))
            
            # Cache for future
            url_hash = get_url_hash(processed_item['url'])
            cursor.execute("""
                INSERT OR REPLACE INTO url_cache 
                (url_hash, title, summary, content_type, key_topics)
                VALUES (?, ?, ?, ?, ?)
            """, (
                url_hash,
                processed_item['title'],
                processed_item['summary'],
                processed_item['content_type'],
                json.dumps(processed_item['key_topics'])
            ))
            
            stored_count += 1
            
        except Exception as e:
            logger.error(f"Failed to store {processed_item['url']}: {e}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Successfully processed and stored {stored_count} URLs")
    
    return {
        "processed": stored_count + cached_count,
        "total": len(items),
        "new": stored_count,
        "cached": cached_count
    }

# API Endpoints
@app.post("/api/ingest")
async def ingest_history(items: List[HistoryItem]):
    """Process URLs with OpenAI Batch API - ONCE ONLY"""
    
    if not items:
        return {"status": "success", "processed": 0}
    
    logger.info(f"Received {len(items)} URLs for batch processing")
    
    results = await process_urls_efficiently(items)
    
    return {
        "status": "success",
        **results,
        "message": f"Processed {results['processed']} URLs ({results['new']} new, {results['cached']} cached)",
        "api_model": "gpt-4.1-nano"
    }

@app.get("/api/content")
async def get_processed_content():
    """Get ALL pre-processed content - instant display"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT url, title, summary, content_type, key_topics, visit_timestamp, processing_method
        FROM processed_content
        ORDER BY visit_timestamp DESC
        LIMIT 1000
    """)
    
    results = []
    for row in cursor.fetchall():
        try:
            results.append({
                "id": len(results) + 1,
                "url": row[0],
                "title": row[1],
                "description": row[2],
                "content_type": row[3],
                "key_details": json.loads(row[4]) if row[4] else [],
                "visit_timestamp": row[5],
                "processing_method": row[6]
            })
        except json.JSONDecodeError:
            continue
    
    conn.close()
    return {"content": results, "total": len(results)}

@app.get("/api/stats")
async def get_stats():
    """Get processing statistics"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    # Total content
    cursor.execute("SELECT COUNT(*) FROM processed_content")
    total = cursor.fetchone()[0]
    
    # By processing method
    cursor.execute("SELECT processing_method, COUNT(*) FROM processed_content GROUP BY processing_method")
    by_method = dict(cursor.fetchall())
    
    # By content type
    cursor.execute("SELECT content_type, COUNT(*) FROM processed_content GROUP BY content_type")
    by_type = dict(cursor.fetchall())
    
    conn.close()
    
    openai_count = by_method.get('openai_batch', 0)
    pattern_count = by_method.get('pattern', 0)
    cached_count = by_method.get('cached', 0)
    
    return {
        "status_counts": {"completed": total, "pending": 0, "failed": 0},
        "content_types": by_type,
        "total_extracted": total,
        "processing_efficiency": {
            "openai_batch_processed": openai_count,
            "pattern_processed": pattern_count,
            "cached_reused": cached_count,
            "batch_api_calls": math.ceil(openai_count / BATCH_SIZE),
            "efficiency_percent": round(((pattern_count + cached_count) / max(total, 1)) * 100, 1)
        },
        "api_model": "gpt-4.1-nano"
    }

@app.get("/api/search")
async def search_content(q: str):
    """Search pre-processed content"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    search_term = f"%{q.lower()}%"
    cursor.execute("""
        SELECT url, title, summary, content_type, key_topics
        FROM processed_content
        WHERE LOWER(title) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(key_topics) LIKE ?
        ORDER BY visit_timestamp DESC
        LIMIT 50
    """, (search_term, search_term, search_term))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "url": row[0],
            "title": row[1],
            "description": row[2],
            "content_type": row[3],
            "key_details": json.loads(row[4]) if row[4] else []
        })
    
    conn.close()
    return {"results": results, "total": len(results)}

@app.delete("/api/reset")
async def reset_database():
    """Reset all data"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processed_content")
    cursor.execute("DELETE FROM url_cache")
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Database reset"}

@app.get("/")
async def root():
    return {
        "message": "MindCanvas Final - OpenAI Batch API",
        "version": "5.0",
        "api_model": "gpt-4.1-nano",
        "features": [
            "openai_batch_api",
            "single_pass_processing", 
            "intelligent_caching",
            "zero_frontend_processing",
            "20_urls_per_batch"
        ],
        "efficiency": {
            "batch_size": BATCH_SIZE,
            "concurrency": CONCURRENCY,
            "processing_model": "Process once with AI, display forever"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")