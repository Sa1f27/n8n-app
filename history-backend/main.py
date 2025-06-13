import sqlite3
import asyncio
import httpx
import json
import os
import logging
import re
import hashlib
import math
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bs4 import BeautifulSoup, Tag
from openai import OpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from groq import Groq
import time

# Import your crawler
try:
    from crawler import crawl_and_extract
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logging.warning("Crawl4AI not available, using basic extraction only")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONCURRENT_TASKS = 5
BATCH_SIZE = 15  # Optimized for Groq
CONCURRENCY = 3
MAX_OUTPUT_TOKENS = 1500  # Optimized for summaries
MAX_CONTENT_LENGTH = 2000  # Increased as per docs

# Quality Control
MIN_CONTENT_LENGTH = 50
MIN_TITLE_LENGTH = 5
QUALITY_SCORE_RANGE = (1, 10)

# "No Data" Detection Patterns
NO_DATA_PATTERNS = [
    r"no data available",
    r"insufficient content", 
    r"unable to extract",
    r"access denied",
    r"page not available",
    r"content not found",
    r"error accessing",
    r"failed to load",
    r"no meaningful content",
    r"cannot process"
]

# Cache Configuration
CACHE_TTL_DAYS = 7
L1_CACHE_SIZE = 100  # In-memory cache for recent URLs

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
    '.article-content', '[role="main"]', '.post', '.article'
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

# L1 Cache (In-memory)
class L1Cache:
    def __init__(self, max_size: int = L1_CACHE_SIZE):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest accessed item
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

# Initialize L1 Cache
l1_cache = L1Cache()

# Enhanced Database schema
def init_db():
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    # Check if new columns exist, if not add them
    cursor.execute("PRAGMA table_info(processed_content)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Add new columns if they don't exist (backward compatibility)
    if 'quality_score' not in columns:
        cursor.execute("ALTER TABLE processed_content ADD COLUMN quality_score INTEGER DEFAULT 5")
    if 'content_hash' not in columns:
        cursor.execute("ALTER TABLE processed_content ADD COLUMN content_hash TEXT")
    if 'retry_count' not in columns:
        cursor.execute("ALTER TABLE processed_content ADD COLUMN retry_count INTEGER DEFAULT 0")
    if 'processing_status' not in columns:
        cursor.execute("ALTER TABLE processed_content ADD COLUMN processing_status TEXT DEFAULT 'completed'")
    if 'error_message' not in columns:
        cursor.execute("ALTER TABLE processed_content ADD COLUMN error_message TEXT")
    
    # Enhanced cache table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE,
            url_hash TEXT,
            title TEXT,
            summary TEXT,
            content_type TEXT,
            key_topics TEXT,
            quality_score INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            hit_count INTEGER DEFAULT 1
        )
    """)
    
    # Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON processed_content(content_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_status ON processed_content(processing_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON content_cache(expires_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_hash ON content_cache(content_hash)")
    
    conn.commit()
    conn.close()

init_db()

# Utility functions
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
    """Generate URL hash for duplicate detection"""
    normalized_url = url.split('?')[0].split('#')[0].lower().strip('/')
    return hashlib.sha256(normalized_url.encode()).hexdigest()[:16]

def get_content_hash(content: str) -> str:
    """Generate content hash for similarity detection"""
    # Use first 500 chars for content similarity
    sample = re.sub(r'\s+', ' ', content[:500]).strip().lower()
    return hashlib.sha256(sample.encode()).hexdigest()[:16]

def clean_title(title: str) -> str:
    if not title:
        return "Untitled"
    
    suffixes = [' - Google Search', ' - Bing', ' | Facebook', ' | Twitter', ' | LinkedIn']
    for suffix in suffixes:
        if title.endswith(suffix):
            title = title[:-len(suffix)]
    
    return title.strip()[:100] or "Untitled"

def calculate_quality_score(title: str, content: str, url: str) -> int:
    """Calculate content quality score (1-10)"""
    score = 5  # Base score
    
    # Title quality
    if len(title) > MIN_TITLE_LENGTH:
        score += 1
    if len(title) > 20:
        score += 1
    
    # Content quality  
    if len(content) > 200:
        score += 1
    if len(content) > 500:
        score += 1
    
    # URL quality
    if any(indicator in url.lower() for indicator in ['docs', 'tutorial', 'guide', 'article']):
        score += 1
    
    # Penalty for low quality indicators
    if len(content) < MIN_CONTENT_LENGTH:
        score -= 2
    if len(title) < MIN_TITLE_LENGTH:
        score -= 1
    
    return max(1, min(10, score))

def detect_no_data_response(text: str) -> bool:
    """Detect if LLM response indicates no data"""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in NO_DATA_PATTERNS)

# Enhanced Content Extractor with hybrid approach
class HybridContentExtractor:
    def __init__(self):
        self.crawl4ai_semaphore = asyncio.Semaphore(2)  # Limit Crawl4AI concurrent tasks
    
    async def extract_content(self, url: str, title: str, html: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Extract content using hybrid approach"""
        
        # Try fast extraction first
        if html:
            fast_result = self.fast_extract(html, url, title)
            if fast_result and self.is_good_extraction(fast_result):
                fast_result['extraction_method'] = 'fast'
                return fast_result
        
        # Fallback to Crawl4AI if available and fast extraction failed
        if CRAWL4AI_AVAILABLE:
            try:
                crawl_result = await self.crawl4ai_extract(url, html)
                if crawl_result and self.is_good_extraction(crawl_result):
                    crawl_result['extraction_method'] = 'crawl4ai'
                    return crawl_result
            except Exception as e:
                logger.warning(f"Crawl4AI extraction failed for {url}: {e}")
        
        # Final fallback - use fast extraction even if low quality
        if html:
            fast_result = self.fast_extract(html, url, title)
            if fast_result:
                fast_result['extraction_method'] = 'fast_fallback'
                return fast_result
        
        return None
    
    def fast_extract(self, html: str, url: str, title: str) -> Optional[Dict[str, str]]:
        """Fast extraction using BeautifulSoup"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                if hasattr(tag, 'decompose'):
                    tag.decompose()
            
            # Find main content
            main_content = None
            for selector in CONTENT_SELECTORS:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 50:  # Reduced threshold
                    main_content = element
                    break
            
            if not main_content:
                main_content = soup.body
            
            if not main_content:
                # Try to get any text content as fallback
                main_content = soup
            
            content_text = main_content.get_text(separator=' ', strip=True)
            content_text = re.sub(r'\s+', ' ', content_text)
            
            # More lenient minimum content check
            if len(content_text) < 20:
                logger.warning(f"Content too short for {url}: {len(content_text)} chars")
                return None
            
            return {
                'url': url,
                'title': clean_title(title),
                'content': content_text[:MAX_CONTENT_LENGTH],
                'domain': urlparse(url).netloc
            }
            
        except Exception as e:
            logger.error(f"Fast extraction failed for {url}: {e}")
            return None
    
    async def crawl4ai_extract(self, url: str, html: Optional[str]) -> Optional[Dict[str, str]]:
        """Extract using Crawl4AI"""
        try:
            result_json = await crawl_and_extract(url, html, self.crawl4ai_semaphore)
            result_data = json.loads(result_json)
            
            title = result_data.get('headline', '') or urlparse(url).path.split('/')[-1]
            content = result_data.get('summary', '')
            
            if content and len(content) > MIN_CONTENT_LENGTH:
                return {
                    'url': url,
                    'title': clean_title(title),
                    'content': content[:MAX_CONTENT_LENGTH],
                    'domain': urlparse(url).netloc
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Crawl4AI extraction failed for {url}: {e}")
            return None
    
    def is_good_extraction(self, result: Dict[str, str]) -> bool:
        """Check if extraction result is good quality"""
        if not result:
            return False
        
        content_len = len(result.get('content', ''))
        title_len = len(result.get('title', ''))
        
        # More lenient validation for testing
        return (content_len >= 30 and  # Reduced from MIN_CONTENT_LENGTH (50)
                title_len >= 3)        # Reduced from MIN_TITLE_LENGTH (5)

# Dual LLM Processor (Groq primary, OpenAI fallback)
class DualLLMProcessor:
    def __init__(self, groq_api_key: str, openai_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        if not self.groq_client and not self.openai_client:
            logger.warning("No LLM APIs available, using pattern-based processing only")
    
    async def process_content_batch(self, content_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process content using dual LLM strategy"""
        
        if not content_items:
            return []
        
        # Try Groq first (primary)
        if self.groq_client:
            try:
                logger.info(f"Processing {len(content_items)} items with Groq Llama")
                return await self._process_with_groq(content_items)
            except Exception as e:
                logger.warning(f"Groq processing failed: {e}, falling back to OpenAI")
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                logger.info(f"Processing {len(content_items)} items with OpenAI")
                return await self._process_with_openai(content_items)
            except Exception as e:
                logger.error(f"OpenAI processing failed: {e}, using pattern-based processing")
        
        # Final fallback to pattern-based processing
        return [self._pattern_process(item) for item in content_items]
    
    async def _process_with_groq(self, content_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process with Groq Llama"""
        results = []
        
        # Process in batches
        for i in range(0, len(content_items), BATCH_SIZE):
            batch = content_items[i:i + BATCH_SIZE]
            batch_results = await self._groq_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _groq_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process single batch with Groq"""
        tasks = []
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        for item in batch:
            task = self._groq_single(item, semaphore)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _groq_single(self, item: Dict[str, str], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process single item with Groq"""
        async with semaphore:
            try:
                prompt = self._create_prompt(item)
                
                # Retry logic for Groq
                for attempt in range(3):
                    try:
                        response = self.groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=MAX_OUTPUT_TOKENS,
                            temperature=0.1
                        )
                        
                        response_text = response.choices[0].message.content.strip()
                        
                        # Check for "no data" response
                        if detect_no_data_response(response_text):
                            logger.info(f"No data detected for {item['url']}, excluding")
                            return None  # Will be filtered out
                        
                        return self._parse_llm_response(item, response_text, 'groq')
                        
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            raise e
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Groq processing failed for {item['url']}: {e}")
                return self._pattern_process(item)
    
    async def _process_with_openai(self, content_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process with OpenAI (your existing working code)"""
        # Use your existing OpenAI batch processing logic here
        # For now, using similar structure but with OpenAI client
        results = []
        
        for i in range(0, len(content_items), BATCH_SIZE):
            batch = content_items[i:i + BATCH_SIZE]
            batch_results = await self._openai_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _openai_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process batch with OpenAI"""
        tasks = []
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        for item in batch:
            task = self._openai_single(item, semaphore)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _openai_single(self, item: Dict[str, str], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process single item with OpenAI"""
        async with semaphore:
            try:
                prompt = self._create_prompt(item)
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Check for "no data" response
                if detect_no_data_response(response_text):
                    logger.info(f"No data detected for {item['url']}, excluding")
                    return None
                
                return self._parse_llm_response(item, response_text, 'openai')
                
            except Exception as e:
                logger.error(f"OpenAI processing failed for {item['url']}: {e}")
                return self._pattern_process(item)
    
    def _create_prompt(self, item: Dict[str, str]) -> str:
        """Create prompt for LLM processing"""
        return f"""Analyze this webpage content and respond with JSON only:

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

If content is insufficient or meaningless, respond: "no data available"

JSON only, no explanation:"""
    
    def _parse_llm_response(self, item: Dict[str, str], response: str, method: str) -> Dict[str, Any]:
        """Parse LLM response to structured data"""
        try:
            # Clean JSON response
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
            
            # Calculate quality score
            quality_score = calculate_quality_score(
                parsed.get('title', item['title']),
                item['content'],
                item['url']
            )
            
            return {
                'url': item['url'],
                'title': parsed.get('title', item['title'])[:100],
                'summary': parsed.get('summary', '')[:500],
                'content_type': parsed.get('content_type', 'Web Content'),
                'key_topics': parsed.get('key_topics', ['General'])[:4],
                'quality_score': quality_score,
                'processing_method': method,
                'content_hash': get_content_hash(item['content'])
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for {item['url']}: {e}")
            return self._pattern_process(item)
    
    def _pattern_process(self, item: Dict[str, str]) -> Dict[str, Any]:
        """Pattern-based processing fallback"""
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
        tech_terms = ['python', 'javascript', 'react', 'api', 'ai', 'machine learning', 'web development']
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
        
        quality_score = calculate_quality_score(item['title'], item['content'], item['url'])
        
        return {
            'url': item['url'],
            'title': item['title'],
            'summary': summary[:400],
            'content_type': content_type,
            'key_topics': topics[:4],
            'quality_score': quality_score,
            'processing_method': 'pattern',
            'content_hash': get_content_hash(item['content'])
        }

# Enhanced HTML fetcher
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
extractor = HybridContentExtractor()
ai_processor = DualLLMProcessor(GROQ_API_KEY, OPENAI_API_KEY)

# Enhanced cache management
def get_from_cache(url_hash: str) -> Optional[Dict[str, Any]]:
    """Get from L1 cache first, then L2 database cache"""
    
    # Check L1 cache
    l1_result = l1_cache.get(url_hash)
    if l1_result:
        return l1_result
    
    # Check L2 cache (database)
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    # Clean expired cache entries
    cursor.execute("DELETE FROM content_cache WHERE expires_at < ?", (datetime.now(),))
    
    cursor.execute("""
        SELECT title, summary, content_type, key_topics, quality_score, hit_count
        FROM content_cache 
        WHERE content_hash = ? AND expires_at > ?
    """, (url_hash, datetime.now()))
    
    cached = cursor.fetchone()
    if cached:
        # Update hit count
        cursor.execute("UPDATE content_cache SET hit_count = hit_count + 1 WHERE content_hash = ?", (url_hash,))
        conn.commit()
        
        result = {
            'title': cached[0],
            'summary': cached[1], 
            'content_type': cached[2],
            'key_topics': json.loads(cached[3]) if cached[3] else [],
            'quality_score': cached[4],
            'processing_method': 'cached'
        }
        
        # Store in L1 cache for future
        l1_cache.set(url_hash, result)
        conn.close()
        return result
    
    conn.close()
    return None

def store_in_cache(content_hash: str, processed_item: Dict[str, Any]):
    """Store in both L1 and L2 cache"""
    
    # Store in L1 cache
    l1_cache.set(content_hash, processed_item)
    
    # Store in L2 cache (database)
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    expires_at = datetime.now() + timedelta(days=CACHE_TTL_DAYS)
    
    cursor.execute("""
        INSERT OR REPLACE INTO content_cache 
        (content_hash, url_hash, title, summary, content_type, key_topics, quality_score, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        content_hash,
        get_url_hash(processed_item['url']),
        processed_item['title'],
        processed_item['summary'],
        processed_item['content_type'],
        json.dumps(processed_item['key_topics']),
        processed_item['quality_score'],
        expires_at
    ))
    
    conn.commit()
    conn.close()

# Main processing pipeline with enhancements
async def process_urls_efficiently(items: List[HistoryItem]) -> Dict[str, int]:
    """Enhanced URL processing with hybrid extraction and dual LLM"""
    
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
        url_hash = get_url_hash(item.url)
        cursor.execute("SELECT id FROM processed_content WHERE url = ?", (item.url,))
        if cursor.fetchone():
            continue
        
        # Check cache
        cached = get_from_cache(url_hash)
        if cached:
            cursor.execute("""
                INSERT INTO processed_content 
                (url, title, summary, content_type, key_topics, visit_timestamp, processing_method, quality_score, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.url, cached['title'], cached['summary'], cached['content_type'],
                json.dumps(cached['key_topics']), datetime.fromtimestamp(item.lastVisitTime / 1000.0),
                'cached', cached['quality_score'], url_hash
            ))
            cached_count += 1
        else:
            new_items.append(item)
    
    conn.commit()
    conn.close()
    
    if not new_items:
        logger.info(f"All URLs already processed or cached")
        return {"processed": cached_count, "total": len(items), "cached": cached_count, "new": 0}
    
    logger.info(f"Processing {len(new_items)} new URLs with hybrid extraction + dual LLM")
    
    # Fetch HTML content
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    html_tasks = [fetch_html_fast(item.url, semaphore) for item in new_items]
    html_results = await asyncio.gather(*html_tasks, return_exceptions=True)
    
    # Extract content using hybrid approach
    content_items = []
    extraction_stats = {"fast_success": 0, "crawl4ai_success": 0, "failed": 0}
    
    for item, html in zip(new_items, html_results):
        if isinstance(html, str) and html:
            try:
                # Debug: Log HTML sample
                html_sample = html[:200] + "..." if len(html) > 200 else html
                logger.debug(f"HTML sample for {item.url}: {html_sample}")
                
                extracted = await extractor.extract_content(item.url, item.title, html)
                if extracted:
                    extracted['visit_time'] = item.lastVisitTime
                    content_items.append(extracted)
                    
                    # Track extraction method
                    method = extracted.get('extraction_method', 'unknown')
                    if 'fast' in method:
                        extraction_stats["fast_success"] += 1
                    elif 'crawl4ai' in method:
                        extraction_stats["crawl4ai_success"] += 1
                    
                    logger.info(f"Successfully extracted content from {item.url} using {method}")
                else:
                    extraction_stats["failed"] += 1
                    logger.warning(f"Content extraction failed for {item.url} - no meaningful content found")
            except Exception as e:
                extraction_stats["failed"] += 1
                logger.error(f"Content extraction error for {item.url}: {e}")
        else:
            extraction_stats["failed"] += 1
            if isinstance(html, Exception):
                logger.warning(f"HTML fetch failed for {item.url}: {html}")
            else:
                logger.warning(f"HTML fetch failed for {item.url} - no content returned")
    
    logger.info(f"Extraction stats: {extraction_stats}")
    logger.info(f"Extracted content from {len(content_items)} URLs")
    
    if not content_items:
        return {
            "processed": cached_count, 
            "total": len(items), 
            "cached": cached_count, 
            "new": 0,
            "excluded": 0
        }
    
    # Process with Dual LLM (Groq primary, OpenAI fallback)
    logger.info(f"Processing {len(content_items)} URLs with Groq/OpenAI dual LLM")
    processed_items = await ai_processor.process_content_batch(content_items)
    
    # Filter out None results (no data responses)
    valid_processed_items = []
    excluded_count = 0
    
    for original_item, processed_item in zip(content_items, processed_items):
        if processed_item is None:
            excluded_count += 1
            logger.info(f"Excluded URL (no data): {original_item['url']}")
        else:
            valid_processed_items.append((original_item, processed_item))
    
    logger.info(f"Valid processed items: {len(valid_processed_items)}, Excluded: {excluded_count}")
    
    # Store results
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    stored_count = 0
    for original_item, processed_item in valid_processed_items:
        try:
            # Store in main table
            cursor.execute("""
                INSERT INTO processed_content 
                (url, title, summary, content_type, key_topics, visit_timestamp, processing_method, 
                 quality_score, content_hash, processing_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed')
            """, (
                processed_item['url'],
                processed_item['title'],
                processed_item['summary'],
                processed_item['content_type'],
                json.dumps(processed_item['key_topics']),
                datetime.fromtimestamp(original_item['visit_time'] / 1000.0),
                processed_item['processing_method'],
                processed_item['quality_score'],
                processed_item['content_hash']
            ))
            
            # Store in cache for future use
            store_in_cache(processed_item['content_hash'], processed_item)
            
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
        "cached": cached_count,
        "excluded": excluded_count
    }

# API Endpoints
@app.post("/api/ingest")
async def ingest_history(items: List[HistoryItem]):
    """Process URLs with Dual LLM (Groq/OpenAI) and hybrid extraction"""
    
    if not items:
        return {"status": "success", "processed": 0, "total": 0, "new": 0, "cached": 0, "excluded": 0}
    
    logger.info(f"Received {len(items)} URLs for dual LLM processing")
    
    try:
        results = await process_urls_efficiently(items)
        
        # Ensure all keys exist with defaults
        processed = results.get('processed', 0)
        total = results.get('total', len(items))
        new = results.get('new', 0)
        cached = results.get('cached', 0)
        excluded = results.get('excluded', 0)
        
        return {
            "status": "success",
            "processed": processed,
            "total": total,
            "new": new,
            "cached": cached,
            "excluded": excluded,
            "message": f"Processed {processed} URLs ({new} new, {cached} cached, {excluded} excluded)",
            "api_model": "groq_llama_openai_fallback"
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "processed": 0,
            "total": len(items),
            "new": 0,
            "cached": 0,
            "excluded": 0,
            "message": f"Processing failed: {str(e)}"
        }

@app.get("/api/content")
async def get_processed_content(limit: int = 1000, min_quality: int = 1):
    """Get processed content with quality filtering"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT url, title, summary, content_type, key_topics, visit_timestamp, 
               processing_method, quality_score, processing_status
        FROM processed_content
        WHERE quality_score >= ? AND processing_status = 'completed'
        ORDER BY quality_score DESC, visit_timestamp DESC
        LIMIT ?
    """, (min_quality, limit))
    
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
                "processing_method": row[6],
                "quality_score": row[7],
                "status": row[8]
            })
        except json.JSONDecodeError:
            continue
    
    conn.close()
    return {"content": results, "total": len(results)}

@app.get("/api/categories/available")
async def get_available_categories():
    """Get categories that have content with counts"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT content_type, COUNT(*) as count
        FROM processed_content 
        WHERE processing_status = 'completed'
        GROUP BY content_type
        ORDER BY count DESC
    """)
    
    content_types = dict(cursor.fetchall())
    
    # Get key topics
    cursor.execute("SELECT key_topics FROM processed_content WHERE processing_status = 'completed'")
    all_topics = {}
    
    for row in cursor.fetchall():
        try:
            topics = json.loads(row[0]) if row[0] else []
            for topic in topics:
                all_topics[topic] = all_topics.get(topic, 0) + 1
        except:
            continue
    
    conn.close()
    
    # Format response
    categories = []
    
    # Add content types
    for content_type, count in content_types.items():
        categories.append({
            "name": content_type,
            "count": count,
            "type": "content_type"
        })
    
    # Add top topics
    sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:15]
    for topic, count in sorted_topics:
        categories.append({
            "name": topic,
            "count": count,
            "type": "topic"
        })
    
    return {
        "categories": categories,
        "total_categories": len(categories),
        "content_types": content_types,
        "top_topics": dict(sorted_topics)
    }

@app.get("/api/content/filter")
async def filter_content(
    categories: str = "", 
    quality_min: int = 1, 
    limit: int = 100,
    content_type: str = ""
):
    """Filter content by categories, quality, and content type"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    # Build dynamic query
    where_conditions = ["processing_status = 'completed'", "quality_score >= ?"]
    params = [quality_min]
    
    if content_type:
        where_conditions.append("content_type = ?")
        params.append(content_type)
    
    if categories:
        category_list = [cat.strip() for cat in categories.split(',') if cat.strip()]
        if category_list:
            # Search in key_topics JSON
            topic_conditions = " OR ".join(["key_topics LIKE ?" for _ in category_list])
            where_conditions.append(f"({topic_conditions})")
            params.extend([f'%"{cat}"%' for cat in category_list])
    
    where_clause = " AND ".join(where_conditions)
    params.append(limit)
    
    cursor.execute(f"""
        SELECT url, title, summary, content_type, key_topics, visit_timestamp, 
               processing_method, quality_score
        FROM processed_content
        WHERE {where_clause}
        ORDER BY quality_score DESC, visit_timestamp DESC
        LIMIT ?
    """, params)
    
    results = []
    for row in cursor.fetchall():
        try:
            results.append({
                "url": row[0],
                "title": row[1],
                "description": row[2],
                "content_type": row[3],
                "key_details": json.loads(row[4]) if row[4] else [],
                "visit_timestamp": row[5],
                "processing_method": row[6],
                "quality_score": row[7]
            })
        except:
            continue
    
    conn.close()
    
    return {
        "data": results,
        "total": len(results),
        "filters_applied": {
            "categories": categories,
            "quality_min": quality_min,
            "content_type": content_type
        }
    }

@app.get("/api/stats")
async def get_enhanced_stats():
    """Get enhanced processing statistics"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    # Total content
    cursor.execute("SELECT COUNT(*) FROM processed_content WHERE processing_status = 'completed'")
    total = cursor.fetchone()[0]
    
    # By processing method
    cursor.execute("""
        SELECT processing_method, COUNT(*) 
        FROM processed_content 
        WHERE processing_status = 'completed'
        GROUP BY processing_method
    """)
    by_method = dict(cursor.fetchall())
    
    # By content type
    cursor.execute("""
        SELECT content_type, COUNT(*) 
        FROM processed_content 
        WHERE processing_status = 'completed'
        GROUP BY content_type
    """)
    by_type = dict(cursor.fetchall())
    
    # Quality distribution
    cursor.execute("""
        SELECT quality_score, COUNT(*) 
        FROM processed_content 
        WHERE processing_status = 'completed'
        GROUP BY quality_score
        ORDER BY quality_score
    """)
    quality_dist = dict(cursor.fetchall())
    
    # Cache performance
    cursor.execute("SELECT COUNT(*), AVG(hit_count) FROM content_cache WHERE expires_at > ?", (datetime.now(),))
    cache_info = cursor.fetchone()
    cache_entries = cache_info[0] if cache_info[0] else 0
    avg_hits = round(cache_info[1], 2) if cache_info[1] else 0
    
    # Processing efficiency
    groq_count = by_method.get('groq', 0)
    openai_count = by_method.get('openai', 0)
    pattern_count = by_method.get('pattern', 0)
    cached_count = by_method.get('cached', 0)
    
    conn.close()
    
    return {
        "status_counts": {"completed": total, "pending": 0, "failed": 0},
        "content_types": by_type,
        "quality_distribution": quality_dist,
        "total_extracted": total,
        "processing_efficiency": {
            "groq_processed": groq_count,
            "openai_fallback": openai_count,
            "pattern_processed": pattern_count,
            "cached_reused": cached_count,
            "total_llm_calls": groq_count + openai_count,
            "cache_hit_rate": round((cached_count / max(total, 1)) * 100, 1),
            "groq_success_rate": round((groq_count / max(groq_count + openai_count, 1)) * 100, 1)
        },
        "cache_performance": {
            "active_entries": cache_entries,
            "average_hits_per_entry": avg_hits,
            "l1_cache_size": len(l1_cache.cache)
        },
        "api_models": ["groq_llama_3.1_8b", "openai_gpt_4o_mini_fallback"],
        "extraction_methods": ["hybrid_fast_crawl4ai"]
    }

@app.get("/api/search")
async def search_content(q: str, limit: int = 50):
    """Enhanced search with quality scoring"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    search_term = f"%{q.lower()}%"
    cursor.execute("""
        SELECT url, title, summary, content_type, key_topics, quality_score, processing_method
        FROM processed_content
        WHERE processing_status = 'completed' AND (
            LOWER(title) LIKE ? OR 
            LOWER(summary) LIKE ? OR 
            LOWER(key_topics) LIKE ?
        )
        ORDER BY quality_score DESC, visit_timestamp DESC
        LIMIT ?
    """, (search_term, search_term, search_term, limit))
    
    results = []
    for row in cursor.fetchall():
        try:
            results.append({
                "url": row[0],
                "title": row[1],
                "description": row[2],
                "content_type": row[3],
                "key_details": json.loads(row[4]) if row[4] else [],
                "quality_score": row[5],
                "processing_method": row[6]
            })
        except:
            continue
    
    conn.close()
    return {
        "results": results, 
        "total": len(results),
        "query": q,
        "sorted_by": "quality_score_desc"
    }

@app.delete("/api/reset")
async def reset_database():
    """Reset all data including cache"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processed_content")
    cursor.execute("DELETE FROM content_cache")
    cursor.execute("DELETE FROM url_cache")  # Old cache table
    conn.commit()
    conn.close()
    
    # Clear L1 cache
    l1_cache.clear()
    
    return {"status": "success", "message": "Database and caches reset"}

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    
    # L2 cache stats
    cursor.execute("SELECT COUNT(*) FROM content_cache WHERE expires_at > ?", (datetime.now(),))
    active_cache = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM content_cache WHERE expires_at <= ?", (datetime.now(),))
    expired_cache = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(hit_count) FROM content_cache WHERE expires_at > ?", (datetime.now(),))
    avg_hits = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "l1_cache": {
            "size": len(l1_cache.cache),
            "max_size": l1_cache.max_size,
            "utilization": round((len(l1_cache.cache) / l1_cache.max_size) * 100, 1)
        },
        "l2_cache": {
            "active_entries": active_cache,
            "expired_entries": expired_cache,
            "average_hits": round(avg_hits, 2),
            "ttl_days": CACHE_TTL_DAYS
        }
    }

@app.post("/api/cache/clear")
async def clear_caches():
    """Clear all caches"""
    # Clear L1 cache
    l1_cache.clear()
    
    # Clear expired L2 cache entries
    conn = sqlite3.connect("mindcanvas.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM content_cache WHERE expires_at <= ?", (datetime.now(),))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return {
        "status": "success", 
        "message": f"Caches cleared. Removed {deleted} expired entries",
        "l1_cleared": True,
        "l2_expired_removed": deleted
    }

@app.get("/")
async def root():
    return {
        "message": "MindCanvas Enhanced - Dual LLM + Hybrid Extraction",
        "version": "6.0",
        "features": [
            "groq_llama_primary_openai_fallback",
            "hybrid_extraction_crawl4ai",
            "multi_level_caching_l1_l2", 
            "quality_scoring_no_data_detection",
            "smart_content_validation",
            "enhanced_filtering_categories"
        ],
        "llm_strategy": {
            "primary": "groq_llama_3.1_8b_instant",
            "fallback": "openai_gpt_4o_mini",
            "pattern_fallback": "enabled"
        },
        "extraction_strategy": {
            "primary": "httpx_beautifulsoup_fast",
            "fallback": "crawl4ai_javascript_heavy_sites",
            "quality_validation": "enabled"
        },
        "caching": {
            "l1_memory": f"{L1_CACHE_SIZE}_recent_urls",
            "l2_database": f"{CACHE_TTL_DAYS}_day_ttl",
            "duplicate_detection": "url_content_hash"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")