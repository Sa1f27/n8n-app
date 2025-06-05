import sqlite3
import asyncio
import httpx
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging
from crawler import crawl_and_extract  # Import Crawl4AI logic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Pydantic model for incoming history data
class HistoryItem(BaseModel):
    url: str
    title: str
    lastVisitTime: float

# Initialize SQLite database
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

# Async HTML fetcher with concurrency control
async def fetch_html(url: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.text
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return ""

# POST endpoint to ingest history
@app.post("/api/ingest")
async def ingest_history(items: List[HistoryItem]):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
    tasks = []

    for item in items:
        try:
            # Check if URL already exists
            cursor.execute("SELECT id FROM visited_sites WHERE url = ?", (item.url,))
            if cursor.fetchone():
                logger.info(f"Skipping duplicate URL: {item.url}")
                continue
            # Insert new record without HTML (to be fetched async)
            cursor.execute(
                "INSERT INTO visited_sites (url, title, last_visit_time) VALUES (?, ?, ?)",
                (item.url, item.title, datetime.fromtimestamp(item.lastVisitTime / 1000.0))
            )
            tasks.append(fetch_html(item.url, semaphore))
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate URL skipped: {item.url}")
            continue

    conn.commit()
    conn.close()

    # Fetch HTML content asynchronously
    html_results = await asyncio.gather(*tasks, return_exceptions=True)
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    for item, html in zip([i for i in items if i.url not in [row[0] for row in cursor.execute("SELECT url FROM visited_sites WHERE url = ?", (i.url,)).fetchall()]], html_results):
        if isinstance(html, str) and html:
            cursor.execute(
                "UPDATE visited_sites SET html_content = ? WHERE url = ?",
                (html, item.url)
            )
            logger.info(f"Stored HTML for {item.url}")
    conn.commit()
    conn.close()

    return {"status": "success", "processed": len(items)}

# POST endpoint to trigger Crawl4AI extraction
@app.post("/api/extract")
async def extract_data(url: Optional[str] = None):
    logger.info(f"Starting extraction for {'specific URL: ' + url if url else 'all URLs'}")
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    
    # Fetch URLs to process
    if url:
        cursor.execute("SELECT url, html_content FROM visited_sites WHERE url = ?", (url,))
        results = [cursor.fetchone()]
        if not results[0]:
            conn.close()
            logger.error(f"URL not found: {url}")
            raise HTTPException(status_code=404, detail="URL not found")
    else:
        cursor.execute("SELECT url, html_content FROM visited_sites")
        results = cursor.fetchall()

    conn.close()

    # Process URLs with Crawl4AI
    extracted_results = []
    semaphore = asyncio.Semaphore(5)  # Limit concurrent Crawl4AI tasks
    tasks = []
    
    for row in results:
        url, html_content = row
        tasks.append(crawl_and_extract(url, html_content, semaphore))

    # Run extraction tasks
    extraction_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    for row, result in zip(results, extraction_results):
        url = row[0]
        if isinstance(result, str) and result:
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO extracted_data (url, extracted_json, extraction_timestamp) VALUES (?, ?, ?)",
                    (url, result, datetime.now())
                )
                extracted_data = json.loads(result)
                extracted_results.append({"url": url, "extracted_data": extracted_data})
                logger.info(f"Successfully extracted data for {url}: {extracted_data}")
            except Exception as e:
                logger.error(f"Failed to store extracted data for {url}: {e}")
                extracted_results.append({"url": url, "error": str(e)})
        else:
            logger.error(f"Extraction failed for {url}: {result}")
            extracted_results.append({"url": url, "error": str(result)})
    
    conn.commit()
    conn.close()

    logger.info(f"Extraction completed. Processed {len(extracted_results)} URLs")
    return {"status": "success", "results": extracted_results}

# GET endpoint to retrieve all URLs with extracted data
@app.get("/api/urls")
async def get_urls():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.id, v.url, v.title, v.last_visit_time, e.extracted_json
        FROM visited_sites v
        LEFT JOIN extracted_data e ON v.url = e.url
    """)
    results = [
        {
            "id": row[0],
            "url": row[1],
            "title": row[2],
            "last_visit_time": row[3],
            "extracted_data": json.loads(row[4]) if row[4] else None
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return results

# GET endpoint to retrieve HTML by URL
@app.get("/api/html/{url:path}")
async def get_html_by_url(url: str):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT html_content FROM visited_sites WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    if not result or not result[0]:
        logger.error(f"HTML not found for URL: {url}")
        raise HTTPException(status_code=404, detail="HTML not found for URL")
    return {"url": url, "html_content": result[0]}

# GET endpoint to retrieve HTML by ID
@app.get("/api/html/id/{id}")
async def get_html_by_id(id: int):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT url, html_content FROM visited_sites WHERE id = ?", (id,))
    result = cursor.fetchone()
    conn.close()
    if not result or not result[1]:
        logger.error(f"HTML not found for ID: {id}")
        raise HTTPException(status_code=404, detail="HTML not found for ID")
    return {"url": result[0], "html_content": result[1]}
