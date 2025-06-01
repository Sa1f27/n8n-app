# history-backend/main.py
import sqlite3
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

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
                continue  # Skip if URL exists
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
    conn.commit()
    conn.close()

    return {"status": "success", "processed": len(items)}

# GET endpoint to retrieve all URLs
@app.get("/api/urls")
async def get_urls():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, title, last_visit_time FROM visited_sites")
    results = [{"id": row[0], "url": row[1], "title": row[2], "last_visit_time": row[3]} for row in cursor.fetchall()]
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
        raise HTTPException(status_code=404, detail="HTML not found for ID")
    return {"url": result[0], "html_content": result[1]}