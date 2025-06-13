History Exporter Project
A Chrome extension (Manifest V3, TypeScript) and FastAPI backend (Python) to export browsing history (URLs, titles, visit times) from the last 24 hours, store it in a SQLite database, asynchronously fetch and store raw HTML content, and extract structured data using Crawl4AI.
Project Structure
history-exporter-project/
├── history-exporter-extension/  # Chrome extension
│   ├── manifest.json
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── background.ts
│   │   ├── popup.ts
│   │   ├── types.ts
│   │   ├── popup.html
│   ├── dist/ (generated)
├── history-backend/            # FastAPI backend
│   ├── main.py
│   ├── crawler.py
│   ├── requirements.txt
│   ├── history.db (generated)
├── README.md

Prerequisites

Node.js (v16 or later) and npm
Python (3.8 or later)
Google Chrome browser
SQLite (included with Python)
Crawl4AI: Ensure internet access for Crawl4AI dependencies

Setup Instructions
1. Chrome Extension Setup

Navigate to the extension directory:cd exporter-extension


Build the extension:npm run build

This compiles TypeScript files and copies popup.html and manifest.json to the extension-exporter/ folder.
Load the extension in Chrome:
Open Chrome and go to chrome://extensions/.
Enable "Developer mode" (top right).
Click "Load unpacked" and select the history-exporter-extension/dist folder.
Approve the history permission when prompted.



2. FastAPI Backend Setup

Navigate to the backend directory:cd history-exporter-project/history-backend


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Start the FastAPI server:python main.py

The server will be accessible at http://localhost:8000.

Running and Testing

Start the Backend:
Ensure the FastAPI server is running (uvicorn main:app --host 0.0.0.0 --port 8000).


Test the Extension:
Visit a few websites in Chrome (e.g., https://example.com, https://www.bbc.com) to generate history.
Click the extension icon to open the popup.
Click "Export History" to send history to the backend.
Check the Chrome console (chrome://extensions/, "Inspect views: background page") for logs.


Test Crawl4AI Extraction:
Trigger extraction for all URLs:curl -X POST "http://localhost:8000/api/extract"


Trigger extraction for a specific URL:curl -X POST "http://localhost:8000/api/extract?url=https://www.bbc.com"


Use a custom schema description (optional):curl -X POST "http://localhost:8000/api/extract?url=https://www.bbc.com" -H "Content-Type: application/json" -d '{"schema_description": "Extract title, author, and date from news articles"}'


Check extracted data in the database:sqlite3 history-backend/history.db
SELECT * FROM extracted_data;




Verify Backend Data:
Get all URLs with extracted data:curl http://localhost:8000/api/urls


Get HTML by URL:curl http://localhost:8000/api/html/https://example.com


Get HTML by ID:curl http://localhost:8000/api/html/id/1


Use Postman or a browser to test endpoints.



Troubleshooting

Extension Errors: Check the background page console for network or permission issues.
Backend Errors: Review server logs for HTTP, database, or Crawl4AI errors. Ensure history.db is writable.
CORS Issues: The extension’s host_permissions allow http://localhost:8000/*. Adjust if the backend URL changes.
HTML Fetching Failures: Reduce the Semaphore limit in main.py (e.g., from 5 to 3) or increase the timeout.
Crawl4AI Issues: Ensure crawl4ai is installed (pip install crawl4ai) and internet access is available. Check logs for extraction errors.

Notes

The extension requires the history permission, which Chrome prompts users to approve.
For production, add authentication to FastAPI endpoints and deploy to a cloud provider.
Test with simple URLs (e.g., https://example.com) to avoid anti-bot measures.
Crawl4AI is used to extract structured data (e.g., headlines, dates, images) from news sites by default. Update the schema description in /api/extract for other domains.

