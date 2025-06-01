History Exporter Project
A Chrome extension (Manifest V3, TypeScript) and FastAPI backend (Python) to export browsing history (URLs, titles, visit times) from the last 24 hours, store it in a SQLite database, and asynchronously fetch and store raw HTML content.
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
│   ├── requirements.txt
│   ├── history.db (generated)
├── README.md

Prerequisites

Node.js (v16 or later) and npm
Python (3.8 or later)
Google Chrome browser
SQLite (included with Python)

Setup Instructions
1. Chrome Extension Setup

Navigate to the extension directory:cd history-exporter-project/history-exporter-extension


Install dependencies:npm install


Build the extension:npm run build

This compiles TypeScript files and copies popup.html and manifest.json to the dist/ folder.
Load the extension in Chrome:
Open Chrome and go to chrome://extensions/.
Enable "Developer mode" (top right).
Click "Load unpacked" and select the history-exporter-extension/dist folder.
The extension icon should appear in the toolbar.



2. FastAPI Backend Setup

Navigate to the backend directory:cd history-exporter-project/history-backend


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Start the FastAPI server:uvicorn main:app --host 0.0.0.0 --port 8000

The server will be accessible at http://localhost:8000.

Running and Testing

Start the Backend:
Ensure the FastAPI server is running (uvicorn main:app --host 0.0.0.0 --port 8000).


Test the Extension:
Visit a few websites in Chrome to generate browsing history.
Click the extension icon to open the popup.
Click "Export History" to send the last 24 hours of history to the backend.
Check the Chrome console (chrome://extensions/, "Inspect views: background page") for logs.


Verify Backend Data:
Check the SQLite database (history.db) using:sqlite3 history-backend/history.db
SELECT * FROM visited_sites;


Test API endpoints using curl or a browser:
Get all URLs: curl http://localhost:8000/api/urls
Get HTML by URL: curl http://localhost:8000/api/html/https://example.com
Get HTML by ID: curl http://localhost:8000/api/html/id/1


Use Postman to send a test POST request to /api/ingest:curl -X POST "http://localhost:8000/api/ingest" -H "Content-Type: application/json" -d '[{"url":"https://example.com","title":"Example","lastVisitTime":1622559600000}]'





Troubleshooting

Extension Errors: Check the background page console for network or permission issues.
Backend Errors: Review server logs for HTTP or database errors. Ensure history.db is writable.
CORS Issues: The extension’s host_permissions allow http://localhost:8000/*. Adjust if the backend URL changes.
HTML Fetching Failures: Reduce the Semaphore limit in main.py (e.g., from 5 to 3) or increase the timeout if fetching fails.

Notes

The extension requires the history permission, which Chrome prompts users to approve.
For production, add authentication to FastAPI endpoints and deploy to a cloud provider.
Test with simple URLs (e.g., https://example.com) to avoid anti-bot measures on complex sites.

