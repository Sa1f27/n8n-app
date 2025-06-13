# MindCanvas - AI-Powered Knowledge Management System

AI-powered browser history processing with intelligent categorization and filtering capabilities.

## Quick Setup (Windows)

### Prerequisites
```cmd
# Install Python 3.9+
choco install python

# Install Node.js (for development tools)
choco install nodejs

# Verify installations
python --version
pip --version
```

### Environment Setup
```cmd
# Clone repository
git clone https://github.com/Sa1f27/MindCanvas.git
cd MindCanvas

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
cd history-backend
pip install -r requirements.txt
```

### API Keys Setup
Create `.env` file in `history-backend` directory or store in system's environmental variables:
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Get API Keys:**
- Groq API (Primary): https://console.groq.com/ (Free tier available)
- OpenAI API (Fallback): https://platform.openai.com/ (Optional)

### Running the Application

1. **Start Backend:**
```cmd
cd history-backend
python main.py
```
Backend runs on: http://localhost:8000

2. **Install Chrome Extension:**
   - Open Chrome → Extensions → Developer Mode ON
   - Click "Load unpacked" → Select `extension-exporter` folder
   - Extension icon appears in toolbar

3. **Access Dashboard:**
   - Open: http://localhost:8000/static/index.html

## Usage Flow
1. Browse websites normally
2. Click MindCanvas extension → Export History (24h)
3. View processed content in Dashboard
4. Use filters and search to explore your knowledge

## Tech Stack
- **Backend:** FastAPI + SQLite + Groq Llama + OpenAI
- **Content Extraction:** Hybrid (BeautifulSoup + Crawl4AI)
- **Caching:** Multi-level (Memory + Database)
- **Frontend:** Vanilla JS + Modern CSS

## Troubleshooting
- **Backend not starting:** Check Python version and dependencies
- **Extension not working:** Ensure backend is running on port 8000
- **No content processed:** Verify API keys in `.env` file
- **Database issues:** Delete `mindcanvas.db` and restart

## Data Management
```cmd
# Clear all data from DB
python cleanup.py
```

## Testing API endpoints and functionalities
```cmd
# Unit tests
python test.py
```

## Project Structure
```
mindcanvas/
├── extension-exporter/  # Chrome extension
├── history-backend/     # FastAPI backend
└── static/             # Dashboard frontend
```
