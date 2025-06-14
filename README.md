# KnowledgeGraph - AI-Powered Knowledge Management System

AI-powered intelligent categorization and filtering capabilities.

## Quick Setup (Windows)

### Environment Setup
```cmd
# Clone repository
git clone https://github.com/Sa1f27/KnowledgeGraph.git
cd KnowledgeGraph

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

2. **Install Chrome Extension:**
   - Open Chrome → Extensions → Developer Mode ON
   - Click "Load unpacked" → Select `extension-exporter` folder
   - Extension icon appears in toolbar

3. **Access Dashboard:**
   - Open: http://localhost:8000/static/index.html

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
