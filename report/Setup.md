# Setup Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

## 1. Install Ollama Models

```bash
ollama pull llama3.2:3b
ollama pull granite3.1-dense:2b
```

Verify Ollama is running:
```bash
ollama list
```

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## 4. Environment Setup

```bash
cp .env.example .env
```

Edit `.env` if you want to use Groq (cloud model):
- Get a free API key at https://console.groq.com
- Set `GROQ_API_KEY=your_key`

For local-only usage (ollama models), no `.env` changes are needed.

## 5. Build the RAG Index (first time only)

```bash
python -m src.rag.ingest
```

Processes the recipe dataset, builds the ChromaDB vector store and BM25 index. Takes 2-3 minutes.

## 6. Build the Knowledge Graph (first time only)

```bash
python -m src.rag.graph_builder
```

Builds the NetworkX ingredient/tag graph for re-ranking. Takes ~30 seconds.

## 7. Launch the App

You need two terminals:

**Terminal 1 - Backend (FastAPI):**
```bash
uvicorn server:app --reload --port 8001
```

**Terminal 2 - Frontend (Next.js):**
```bash
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## Usage

1. Select a model from the dropdown (start with Llama 3.2 3B for fastest results)
2. Pick a sample request or type your own in natural language:
   - "Plan me 3 high-protein dinners, no peanuts, under 30 minutes"
   - "I want 5 vegetarian meals for the week, avoid gluten and dairy"
   - "Quick keto lunches for 4 days, nothing with shellfish"
3. Or click "Voice Input" to speak your request
4. Click **Generate Meal Plan**
5. Browse results across tabs: Meal Plan, Grocery & Budget, Nutrition, Schedule, Agent Trace, Memory

## Troubleshooting

**Ollama not responding:** Make sure the Ollama app/service is running. On macOS it runs as a menu bar app.

**Slow first run:** The first request loads embedding models into memory (~10s). Subsequent requests are faster.

**Groq rate limit:** The free tier has daily token limits. If you hit them, switch to an ollama model.

**CORS errors:** Make sure the backend is running on port 8001 and the frontend on port 3000.
