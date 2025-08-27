# Biomni API Server

A FastAPI-based server that provides OpenAI Chat Completions API compatible interface, allowing you to use Biomni biomedical AI agent through standard OpenAI API format.

## Quick Start

### 1. Install and Configure


```bash
cd server
pip install -r requirements.txt
cp .app.env.example .app.env
# Edit .app.env to configure API keys etc.
```

### 2. Start Server

```bash
python app.py
# Server runs on http://localhost:8000
```

## Usage Examples

### Basic Call

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "biomni-agent",
    "messages": [{"role": "user", "content": "Analyze this protein sequence: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"}],
    "stream": False
})

print(response.json()["choices"][0]["message"]["content"])
```

### Using OpenAI Client Library

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy-key")
response = client.chat.completions.create(
    model="biomni-agent",
    messages=[{"role": "user", "content": "Design a CRISPR screening experiment"}]
)
```

## Core Features

- ✅ **OpenAI API Compatible** - Supports all mainstream OpenAI client libraries
- ✅ **Streaming Response** - Real-time Biomni agent execution results

## Configuration

Edit `.app.env` file:

```env
DATA_PATH=./data
LLM=gpt-4
OPENAI_BASE_URL=http://your-llm-server.com/v1
OPENAI_API_KEY=sk-your-api-key
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive documentation.

## Deployment

```bash
# Production environment
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
