# Biomni API Changes Documentation

## Breaking Changes

### A1 Agent Interface Adjustment

**Change Content**: The `A1.go()` method now has an optional `sender` parameter

**Original Interface**:
```python
def go(self, prompt):
    """Execute the agent with the given prompt."""
```

**New Interface**:
```python
def go(self, prompt, sender: Callable[[str], None] | None = None):
    """Execute the agent with the given prompt.
    
    Args:
        prompt: The user's query
        sender: Optional function to send intermediate outputs (str) -> None
    """
```

**Impact**: 
- Backward compatible, existing code requires no modification
- Added streaming output support, allowing real-time access to agent execution process

**Usage Examples**:
```python
# Original usage remains unchanged
agent = A1()
result = agent.go("Analyze gene sequence")

# New streaming output usage
def output_handler(output: str):
    print(f"Real-time output: {output}")

result = agent.go("Analyze gene sequence", sender=output_handler)
```

## New Features

### OpenAI Compatible API Server

Added `app.py` file, providing a fully OpenAI API format compatible HTTP service.

**Features**:
- Supports streaming and non-streaming responses
- Compatible with OpenAI API format
- Environment variable configuration
- Automatic API documentation generation

**API Endpoint**: `POST /v1/chat/completions`

**Request Format**:
```json
{
  "messages": [
    {"role": "user", "content": "Analyze this gene sequence"}
  ],
  "stream": false
}
```

**Response Format**:
```json
{
  "id": "biomni_response",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "biomni-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Analysis results..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Running Instructions

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic-settings
```

### 2. Configure Environment
```bash
cp .app.env.example .app.env
# Edit the .app.env file and enter your configuration
```

### 3. Start Server
```bash
python app.py
```

The server will start at `http://localhost:8000`

### 4. Test API

**Non-streaming Request**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Analyze this gene sequence: ATGCGATCG"}
    ],
    "stream": false
  }'
```

**Streaming Request**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Analyze this gene sequence: ATGCGATCG"}
    ],
    "stream": true
  }'
```

### 5. View API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration Details

`.app.env` file configuration items:
- `DATA_PATH`: Data directory path (default: `./data`)
- `LLM`: LLM model to use (default: `gpt-4`)
- `OPENAI_BASE_URL`: OpenAI API base URL (optional)
- `OPENAI_API_KEY`: OpenAI API key (optional)

## Important Notes

1. **Breaking Changes**: If your code directly calls the `A1.go()` method, existing code requires no modification, but you can add streaming output functionality
2. **Environment Configuration**: Ensure the `.app.env` file is correctly configured
3. **Data Path**: Ensure `DATA_PATH` points to the correct data directory 