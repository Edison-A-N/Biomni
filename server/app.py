import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from biomni.agent import A1
except ImportError as e:
    print(f"Warning: Could not import A1 from biomni.agent: {e}")
    A1 = None


class BiomniConfig(BaseSettings):
    data_path: str = Field(default="./data", description="Path to data directory", alias="DATA_PATH")
    mcp_path: str = Field(default="mcp_config.yaml", description="Path to MCP directory", alias="MCP_PATH")
    llm: str = Field(default="gpt-4", description="LLM model to use", alias="LLM")
    base_url: str = Field(default=None, description="Base URL for API", alias="OPENAI_BASE_URL")
    api_key: str = Field(default=None, description="API key for authentication", alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(env_file=".app.env", env_file_encoding="utf-8", extra="ignore")


config = BiomniConfig()

app = FastAPI(title="Biomni API", description="API for running Biomni biomedical AI agent tasks", version="1.0.0")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    stream: bool | None = False
    model: str | None = "biomni-agent"


async def generate_stream_response(task: str):
    try:
        if A1 is None:
            yield f"data: {json.dumps({'error': 'Biomni agent not available'})}\n\n"
            return

        agent = A1(path=config.data_path, llm=config.llm, base_url=config.base_url, api_key=config.api_key)
        agent.add_mcp(config.mcp_path)

        # Directly iterate over the generator from agent.go()
        for output in agent.go_stream(task):
            output = output["output"]
            output += "\n"

            chunk_size = 100
            for i in range(0, len(output), chunk_size):
                chunk = output[i : i + chunk_size]
                response_data = {
                    "id": "biomni_response",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "biomni-agent",
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(response_data)}\n\n"

        final_response = {
            "id": "biomni_response",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "biomni-agent",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Error: {e}")
        yield f"data: {json.dumps({'error': 'Error executing task'})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    """
    try:
        user_messages = [msg.content for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        task = user_messages[-1]

        if request.stream:
            return StreamingResponse(generate_stream_response(task), media_type="text/event-stream")
        else:
            if A1 is None:
                raise HTTPException(status_code=500, detail="Biomni agent not available")

            agent = A1(path=config.data_path, llm=config.llm, base_url=config.base_url, api_key=config.api_key)
            agent.add_mcp(config.mcp_path)

            all_outputs = []

            # Collect all outputs from the generator
            for output in agent.go(task):
                all_outputs.append(output)

            final_content = "\n".join(all_outputs) if all_outputs else "No output generated"

            return {
                "id": "biomni_response",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "biomni-agent",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": len(task),
                    "completion_tokens": len(final_content),
                    "total_tokens": len(task) + len(final_content),
                },
            }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error executing task") from e


if __name__ == "__main__":
    import uvicorn

    print("Starting Biomni API server...")
    print("API endpoint: POST /v1/chat/completions")

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
