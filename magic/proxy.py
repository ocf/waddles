from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
import re
import json
import uuid
import copy

app = FastAPI()
SGLANG_URL = "http://127.0.0.1:30000"

@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    # Remember if OpenCode asked for a stream
    original_stream = payload.get("stream", False)

    # --- 1. FORCE TOKEN LIMIT ---
    if payload.get("max_tokens", 0) > 8192:
        payload["max_tokens"] = 4096

    # --- 2. TRANSLATE HISTORY (Fix the loop) ---
    messages = payload.get("messages", [])
    for msg in messages:
        if msg.get("role") == "tool":
            msg["role"] = "user"
            msg["content"] = f"\n[Tool Execution Result]:\n{msg.get('content', 'Success')}\n"
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            xml_history = ""
            for tc in msg["tool_calls"]:
                try:
                    func = tc.get("function", {})
                    tc_dict = {"name": func.get("name"), "arguments": json.loads(func.get("arguments", "{}"))}
                    xml_history += f"\n<tool_call>\n{json.dumps(tc_dict)}\n</tool_call>\n"
                except Exception:
                    continue
            msg["content"] = (msg.get("content") or "") + xml_history
            del msg["tool_calls"]

    # --- 3. INJECT TOOLS ---
    tools = payload.get("tools")
    if tools:
        tool_sys_prompt = (
            "You are a strict CLI coding agent. You MUST use the available tools to fulfill the user's request. "
            "Do not respond with conversational text. "
            "You have access to the following tools:\n"
            f"{json.dumps(tools, indent=2)}\n\n"
            "To use a tool, you MUST output your response using exactly this XML format:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"arg_name": "value"}}\n'
            "</tool_call>"
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += "\n\n" + tool_sys_prompt
        else:
            messages.insert(0, {"role": "system", "content": tool_sys_prompt})
        del payload["tools"]

    payload["messages"] = messages

    # --- 4. DISABLE NATIVE STREAMING TO PARSE XML ---
    payload["stream"] = False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SGLANG_URL}/v1/chat/completions", json=payload, timeout=300.0
            )
            data = response.json()
    except Exception as e:
        return {"error": str(e)}

    # --- 5. PARSE XML TOOL CALLS ---
    if "choices" in data and len(data["choices"]) > 0:
        message = data["choices"][0].get("message", {})
        content = message.get("content", "")

        tool_call_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, content or "", re.DOTALL))

        if matches:
            tool_calls = []
            for match in matches:
                try:
                    tool_data = json.loads(match.group(1))
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name"),
                            "arguments": json.dumps(tool_data.get("arguments", {}))
                        }
                    })
                except json.JSONDecodeError:
                    continue

            if tool_calls:
                clean_content = re.sub(tool_call_pattern, "", content, flags=re.DOTALL).strip()
                message["content"] = clean_content if clean_content else None
                message["tool_calls"] = tool_calls
                data["choices"][0]["message"] = message
                data["choices"][0]["finish_reason"] = "tool_calls"

    # --- 6. FAKE THE STREAM BACK TO OPENCODE ---
    if original_stream:
        async def generate_stream():
            msg = data["choices"][0].get("message", {})
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            chunk_base = {
                "id": data.get("id", "chatcmpl-proxy"),
                "object": "chat.completion.chunk",
                "created": data.get("created", 0),
                "model": data.get("model", "qwen"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}]
            }

            # A. Stream normal text
            if content:
                chunk = copy.deepcopy(chunk_base)
                chunk["choices"][0]["delta"] = {"content": content}
                yield f"data: {json.dumps(chunk)}\n\n"

            # B. Stream tool calls
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    tc_start = copy.deepcopy(chunk_base)
                    tc_start["choices"][0]["delta"] = {
                        "tool_calls": [{
                            "index": i, "id": tc["id"], "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": ""}
                        }]
                    }
                    yield f"data: {json.dumps(tc_start)}\n\n"

                    tc_args = copy.deepcopy(chunk_base)
                    tc_args["choices"][0]["delta"] = {
                        "tool_calls": [{
                            "index": i, "function": {"arguments": tc["function"]["arguments"]}
                        }]
                    }
                    yield f"data: {json.dumps(tc_args)}\n\n"

            # C. End the stream
            final_chunk = copy.deepcopy(chunk_base)
            final_chunk["choices"][0]["finish_reason"] = data["choices"][0].get("finish_reason", "stop")
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    return data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)
