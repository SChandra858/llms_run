import uvicorn

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.playground import Playground
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

reasoning_agent = Agent(
    name="Reasoning Agent",
    model=Ollama(id="qwen3:1.7b"),
    markdown=True,
    debug_mode=True,
    debug_level=2,
)

# UI for Reasoning agent
app = Playground(agents=[reasoning_agent]).get_app()


class ChatRequest(BaseModel):
    message: str


@app.get("/", include_in_schema=False)
def chat_page() -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Reasoning Agent</title>
<style>body{max-width:800px;margin:40px auto;padding:0 20px;font:16px system-ui;background:#101114;color:#eee}textarea,button{font:inherit}textarea{width:100%;box-sizing:border-box;padding:12px;border-radius:8px;min-height:120px}button{margin-top:10px;padding:10px 16px;border:0;border-radius:8px;background:#4f46e5;color:#fff;cursor:pointer}pre{white-space:pre-wrap;background:#1b1d23;padding:16px;border-radius:8px;line-height:1.5}</style>
</head><body><h1>Local Reasoning Agent</h1><p>Powered locally by Qwen3 1.7B.</p>
<textarea id="message" placeholder="Ask a question..."></textarea><br><button id="send">Send</button><pre id="answer">Your response will appear here.</pre>
<script>const message=document.getElementById('message'),answer=document.getElementById('answer'),send=document.getElementById('send');send.onclick=async()=>{const text=message.value.trim();if(!text)return;send.disabled=true;answer.textContent='Thinking…';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});const data=await r.json();if(!r.ok)throw new Error(data.detail||'Request failed');answer.textContent=data.response||'(No response)';}catch(e){answer.textContent='Error: '+e.message;}finally{send.disabled=false;}};</script>
</body></html>"""
    )


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    response = reasoning_agent.run(request.message)
    return {"response": str(response.content)}

# Run the Playground app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7777)
