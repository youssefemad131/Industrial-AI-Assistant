"""FastAPI backend.  Run from the repo root:  uvicorn backend.main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel
from .engine import FactoryLens, ERROR_MSG

app = FastAPI(title="FactoryLens AI", description="Synthetic data only")
bot = FactoryLens()


class Ask(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "data": "synthetic"}


@app.post("/chat")
def chat(body: Ask):
    try:
        return bot.ask(body.question, body.session_id)
    except Exception:  # E5: plain message, never a stack trace
        return {"type": "error", "reply": ERROR_MSG["en"], "quick_replies": ["Retry", "Talk to a person"]}
