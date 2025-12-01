"""FastAPI entrypoint for AI services (OCR, semantic parsing, graph builder).
Endpoints:
- POST /process-note
- GET /health
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from ai.modules.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="DigiNotes AI Service")


class ProcessRequest(BaseModel):
	s3Url: str
	userId: int = None


@app.post('/process-note')
async def process_note(req: ProcessRequest):
	# Placeholder: orchestration logic goes here
	logger.info(f"Received process request for {req.s3Url}")
	# Normally you'd call OCR -> preprocess -> llm_parser -> graph_builder -> summarizer
	return {"status": "queued", "s3Url": req.s3Url}


@app.get('/health')
async def health():
	return {"ok": True}


if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port=8000)

