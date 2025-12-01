# System Architecture — Digi Notes

This document describes the high-level design mapping to the repository layout.

Services
- AI (Python / FastAPI): `ai/` — handles OCR (TrOCR/Textract), semantic parsing (LLM), summarization, knowledge graph building (Neptune), and flashcard generation.
- Backend (Node.js): `backend/` — API gateway, user management, orchestrates calls to AI and persists metadata to PostgreSQL.
- Frontend (React Native): `frontend/` — mobile UI for uploading notes, viewing summaries, and studying flashcards.

Data Flow
1. Mobile uploads image → Backend receives and stores metadata and S3 presigned URL.
2. Backend calls AI service `/process-note` with S3 URL.
3. AI service runs OCR → LLM parsing → graph building → summarization → flashcard generation and stores outputs (S3 / Neptune / embeddings).
4. Backend stores metadata (Postgres) and exposes endpoints for frontend.

Deployment
- Each service runs in Docker. AI requires GPU support optionally.
- Neptune is hosted on AWS. S3 for media. RDS/Postgres for metadata. Cognito for auth.

