# API Documentation Overview

This document provides a high-level overview of the Digi Notes APIs and where to find the interactive docs.

## Backend API
- Base path: `/api`
- Authentication: JWT bearer tokens on `Authorization: Bearer <token>`.
- Live Swagger UI: `/api-docs` (served by the backend when running). A JSON spec is available at `/api-docs.json`.

### Core resources
- Auth: `/api/auth/*` — register, login, refresh token, logout, current user.
- Notes: `/api/notes/*` — CRUD for notes and AI processing triggers (`/api/notes/{id}/process` and `/api/notes/{id}/status`).
- Subjects & Chapters: `/api/subjects/*`, `/api/chapters/*` — organize notes.
- Flashcards: `/api/flashcards/*` — create, review and get statistics.
- Dashboard: `/api/dashboard` — aggregated metrics.
- S3 helpers: `/api/s3/presign-upload` — get presigned URL for uploads.
- AI proxies: `/api/ai/*` — endpoints that proxy to the AI microservice (OCR, parsing, summarization, tools).

## AI Service
- FastAPI provides automatic interactive docs at `/docs` and `/redoc` when running the AI service.
- Important endpoints:
  - `/health` and `/ready` — health and readiness checks.
  - `/process-note` — full processing pipeline (OCR, parse, summarize, graph, flashcards).
  - `/process-note/status/{task_id}` — polling endpoint for background tasks.
  - `/ocr/extract`, `/parse/semantic`, `/summarize`, `/flashcards/generate`, `/tools/*` — modular endpoints used by the backend proxies.

## Observability & Tracing
- Use `X-Request-ID` to trace a request through backend -> AI -> background workers.
- Backend will forward `X-Request-ID` to the AI service; AI will echo it on responses.

## Notes
- Some AI endpoints are still implemented in the AI microservice; backend proxies exist to forward requests and aggregate results.
- For more details and schema examples, open the backend `/api-docs` and AI `/docs` while services are running.

