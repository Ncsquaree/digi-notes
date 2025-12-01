# API Spec (Minimal)

POST /api/ai/process-note
- Request: { s3Url: string, userId?: number }
- Response: { status: 'queued' | 'processing' | 'done', taskId?: string }

GET /api/ai/health
- Response: { ok: true }

POST /api/notes
- Create note metadata

GET /api/notes/:id
- Get note metadata

GET /api/notes
- List recent notes

