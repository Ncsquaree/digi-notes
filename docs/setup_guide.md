# Digi Notes — Setup Guide

This guide covers local development, Docker-based setup, and production notes.

## Local Development

Prerequisites:
- Node.js 18+
- Python 3.10+ (for AI service)
- PostgreSQL (local or docker)
- Redis (optional for caching)
- AWS CLI credentials (for S3/Textract) if you plan to test S3/Textract locally

1. Clone the repo and install dependencies:

```powershell
git clone <repo-url>
cd "f:\Intership\NC SQUAREE ED TECH\digi notes\backend"
npm install
cd ../frontend
npm install
cd ../ai
pip install -r requirements.txt
```

2. Copy environment examples and update values:

```powershell
cp backend/.env.example backend/.env
cp ai/.env.example ai/.env
cp frontend/.env.example frontend/.env
```

Edit `.env` files to set DB credentials, S3 bucket, OpenAI key, and other service-specific variables.

3. Start services for local development:
- Start Postgres (via Docker or local)
- Start Redis (optional)
- Start backend:

```powershell
cd backend
npm run dev
```

- Start AI service (in a separate terminal):

```powershell
cd ai
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Start frontend:

```powershell
cd frontend
npm run dev
```

4. Troubleshooting
- If a service cannot connect to Postgres, verify `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`.
- For S3/Textract issues, ensure AWS credentials are configured in environment or via `~/.aws/credentials`.
- If OpenAI requests fail, confirm `OPENAI_API_KEY` and network connectivity.

---

## Docker

We provide images for backend and AI services. Typical docker-compose setup:

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: digi_notes
    volumes:
      - db_data:/var/lib/postgresql/data
  redis:
    image: redis:7
  backend:
    build: ./backend
    ports:
      - '5000:5000'
    environment:
      - DB_HOST=postgres
      - DB_USER=postgres
      - DB_PASSWORD=postgres
      - DB_NAME=digi_notes
    depends_on:
      - postgres
      - redis
  ai:
    build: ./ai
    ports:
      - '8000:8000'
    environment:
      - BACKEND_URL=http://backend:5000
    depends_on:
      - backend

volumes:
  db_data:
```

Run with:

```powershell
docker-compose up --build
```

## Production

- Use environment variable secrets via your orchestration (Kubernetes Secrets, AWS Parameter Store, etc.).
- Configure log aggregation (e.g., export logs to a central ELK/CloudWatch endpoint) and ensure `LOG_FILE_PATH` is writable.
- Ensure `JWT_REFRESH_SECRET` and `JWT_SECRET` are long, random strings (at least 32 chars for refresh secret recommended).
- Configure health checks and readiness probes for `backend:/health` and `ai:/ready`.

## Additional Notes

- To generate API docs for the backend, visit `/api-docs` after starting the backend.
- AI docs are available at the FastAPI automatic docs when running the AI service.

