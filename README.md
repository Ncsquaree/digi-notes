# Digi Notes - AI-Powered Knowledge Extraction System

Digi Notes transforms handwritten academic notes into structured, searchable digital study aids using OCR, LLMs, and knowledge graphs.

Key features
- Handwriting OCR (TrOCR)
- LLM-based semantic parsing and summary
- Knowledge Graph sync (AWS Neptune)
- Flashcard generation with spaced repetition (SM-2)

Tech stack
- Backend: Node.js, Express, PostgreSQL, Redis
- AI Service: Python, FastAPI, TrOCR, transformers, OpenAI
- Storage: AWS S3
- Deployment: Docker, Docker Compose

Architecture overview
- microservices: `backend` (API) and `ai-service` (ML/LLM)
- persistence: PostgreSQL for metadata, S3 for images, optional Neptune for KG
- cache: Redis for session/response caching

Prerequisites
- Docker & Docker Compose
- AWS account with S3 (and Neptune if using knowledge graph)
- OpenAI API key (or alternative LLM credentials)
- Node.js 18+ and Python 3.10+ for local development

## How to Run the Digi Notes Application Locally

The app consists of 4 main parts: **Frontend** (React/Vite), **Backend** (Node.js/Express), **AI Service** (Python/FastAPI), and **Databases** (Postgres + Redis). Use Docker Compose for backend/AI/DB (recommended) + separate frontend dev server.

### Prerequisites for Local Setup
- Docker & Docker Compose (for services)
- Node.js 18+ (for frontend/backend dev)
- AWS Account: S3 bucket + IAM keys (for uploads/OCR fallback)
- OpenAI API Key (for LLM parsing/summaries/flashcards)
- Git/Bash/PowerShell on Windows

### Step 1: Environment Setup

Copy and edit `.env` files (fill secrets, never commit real values):

```bash
# Root
cp .env.example .env  # Common vars (DB/Redis/Neptune)

# Backend
cp backend/.env.example backend/.env  # DB, JWT_SECRET, AWS keys, AI_SERVICE_URL=http://localhost:8000

# AI
cp ai/.env.example ai/.env  # OpenAI key, AWS, TROCR_DEVICE=cpu (or cuda for GPU)

# Frontend (create if missing)
cat > frontend/.env << EOF
VITE_API_URL=http://localhost:5000
EOF
```

**Key Secrets to Set:**
- `POSTGRES_PASSWORD` (e.g., mypassword123)
- `JWT_SECRET` / `JWT_REFRESH_SECRET` (generate: `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_S3_BUCKET`
- `OPENAI_API_KEY=sk-...`

### Step 2: Database Setup

```bash
docker-compose up postgres redis  # Start DBs first (or all)
cd backend
npm install
npm run db:migrate  # Run migrations (node-pg-migrate)
```

### Step 3: Run Services

#### Option A: Docker (Recommended - Full Stack Backend/AI/DB)

```bash
docker-compose up --build -d  # Detached: postgres(5432), redis(6379), backend(5000), ai(8000)
```

**Check Health:**
- Backend: `curl http://localhost:5000/health`
- AI: `curl http://localhost:8000/health`
- Logs: `docker-compose logs -f`

#### Option B: Local Dev (No Docker for Services)

```bash
# Terminal 1: Backend
cd backend && npm run dev  # http://localhost:5000

# Terminal 2: AI
cd ai && pip install -r requirements.txt && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Ensure Postgres/Redis running locally
```

### Step 4: Run Frontend

```bash
cd frontend
npm install
npm run dev  # http://localhost:3000 (Vite dev server)
```

### Step 5: Test the App

1. Open `http://localhost:3000`
2. Signup/Login (JWT auto-handled)
3. Dashboard → Subjects → Add subjects/chapters/notes
4. Tools → Upload image → Process (OCR → AI → Flashcards/Quiz)

### Troubleshooting

| Issue | Fix |
|-------|-----|
| DB Connection Failed | Check `DB_HOST=localhost` (local) or `postgres` (docker), password matches |
| AWS/S3 Errors | Verify bucket exists, IAM policy allows `s3:PutObject`/`GetObject`, region matches |
| OpenAI Fails | Set valid `OPENAI_API_KEY`, check rate limits |
| CORS Errors | Update `CORS_ORIGIN` in `backend/.env` to include `http://localhost:3000` |
| AI Model Download Slow | First run downloads TrOCR (~1GB), use GPU for speed (`TROCR_DEVICE=cuda`) |
| Migrations Fail | Run `npm run db:migrate` after DB up |

**Stop all services:**
```bash
docker-compose down -v  # removes volumes/DB data
```

For additional docs: see `docs/setup_guide.md`, `docs/aws_setup.md`
- Backend API: `http://localhost:5000/api-docs`
- AI Service: `http://localhost:8000/docs`

Development
- Backend: `cd backend && npm install && npm run dev`
- AI Service: `cd ai && pip install -r requirements.txt && uvicorn main:app --reload`
- DB migrations: `npm run db:migrate` (after implementing migration scripts)

Docker services
- backend: `5000:5000`
- ai-service: `8000:8000`
- postgres: `5432:5432`
- redis: `6379:6379`

Database schema
- Main tables: `users`, `subjects`, `chapters`, `notes`, `flashcards`, `study_sessions`.
- `database/init.sql` contains full schema, indexes, triggers, and seed data.

## AWS Services

### S3 Storage
- Used for storing uploaded note images
- Backend generates presigned URLs for secure uploads
- AI service downloads images for processing
- See `docs/aws_setup.md` for setup instructions

### Neptune (Optional)
- Graph database for knowledge graph storage
- Can be enabled later for advanced features
- Not required for MVP

Testing
- Backend tests: `cd backend && npm test`
- AI tests: `cd ai && pytest`

Contributing
- Please follow project coding standards, open PRs against `main`, and include tests for new features.

License
- MIT
