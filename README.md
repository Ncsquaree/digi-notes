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

Quick start
1. Clone repository
2. Copy `.env.example` to `.env` and update values
3. Copy `backend/.env.example` to `backend/.env` and update values
4. Copy `ai/.env.example` to `ai/.env` and update values

S3 Setup

4. Set up AWS S3 bucket and IAM user (see `docs/aws_setup.md` for detailed instructions):
	- Create S3 bucket (e.g., `digi-notes-uploads-dev`)
	- Create IAM user with S3 access policy
	- Add AWS credentials to `.env` files

5. Run:
```
docker-compose up --build
```
6. Access backend: `http://localhost:5000`
7. Access AI service docs: `http://localhost:8000/docs`

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
