# Backend (digi-notes)

This backend is a Node.js + Express application using PostgreSQL. The codebase intentionally uses lightweight repository-style models built on `pg` rather than a full ORM (Sequelize/Knex). The repository classes live in `src/models` and services with business logic live in `src/services`.

Key decisions
- Database access: `pg` + simple repository classes. This avoids ORM abstractions and keeps SQL visible and explicit. If you want to migrate to Sequelize/Knex later, implement a compatibility layer that preserves existing method signatures in `src/models/*`.

Getting started

1. Install dependencies

```powershell
cd backend
npm install
```

2. Copy environment example and set secrets

```powershell
copy .env.example .env
# Edit backend/.env and set DB_PASSWORD, JWT secrets, and DATABASE_URL if desired
```

3. Run database migrations

This project uses `node-pg-migrate`. The migration configuration is in `backend/.migrate` and migrations live in `backend/migrations`.

Make sure `DATABASE_URL` is set (it can be constructed from `DB_*` vars or provided directly). Example:

```powershell
$env:DATABASE_URL = 'postgresql://postgres:your_secure_password_here@postgres:5432/digi_notes'
npm run db:migrate
```

To roll back the most recent migration:

```powershell
npm run db:migrate:down
```

Project structure
- `src/models` — lightweight repository classes using `pg`. Each file exports parameterized methods (findById, create, update, delete, countByUserId, etc.).
- `src/services` — business logic layer. Services orchestrate models, enforce authorization, and encapsulate transactions.
	- **S3Service** — AWS S3 operations (presigned URLs, file validation, uploads/downloads)
- `src/middleware` — Express middleware (validation, error handling, etc.).
- `migrations` — `node-pg-migrate` migration scripts.

Why `pg` instead of an ORM?
- Simple, explicit SQL is used to keep queries readable and tuned for this project's needs.
- Lower abstraction overhead and clearer performance characteristics.
- Easier to reason about migrations and exact SQL behavior.

If you'd prefer ORM usage, we can migrate incrementally by implementing model wrappers that preserve current method names.

Running the server

```powershell
npm run dev
```

Notes
- Ensure `DATABASE_URL` is set when running `npm run db:migrate` as `node-pg-migrate` reads it by default from the environment.
- Models include safeguards (whitelists) to prevent arbitrary column updates.

## AWS S3 Integration

The backend uses AWS SDK v3 for S3 operations via `S3Service`:

- **Presigned URLs**: Frontend uploads files directly to S3 using presigned PUT URLs
- **File Validation**: Size limits (10MB default), MIME types (jpeg, png, pdf)
- **Key Generation**: User-namespaced keys (`notes/{userId}/{timestamp}-{uuid}-{filename}`)
- **Security**: AWS credentials never exposed to frontend

See `docs/aws_setup.md` for S3 bucket and IAM configuration.

### Environment Variables

Required AWS variables in `backend/.env`:
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET=your-bucket-name
AWS_S3_PRESIGNED_URL_EXPIRY=3600
MAX_FILE_SIZE_MB=10
```

## Flashcards & AI Proxy

This backend exposes flashcard CRUD and review endpoints and proxies AI requests to the AI microservice.

### Flashcard Routes

- `GET /api/flashcards` - list user's flashcards (query: `noteId`, `page`, `limit`)
- `GET /api/flashcards/due` - list due flashcards
- `GET /api/flashcards/:id` - get a single flashcard
- `POST /api/flashcards` - create a flashcard
- `PUT /api/flashcards/:id` - update a flashcard
- `DELETE /api/flashcards/:id` - delete a flashcard
- `POST /api/flashcards/:id/review` - review a flashcard (SM-2 algorithm)
- `GET /api/flashcards/stats` - aggregated flashcard/study stats

### AI Proxy Routes

- `POST /api/ai/flashcards/generate` - Proxy to AI service `/flashcards/generate`. Body: `{ content: ParsedContent, count?: number, noteId?: uuid }`. If `noteId` is provided, generated flashcards are persisted and returned with `persisted: true`.

### Env vars

- `AI_SERVICE_URL` - URL for AI microservice (default in example: `http://ai:8000`).
- `AI_SERVICE_TIMEOUT` - Timeout for AI proxy requests in milliseconds (default `60000`).

## Testing

Run tests with the included Jest configuration and helpers.

Running Tests:

```bash
# Run all tests
npm test

# Run with coverage report
npm run test:coverage

# Run only unit tests
npm run test:unit

# Run only integration tests
npm run test:integration

# Watch mode for development
npm run test:watch
```

Test Structure:
- `tests/unit/controllers/` - Controller unit tests with mocked services
- `tests/unit/middleware/` - Middleware unit tests
- `tests/unit/utils/` - Utility function tests
- `tests/integration/` - End-to-end API tests with Supertest
- `tests/helpers/` - Shared mocks and test utilities

Coverage Requirements:
- Target: >80% coverage for controllers, services, middleware, utils
- Run `npm run test:coverage` to generate HTML report in `coverage/`

Writing Tests:
- Use `jest.mock()` for service/model mocking
- Use `supertest` for integration tests
- Import test helpers from `tests/helpers/`
- Follow existing test patterns in `tests/unit/controllers/`

Test Environment:
- Tests use `.env.test` for configuration
- Database/Redis/AI service calls are mocked
- JWT tokens use test secrets (safe for CI/CD)

## New API Endpoints (added)

This release adds several endpoints used by the frontend dashboard and library pages.

- `GET /api/dashboard` — returns aggregated workspace stats for the current user. Response example:

```
{
	"success": true,
	"stats": {
		"total_subjects": 5,
		"total_notes": 42,
		"total_flashcards": 120,
		"due_flashcards": 8,
		"recent_subjects": [ { "id": "...", "name": "Physics" }, ... ]
	}
}
```

- `POST /api/s3/presign-upload` — body `{ filename, contentType }`. Returns `{ uploadUrl, key, expiresIn }` for direct S3 PUT uploads. Usage pattern:
	1. Frontend requests presigned URL from backend.
	2. Frontend performs `PUT` to the `uploadUrl` (S3) with file bytes and the specified `Content-Type`.
	3. Frontend calls `POST /api/notes` with `originalImageUrl: key` to create a note referencing the uploaded file.

- `GET /api/study-sessions` — list user's study sessions. Supports query params `page`, `limit`, `startDate`, `endDate`.

See updated route files in `src/routes` for parameter validation.

Changelog:
- Fixed missing `FlashcardService.getStudyStats()` which now delegates to `StudySession.getStatsByUserId()` and returns normalized stats for the `/api/flashcards/stats` controller.

