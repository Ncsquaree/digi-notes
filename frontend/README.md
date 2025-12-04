# Frontend (Notexa) — JWT Auth Migration Notes

This project uses a backend JWT-based authentication flow. The following notes document the changes and how to set up your local environment.

Quick setup
- Copy `frontend/.env.example` to `.env` and set `VITE_API_URL` to your backend API (e.g. `http://localhost:5000/api`).
- Install dependencies: `npm install` (or `yarn` / `pnpm` depending on your workflow).

Auth behavior
- All API calls should use the shared `apiClient` at `src/api/client.ts`.
- `apiClient` automatically attaches the `Authorization: Bearer <accessToken>` header from `localStorage`.
- On a 401 response, `apiClient` will attempt a single refresh request to `/auth/refresh-token` using the `refreshToken` from `localStorage` and retry the failed request.
- Tokens (`accessToken` and `refreshToken`) are stored in `localStorage` by `AuthContext` after successful login/signup.

AuthContext
- `src/context/AuthContext.tsx` exposes `useAuth()` which returns `{ user, loading, login, signup, logout }`.
- `login(email, password)` and `signup(email, password, name?)` call the backend endpoints and persist tokens.
- `logout()` calls the backend `/auth/logout` endpoint (best-effort), clears tokens, and redirects to `/login`.

Notes
- Keep `.env` out of version control (already added to `.gitignore`).
- Firebase code remains in the repo for now, but authentication has been migrated to the backend JWT flow; Firebase files are planned to be deprecated in a follow-up.

Pages and API integration
- **Dashboard**: `src/pages/Dashboard.tsx` fetches `/api/dashboard` to show aggregated stats (`total_subjects`, `total_notes`, `total_flashcards`, `due_flashcards`) and a list of recent subjects.
- **Library**: `src/pages/Library.tsx` fetches `/api/notes`, `/api/flashcards`, and `/api/study-sessions` to show searchable lists. It implements a direct S3 upload flow using `POST /api/s3/presign-upload`, then a `PUT` to the presigned URL, and finally `POST /api/notes` to create a note referencing the uploaded file.
- **Templates**: `src/pages/Templates.tsx` keeps in-repo templates and routes users to `/tools` with a template in `location.state` when they click "Use this template".

API patterns
- Use `src/api/client.ts` for all requests — it injects JWTs and handles refresh on 401.
- Show loading and error states in pages; follow `Subjects.tsx` and `Dashboard.tsx` for examples.

Upload flow (summary)
1. `POST /api/s3/presign-upload` with `{ filename, contentType }`.
2. `PUT` the file to the returned `uploadUrl` (S3) with correct `Content-Type`.
3. `POST /api/notes` with `{ title, originalImageUrl: key }` to create a note.

Tools page
- **Tools**: `src/pages/Tools.tsx` supports two input modes: `text` and `image`.
	- **Text mode**: sends text to `POST /ai/parse/semantic` then calls one of the tool endpoints depending on selected mode:
		- `POST /ai/tools/generate-quiz` → returns `{ quiz }`
		- `POST /ai/flashcards/generate` → returns `{ flashcards }`
		- `POST /ai/tools/generate-mindmap` → returns `{ mindmap }`
	- **Image mode**: uploads file via presign → S3 PUT → `POST /notes` to create note → `POST /notes/:id/process` to start async processing → polls `GET /notes/:id/processing-status` every 2s to receive progress and parsed content.
	- Generated results are displayed in preview and flashcards can be saved with a bulk save call to `/flashcards`.

Environment variables for Tools
- `VITE_API_URL` — backend API base (e.g. `http://localhost:5000/api`).
- `VITE_POLLING_INTERVAL_MS` — polling interval in milliseconds for note processing status (default `2000`).
- `VITE_MAX_UPLOAD_SIZE_MB` — optional documentation for upload size limits enforced by backend (default `10`).

