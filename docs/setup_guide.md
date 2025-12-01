# Setup Guide (local development)

1. Create a Python virtualenv and install AI dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai/requirements.txt
```

2. Start AI service

```powershell
cd ai
python main.py
```

3. Start backend (Node.js)

```powershell
cd backend
npm install
npm run dev
```

4. Start frontend via React Native tools from `frontend/`.

Notes:
- Fill `.env` with DATABASE_URL and AI_SERVICE_URL.
- Use Docker for production; see `infra/` for compose file and Dockerfiles.

