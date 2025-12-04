#!/bin/sh
set -e

echo "Running AI service environment validation..."
python scripts/validate_env.py

echo "Validation succeeded. Starting Uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
