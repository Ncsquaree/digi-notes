# Logging Guide

This document explains logging, request tracing, rotation, and archival for both backend and AI services.

## Architecture
- Both services use structured JSON-like logging.
- Backend uses `winston` with optional daily rotate file transport; AI service uses a Python logger configured via `modules.utils.get_logger()`.
- All requests are assigned an `X-Request-ID` for end-to-end tracing. The backend middleware prefers incoming `X-Request-ID` and exposes it on responses; the AI service middleware respects an incoming `X-Request-ID` and echoes it back.

## Request Tracing
- Include `X-Request-ID` in incoming requests for correlated logs across services.
- The backend generates `req.id` and sets `X-Request-ID` on responses if none provided.
- In logs, `requestId` or `request_id` fields are included in each request lifecycle log entry.

## Log Rotation
- Backend uses `winston-daily-rotate-file` with sensible defaults (rotate daily, keep 14 days). Configure via `LOG_FILE_PATH`, `LOG_MAX_SIZE`, and `LOG_MAX_FILES` env vars.
- AI service rotates logs via your Python logging handler configuration; use `RotatingFileHandler` or a log shipper.

## Archival and Retention
- Retain logs locally for short term (7-14 days) and push to central storage for long-term archival.
- Use compressed archives (gz) and store in object storage (S3) for compliance.

## Troubleshooting
- If logs fail to write, ensure `LOG_FILE_PATH` directory exists and service user has write permissions.
- For missing `X-Request-ID` traces across services, ensure frontend sets header or backend generates and forwards it to AI service.

