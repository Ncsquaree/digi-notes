# Digi Notes AI Service

This service provides OCR, semantic parsing, knowledge graph building and flashcard generation for Digi Notes.

## Overview

- FastAPI application serving endpoints for OCR, parsing, and pipeline orchestration.
- Modules:
  - `modules/ocr/`
  - `modules/semantic/`
  - `modules/knowledge_graph/`
  - `modules/flashcards/`
  - `modules/utils/` (logging, file handling)

## Setup

1. Install dependencies

```powershell
cd ai
pip install -r requirements.txt
```

2. Copy environment

```powershell
copy .env.example .env
# Edit ai/.env and set AWS/OpenAI/DB creds
```

3. Validate environment

```powershell
python scripts/validate_env.py
```

## Running

Development:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Production:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Endpoints

- `GET /health` - liveness
- `GET /ready` - readiness checks for DB, Redis, S3, OpenAI
 - `GET /ready` - readiness checks for DB, Redis, S3, OpenAI, and optionally Neptune (set `NEPTUNE_REQUIRED_FOR_READY`)
- `POST /ocr/extract` - OCR stub
- `POST /process-note` - pipeline stub
- `POST /parse/semantic` - semantic parsing of OCR text into structured academic content
 - `POST /summarize` - generate brief and/or detailed summaries from `ParsedContent`

## OCR Module

### Overview

The OCR module extracts handwritten text from note images using a local ML
model (TrOCR) with optional preprocessing and AWS Textract as a cloud-based
fallback. The pipeline is designed for reliability and observability: model
loading is cached, preprocessing improves OCR accuracy on noisy scans, and
Textract can be used when model confidence is low.

### Components

- `TrOCRHandler`: singleton wrapper around Hugging Face's TrOCR model. Supports
  GPU/CPU selection and caches model artifacts between requests.
- `TextractHandler`: AWS Textract client with retry logic. Used as a fallback
  when local OCR confidence is below the configured threshold.
- `preprocess_image()`: denoise, threshold, and deskew pipeline implemented
  with OpenCV to improve OCR results on low-quality images.

### Endpoint: `POST /ocr/extract`

Request body (JSON):

- `s3_key` (string): S3 object key of the image to process.
- `s3_url` (string): Full S3 URL. Provide either `s3_key` or `s3_url`.
- `use_preprocessing` (bool): Apply preprocessing before OCR (default: true).
- `fallback_to_textract` (bool): Use Textract if TrOCR confidence < threshold.

Response (200):

- `success` (bool)
- `text` (string): extracted text
- `confidence` (float): confidence score (0-1)
- `method` (string): `trocr` or `textract`
- `preprocessing_applied` (bool)
- `metadata` (object): processing_time_ms, file_info, device, model_name, preprocessing_steps
- `request_id` (string)

Error codes:

- `400` - invalid input or file download/preprocessing errors
- `500` - model load or inference failures
- `502` - Textract fallback errors

Example curl:

```bash
curl -X POST http://localhost:8000/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{"s3_key":"path/to/image.jpg","use_preprocessing":true}'
```

### Configuration

Relevant env vars:

- `TROCR_MODEL` - HF model id (default `microsoft/trocr-large-handwritten`)
- `TROCR_DEVICE` - `cpu`, `cuda`, or `auto`
- `OCR_CONFIDENCE_THRESHOLD` - fallback threshold (0.0-1.0)
- `PREPROCESSING_ENABLED`, `PREPROCESSING_DENOISE`, `PREPROCESSING_THRESHOLD`, `PREPROCESSING_DESKEW`
- `AWS_TEXTRACT_ENABLED`, `AWS_TEXTRACT_MAX_PAGES`, `AWS_TEXTRACT_TIMEOUT`, `AWS_TEXTRACT_RETRY_ATTEMPTS`

## POST /parse/semantic

Performs semantic parsing of OCR-extracted text using an LLM to extract structured
academic content (topics, subtopics, formulas, concepts, questions).

Request body (JSON):

- `text` (string): OCR-extracted text to parse (max configured by `LLM_PARSER_MAX_TEXT_LENGTH`).
- `options` (object): optional parsing options.

Response (200):

- `success` (bool)
- `parsed_content` (object): follows the `ParsedContent` schema (topics, formulas, key_concepts, questions, metadata)
- `metadata` (object): processing_time_ms, text_length, model_used
- `request_id` (string)

Errors:

- `400` - empty text
- `422` - schema validation failed (LLM produced invalid JSON)
- `504` - LLM timeout
- `502` - LLM API error
- `500` - parsing failed

Example curl:

```bash
curl -X POST http://localhost:8000/parse/semantic \
  -H "Content-Type: application/json" \
  -d '{"text":"Photosynthesis is the process...\n\nFormula: 6CO2 + 6H2O -> C6H12O6 + 6O2"}'
```

Configuration:

- `OPENAI_API_KEY` - required for LLM calls
- `OPENAI_MODEL`, `OPENAI_MAX_TOKENS`, `OPENAI_TEMPERATURE`, `OPENAI_TIMEOUT`
- Retry/config: `OPENAI_RETRY_ATTEMPTS`, `OPENAI_RETRY_MAX_WAIT`, `OPENAI_RETRY_MULTIPLIER`
- Parser limits: `LLM_PARSER_MAX_TEXT_LENGTH`, `LLM_PARSER_ENABLE_COST_TRACKING`

Notes:

- The parser calls OpenAI and validates the JSON response against a Pydantic
  schema. The service uses `tenacity` for retries and logs LLM call metrics
  (tokens, duration, estimated cost) when `LLM_PARSER_ENABLE_COST_TRACKING` is enabled.

## POST /summarize

Generate brief and/or detailed summaries from parsed academic content (`ParsedContent`).

Request body (JSON):

- `parsed_content` (object): ParsedContent previously returned by `/parse/semantic`.
- `mode` (string): `brief` | `detailed` | `both` (default `both`).
- `options` (object): reserved for future use.

Response (200):

- `success` (bool)
- `summary_brief` (string | null)
- `summary_detailed` (string | null)
- `metadata` (object): processing_time_ms, mode, cache_hit, model_used
- `request_id` (string)

Errors:

- `400` - invalid request (e.g., unsupported mode)
- `502` - OpenAI API error
- `504` - OpenAI timeout
- `500` - internal error

Example curl:

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"parsed_content": { /* ParsedContent object */ }, "mode":"both"}'
```

## Testing and Coverage

Run tests locally with pytest. The test environment file `ai/.env.test` provides sane defaults for CI and local runs.

Run all tests with coverage and enforce the 80% threshold:

```powershell
cd ai
pytest --cov=modules --cov-report=xml --cov-report=term --cov-fail-under=80
```

Or run a single file:

```powershell
pytest tests/unit/test_llm_parser.py -q
```

Coverage settings are defined in `ai/.coveragerc` (source: `modules/`, omit tests/scripts/main.py). CI runs the same pytest command to ensure consistent enforcement of the 80% threshold.


Caching:

- Summaries are cached in Redis using a deterministic key derived from a SHA-256 hash of the `parsed_content` and the `mode`.
- Configure TTL via `REDIS_CACHE_TTL` (default 3600 seconds). Disable caching with `REDIS_CACHE_ENABLED=false`.

### Summarization Settings

- **SUMMARIZER_BRIEF_MAX_TOKENS**: Controls the maximum number of tokens returned for brief summaries (short, 2-3 sentence summaries). Default: `500`.
- **SUMMARIZER_DETAILED_MAX_TOKENS**: Controls the maximum number of tokens returned for detailed summaries (long, structured summaries). Default: `1500`.
- **SUMMARIZER_TEMPERATURE**: LLM temperature specific to summarization calls. Lower is more deterministic; typical default: `0.3`.
- **REDIS_CACHE_ENABLED**: Enable/disable Redis caching for summaries and other cached items. Set to `false` to disable caching. Default: `true`.
- **REDIS_CACHE_TTL**: Time-to-live (seconds) for cached summaries. Default: `3600` (1 hour).

Notes:
- Redis is used to cache summaries keyed by a SHA-256 hash of the `ParsedContent` plus the requested `mode`. This reduces repeated LLM calls for identical inputs and modes.
- The summarizer module will emit structured `summarization` logs for both cache hits and misses to aid observability.

## Flashcards Generation

### POST /flashcards/generate
Generate flashcards from parsed academic content using LLM.

Request body:

```json
{
  "parsed_content": { /* ParsedContent object */ },
  "count": 10,
  "options": {
    "min_difficulty": 0,
    "max_difficulty": 5,
    "include_formulas": true,
    "include_concepts": true
  }
}
```

Response:

```json
{
  "success": true,
  "flashcards": [
    {
      "question": "What is photosynthesis?",
      "answer": "Process by which plants convert light energy...",
      "difficulty": 3,
      "context": "Biology - Photosynthesis",
      "source_type": "parsed_question"
    }
  ],
  "metadata": {
    "processing_time_ms": 1500,
    "flashcard_count": 10,
    "source_counts": { "parsed_questions": 5, "generated": 5 },
    "model_used": "gpt-4"
  },
  "request_id": "abc123"
}
```

Error codes:

- 400: Invalid count or options
- 422: Validation failed (invalid ParsedContent)
- 502: OpenAI API error
- 504: LLM timeout
- 500: Internal error

Configuration:

- `FLASHCARD_MAX_TOKENS`: Max tokens for flashcard generation (default: 1500)
- `FLASHCARD_TEMPERATURE`: LLM temperature for flashcards (default: 0.5)
- `FLASHCARD_MIN_COUNT`: Minimum flashcards to generate (default: 5)
- `FLASHCARD_MAX_COUNT`: Maximum flashcards per request (default: 50)

Module:

- `modules/flashcards/` - LLM-based flashcard generator (`spaced_repetition.py`). Extracts Q&A pairs from ParsedContent (topics, concepts, formulas), maps difficulty levels to SM-2 scale (0-5), and generates extra flashcards via LLM if needed.


## Tools: Quiz & Mindmap Generation

### POST /tools/generate-quiz
Generate academic quiz questions from parsed content with multiple question types (MCQ, true/false, short answer).

Request body:

- `parsed_content` (ParsedContent, required): Structured academic content
- `question_count` (int, optional, default 10): Number of questions (1-20)
- `question_types` (array of strings, optional): Question types to include ['mcq','true_false','short_answer']
- `options` (object, optional): Additional generation options

Response (200):

- `success` (bool)
- `quiz` (object): Contains `questions` array (each with `question`, `type`, `options`, `correct_answer`, `explanation`, `difficulty`, `points`), `total_points`
- `metadata` (object): processing_time_ms, model_used, question_count, types_requested
- `request_id` (string)

Error responses: 400 (invalid input), 422 (validation failed), 502 (LLM API error), 504 (timeout), 500 (generation failed)

### POST /tools/generate-mindmap
Generate hierarchical concept mindmap from parsed content with nodes and edges.

Request body:

- `parsed_content` (ParsedContent, required): Structured academic content
- `options` (object, optional): Additional generation options

Response (200):

- `success` (bool)
- `mindmap` (object): Contains `nodes` array (each with `id`, `label`, `level`, `node_type`, `content`), `edges` array (each with `source`, `target`, `relationship`, `label`), `root_node_id`
- `metadata` (object): processing_time_ms, model_used, node_count, edge_count
- `request_id` (string)

Error responses: 422 (validation failed), 502 (LLM API error), 504 (timeout), 500 (generation failed)

Configuration:

- `QUIZ_MAX_TOKENS`: Maximum tokens for quiz generation (default 2000)
- `QUIZ_TEMPERATURE`: Temperature for quiz LLM calls (default 0.5)
- `QUIZ_MAX_QUESTIONS`: Maximum questions per request (default 20)
- `MINDMAP_MAX_TOKENS`: Maximum tokens for mindmap generation (default 2000)
- `MINDMAP_TEMPERATURE`: Temperature for mindmap LLM calls (default 0.4)


## Knowledge Graph Module

The knowledge graph module builds hierarchical concept graphs in AWS Neptune
from parsed academic content. It creates Topic, Subtopic, Concept, and Formula
nodes with relationships (CONTAINS, RELATES_TO, PREREQUISITE, USED_IN) enabling
semantic search, learning path discovery, and concept visualization.

### Components

- `NeptuneConnector`: singleton Gremlin client with connection pooling, IAM/static
  auth support, health checks and retry logic.
- `GraphBuilder`: transforms `ParsedContent` (from the semantic parser) into
  Neptune vertices and edges and returns created vertex ids.
- `GraphQueries`: traversal utilities for visualization, related-concepts,
  and learning-path discovery.

### Endpoints

1. `POST /graph/build`
   - Request body: `{user_id, note_id, parsed_content}`
   - Response: `{success, nodes_created, edges_created, metadata, request_id}`
   - Errors: `400`, `500`, `503`

2. `GET /graph/visualize/{userId}?depth=2`
   - Response: `{success, nodes, edges, metadata, request_id}`
   - Errors: `500`, `503`

3. `GET /graph/related-concepts/{conceptId}?limit=10`
   - Response: `{success, concepts, metadata, request_id}`
   - Errors: `500`, `503`

### Graph Schema

Node types:
- Topic
- Subtopic
- Concept
- Formula

Edge types:
- `CONTAINS`: Topic -> Subtopic
- `RELATES_TO`: Concept <-> Concept
- `PREREQUISITE`: Concept -> Concept
- `USED_IN`: Formula -> Topic/Subtopic

### Configuration

Relevant env vars:
- `NEPTUNE_ENDPOINT`, `NEPTUNE_PORT`, `NEPTUNE_USE_IAM`
- `NEPTUNE_USERNAME`, `NEPTUNE_PASSWORD`, `NEPTUNE_POOL_SIZE`
- `NEPTUNE_RETRY_ATTEMPTS`, `NEPTUNE_RETRY_MULTIPLIER`, `NEPTUNE_RETRY_MAX_WAIT`
- `GRAPH_VISUALIZE_MAX_NODES`, `GRAPH_VISUALIZE_MAX_EDGES`

### Usage Flow

1. Parse note with `/parse/semantic` to get `ParsedContent`.
2. Call `/graph/build` with `user_id`, `note_id`, and `parsed_content`.
3. Query graph with `/graph/visualize` or `/graph/related-concepts`.
4. Vertex IDs are stored in PostgreSQL `knowledge_graph_nodes` table for metadata tracking.

Note: when `GraphBuilder.sync_to_postgres()` is invoked, the original node properties used when creating vertices are persisted into the `properties` JSONB column of `knowledge_graph_nodes`.

### Error Handling

Custom exceptions and their mapped HTTP codes:
- `NeptuneConnectionError` -> 503
- `NeptuneQueryError` -> 500
- `GraphBuildError` -> 500
- `GraphQueryError` -> 500


### Usage flow

1. Download image from S3 via `FileHandler`.
2. Optionally run `preprocess_image()` (denoise, threshold, deskew).
3. Run TrOCR inference locally (cached model).
4. If confidence < `OCR_CONFIDENCE_THRESHOLD` and `fallback_to_textract` is true,
   call Textract (sync) as a fallback.
5. Return structured text, confidence, and metadata.

### Error handling

The service raises specific exceptions for easier diagnosis:

- `TrOCRModelError` → 500 (model load failure)
- `TrOCRInferenceError` → 500 (inference failure)
- `TextractError` → 502 (Textract call failed)
- `ImagePreprocessingError` → 400 (invalid/corrupt image or preprocessing failure)

### Performance notes

- The TrOCR model is loaded lazily and cached; the first request may be slow
  (model download/load time). GPU usage is automatic when available.
- Preprocessing adds modest overhead (typically 100-500ms) but can improve
  OCR accuracy on low-quality inputs.

## Logging

Structured JSON logs are written to a configurable path (set `LOG_FILE_PATH`).
For local development the default is a relative `./logs` directory; production
Docker images should set `LOG_FILE_PATH` to an absolute path such as `/app/logs`.

## Planned (deferred) features

- Implement `POST /ocr/extract` to accept files or S3 keys, run OCR (TrOCR/Textract),
  and return structured OCR results (text, bounding boxes, confidences).
- Implement `POST /process-note` full pipeline: download -> OCR -> semantic parsing ->
  knowledge graph construction -> flashcard generation -> persist results and events.
- Add more robust lifecycle hooks: graceful prefetching of model artifacts on startup,
  background worker management for long-running transforms, and metrics export.
- Add health/readiness probes that validate optional external services conditionally
  (OpenAI optional vs required, model cache warming, etc.) — currently OpenAI
  readiness is gated by `OPENAI_REQUIRED_FOR_READY`.

## Development

Follow standard Python development practices. Add new modules under `modules/` and update `main.py` routes as needed.
