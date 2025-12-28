# Phase 3 Embeddings - Implementation Summary

## Changes Implemented

### Comment 1: Embedding Storage Schema Alignment ✅
**Issue**: Schema conflict between `utils/embeddings.py` and `scripts/init_db.py`

**Solution**:
- Removed local CREATE TABLE definition from `utils/embeddings.py`
- Aligned with canonical schema from `init_db.py`: `(id INTEGER PK, vector BLOB, entity_type TEXT, entity_id INTEGER, created_at TIMESTAMP)`
- Updated `store_embedding()` to:
  - Serialize vectors as BLOB using `numpy.ndarray.tobytes()`
  - Set `entity_type` (default: "chunk") and `entity_id` appropriately
  - Return the inserted row `id`
- Updated `fetch_embedding()` to:
  - Accept `embedding_id` (primary key)
  - Deserialize with `np.frombuffer(blob, dtype=np.float32)`
- Added `fetch_embedding_by_entity()` for entity_type/entity_id lookups
- Updated `embed_and_store_chunks()` to capture and store `embedding_id` in chunk metadata

**Files Modified**:
- `utils/embeddings.py`: Schema alignment, BLOB storage/retrieval
- Tests updated to initialize schema properly

---

### Comment 2: Model Auto-Loading with Clear Errors ✅
**Issue**: USE Lite model never loaded by default; always used hash fallback

**Solution**:
- Added sensible defaults in `EmbeddingGenerator.__init__`:
  - `model_path` defaults to `models/use-lite.tflite` (if exists)
  - `spm_path` defaults to `models/use-lite.spm` (if exists)
  - Changed `auto_load` default to `True`
- Added `force_hash_fallback` parameter for testing
- Updated `_load_model()` to raise clear `EmbeddingError` with instructions:
  - "Run 'python scripts/download_models.py' to download USE Lite model"
  - "Run 'pip install tflite-runtime'" if package missing
- Updated `main.py` to:
  - Attempt auto-loading with `auto_load=True`
  - Fall back to `force_hash_fallback=True` if model unavailable
  - Display whether using TFLite or hash fallback

**Files Modified**:
- `utils/embeddings.py`: Default paths, auto_load=True, force_hash_fallback parameter
- `main.py`: Try/except with fallback and status reporting

---

### Comment 3: True Batch Processing ✅
**Issue**: `embed_batch()` performed per-item calls instead of batched inference

**Solution**:
- Refactored `embed_batch()` to:
  - Tokenize all texts
  - Pad to common length
  - Stack into 2D array (shape: `[batch_size, max_seq_len]`)
  - Resize interpreter input tensor
  - Single `invoke()` call for entire batch
  - Split outputs back per text
  - Normalize each output vector
- Added graceful fallback:
  - If model doesn't support batching (input shape[0] == 1), fall back to per-item
  - If batch inference fails, fall back to per-item with warning
- Added clear docstring documenting behavior and limitations

**Files Modified**:
- `utils/embeddings.py`: `embed_batch()` method completely rewritten

---

### Comment 4: BLOB Storage Instead of JSON ✅
**Issue**: Embeddings stored as JSON text, increasing size and diverging from plan

**Solution**:
- Changed `store_embedding()` to:
  - Convert embedding list to `np.float32` array
  - Serialize with `.tobytes()` → BLOB
- Changed `fetch_embedding()` to:
  - Read BLOB from database
  - Deserialize with `np.frombuffer(blob, dtype=np.float32)`
  - Convert back to list
- Updated `similarity_search()` to deserialize BLOBs
- Removed all JSON serialization/deserialization for vectors
- Schema now uses `vector BLOB` (from init_db.py)

**Files Modified**:
- `utils/embeddings.py`: All storage/retrieval methods
- `tests/test_embeddings.py`: Updated to initialize proper schema

---

## Testing Updates

### Test File: `tests/test_embeddings.py`
- Added schema initialization in each test (mimics `init_db.py`)
- Updated to use `force_hash_fallback=True` for deterministic testing
- Updated API calls:
  - `store_embedding()` now returns `embedding_id`
  - `fetch_embedding()` accepts `embedding_id` instead of `chunk_id`
  - `similarity_search()` returns `id`, `entity_type`, `entity_id`, `score`
- All tests pass with new schema and BLOB storage

---

## Database Schema (from init_db.py)

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vector BLOB NOT NULL,               -- Serialized numpy array (512 floats)
    entity_type TEXT NOT NULL,          -- chunk, flashcard, concept, etc.
    entity_id INTEGER,                  -- FK to nodes or flashcards
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Usage Examples

### Basic Embedding Generation (with model)
```python
from utils.embeddings import EmbeddingGenerator

# Auto-loads model from models/use-lite.tflite if available
gen = EmbeddingGenerator(db_path="offline_ai.db", auto_load=True)
embedding = gen.embed_text("Hello world")
```

### Hash Fallback (for testing)
```python
gen = EmbeddingGenerator(db_path="offline_ai.db", force_hash_fallback=True)
embedding = gen.embed_text("Hello world")
```

### Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = gen.embed_batch(texts)  # Single model invocation
```

### Storage and Retrieval
```python
# Store
embedding_id = gen.store_embedding("chunk-1", "Some text", embedding, entity_type="chunk")

# Fetch by ID
vec = gen.fetch_embedding(embedding_id)

# Fetch by entity
vec = gen.fetch_embedding_by_entity("chunk", 1)
```

### Similarity Search
```python
results = gen.similarity_search("query text", top_k=5)
for hit in results:
    print(f"ID: {hit['id']}, Type: {hit['entity_type']}, Score: {hit['score']:.3f}")
```

---

## Verification

### Run Tests
```bash
cd offline-ai
pytest tests/test_embeddings.py -v
```

### Run CLI
```bash
python main.py sample_input.txt
# Should show:
# - Model: TFLite USE Lite (if model available)
# - Model: Hash fallback (testing) (if model missing)
# - Sample embedding_id: 1
```

### Check Database
```bash
sqlite3 offline_ai.db
.schema embeddings
SELECT id, entity_type, entity_id, length(vector) FROM embeddings LIMIT 5;
```

---

## Error Handling

### Model Missing
```
EmbeddingError: No TFLite model found. Run 'python scripts/download_models.py' to download USE Lite model.
```

### TFLite Not Installed
```
EmbeddingError: tflite_runtime not installed. Run 'pip install tflite-runtime'.
```

### Graceful Fallback in main.py
```python
try:
    emb_gen = EmbeddingGenerator(db_path="offline_ai.db", auto_load=True)
except Exception as emb_err:
    logger.warning(f"Failed to load TFLite model: {emb_err}")
    logger.info("Falling back to hash-based embeddings for testing")
    emb_gen = EmbeddingGenerator(db_path="offline_ai.db", force_hash_fallback=True)
```

---

## Implementation Status

✅ **All 4 verification comments implemented**
- Schema aligned with init_db.py
- BLOB storage for vectors
- Auto-loading with clear error messages
- True batch processing with fallback
- Tests updated and passing
- CLI integration complete with fallback handling

---

## Next Steps

1. Download TFLite models: `python scripts/download_models.py`
2. Run integration tests: `pytest tests/test_integration.py -v`
3. Test with real academic text: `python main.py sample.txt`
4. Proceed to Phase 4: Named Entity Recognition
