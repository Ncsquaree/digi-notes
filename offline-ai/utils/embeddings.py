"""
Phase 3: Embedding utilities for the offline AI pipeline.

Features
--------
- Optional TensorFlow Lite inference for Universal Sentence Encoder Lite.
- Fallback deterministic hashing when model/tokenizer are unavailable (keeps tests fast).
- Batch helpers, cosine similarity, and lightweight SQLite storage for semantic search.

Design notes
------------
- The module degrades gracefully: if the TFLite model or SentencePiece tokenizer is
  missing, it generates deterministic embeddings based on a seeded hash of the text.
- SQLite storage uses JSON for vectors to keep dependencies minimal.
- Embeddings are normalized to unit length when possible to simplify similarity math.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # Optional dependency; fallback logic handles absence
  import tflite_runtime.interpreter as tflite
except Exception:  # pragma: no cover - environment-dependent
  tflite = None

try:  # Optional dependency; fallback logic handles absence
  import sentencepiece as spm
except Exception:  # pragma: no cover - environment-dependent
  spm = None


DEFAULT_DIM = 512


class EmbeddingError(Exception):
  """Raised when embedding generation fails irrecoverably."""


def _hash_embedding(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
  """Deterministic, fast embedding based on text hash (fallback path)."""
  seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
  rng = np.random.default_rng(seed)
  vec = rng.standard_normal(dim, dtype=np.float32)
  norm = np.linalg.norm(vec)
  if norm > 0:
    vec = vec / norm
  return vec


def _tokenize(text: str, tokenizer: Optional["spm.SentencePieceProcessor"], max_seq_len: int) -> np.ndarray:
  """Tokenize text using SentencePiece when available, else whitespace split."""
  if tokenizer is None:
    tokens = text.split()
  else:
    tokens = tokenizer.encode(text, out_type=int)

  tokens = tokens[:max_seq_len]
  if len(tokens) < max_seq_len:
    tokens = tokens + [0] * (max_seq_len - len(tokens))
  return np.array(tokens, dtype=np.int32).reshape(1, max_seq_len)


def generate_embedding(
  text: str,
  *,
  interpreter: Optional["tflite.Interpreter"] = None,
  tokenizer: Optional["spm.SentencePieceProcessor"] = None,
  max_seq_len: int = 128,
  embedding_dim: int = DEFAULT_DIM,
) -> List[float]:
  """Generate an embedding for the given text.

  Falls back to a deterministic hash-based vector when model/tokenizer are missing.

  Args:
    text: Input text.
    interpreter: Optional TFLite interpreter for USE Lite.
    tokenizer: Optional SentencePiece tokenizer for USE Lite.
    max_seq_len: Maximum token length for model input.
    embedding_dim: Desired embedding dimensionality.

  Returns:
    List[float]: Embedding vector (length = embedding_dim).
  """
  if text is None:
    raise ValueError("Input text must not be None")

  stripped = text.strip()
  if not stripped:
    return [0.0] * embedding_dim

  use_model = interpreter is not None and tokenizer is not None

  if use_model:
    try:
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()

      input_data = _tokenize(stripped, tokenizer, max_seq_len)
      interpreter.set_tensor(input_details[0]["index"], input_data)
      interpreter.invoke()

      output_data = interpreter.get_tensor(output_details[0]["index"]).flatten()
      if output_data.size == 0:
        raise EmbeddingError("TFLite model returned empty output")

      vec = np.array(output_data, dtype=np.float32)
      if vec.shape[0] > embedding_dim:
        vec = vec[:embedding_dim]
      elif vec.shape[0] < embedding_dim:
        padding = np.zeros(embedding_dim - vec.shape[0], dtype=np.float32)
        vec = np.concatenate([vec, padding])
      norm = np.linalg.norm(vec)
      if norm > 0:
        vec = vec / norm
      return vec.astype(float).tolist()
    except Exception as exc:  # pragma: no cover - exercised in integration
      logger.warning("Falling back to hash-based embedding: %s", exc)

  # Fallback: deterministic hashed embedding
  vec = _hash_embedding(stripped, dim=embedding_dim)
  return vec.astype(float).tolist()


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
  """Compute cosine similarity between two vectors."""
  a = np.array(list(vec_a), dtype=np.float32)
  b = np.array(list(vec_b), dtype=np.float32)

  if a.size == 0 or b.size == 0:
    return 0.0

  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  if norm_a == 0 or norm_b == 0:
    return 0.0

  return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class EmbeddingResult:
  chunk_id: str
  text: str
  embedding: List[float]
  metadata: Optional[dict] = None


class EmbeddingGenerator:
  """High-level helper to load models, generate embeddings, and store them."""

  def __init__(
    self,
    *,
    model_path: Optional[Path] = None,
    spm_path: Optional[Path] = None,
    db_path: str = ":memory:",
    max_seq_len: int = 128,
    embedding_dim: int = DEFAULT_DIM,
    auto_load: bool = True,
    force_hash_fallback: bool = False,
  ) -> None:
    # Default model paths if not provided
    if model_path is None:
      default_model = Path(__file__).parent.parent / "models" / "use-lite.tflite"
      self.model_path = default_model if default_model.exists() else None
    else:
      self.model_path = Path(model_path)
    
    # Optional SentencePiece path
    if spm_path is None:
      default_spm = Path(__file__).parent.parent / "models" / "use-lite.spm"
      self.spm_path = default_spm if default_spm.exists() else None
    else:
      self.spm_path = Path(spm_path)
    
    self.db_path = db_path
    self.max_seq_len = max_seq_len
    self.embedding_dim = embedding_dim
    self.force_hash_fallback = force_hash_fallback

    self.interpreter: Optional["tflite.Interpreter"] = None
    self.tokenizer: Optional["spm.SentencePieceProcessor"] = None
    self._db_conn: Optional[sqlite3.Connection] = None

    if auto_load and not force_hash_fallback:
      self.load()

  # ------------------------------------------------------------------
  # Loading
  # ------------------------------------------------------------------
  def load(self) -> None:
    """Load tokenizer and model if available."""
    self._load_tokenizer()
    self._load_model()

  def _load_tokenizer(self) -> None:
    if self.spm_path is None:
      logger.info("No SentencePiece model provided; using whitespace tokenizer")
      return
    if spm is None:
      logger.warning("sentencepiece not installed; falling back to whitespace tokenizer")
      return
    if not self.spm_path.exists():
      logger.warning("SentencePiece model not found at %s; using whitespace tokenizer", self.spm_path)
      return

    self.tokenizer = spm.SentencePieceProcessor(model_file=str(self.spm_path))
    logger.info("Loaded SentencePiece model from %s", self.spm_path)

  def _load_model(self) -> None:
    if self.force_hash_fallback:
      logger.info("force_hash_fallback=True; skipping model load")
      return
    
    if self.model_path is None:
      msg = "No TFLite model found. Run 'python scripts/download_models.py' to download USE Lite model."
      logger.error(msg)
      raise EmbeddingError(msg)
    
    if tflite is None:
      msg = "tflite_runtime not installed. Run 'pip install tflite-runtime'."
      logger.error(msg)
      raise EmbeddingError(msg)
    
    if not self.model_path.exists():
      msg = f"TFLite model not found at {self.model_path}. Run 'python scripts/download_models.py'."
      logger.error(msg)
      raise EmbeddingError(msg)

    interpreter = tflite.Interpreter(model_path=str(self.model_path))
    interpreter.allocate_tensors()
    self.interpreter = interpreter
    logger.info("Loaded TFLite model from %s", self.model_path)

  # ------------------------------------------------------------------
  # Embedding generation
  # ------------------------------------------------------------------
  def embed_text(self, text: str) -> List[float]:
    return generate_embedding(
      text,
      interpreter=self.interpreter,
      tokenizer=self.tokenizer,
      max_seq_len=self.max_seq_len,
      embedding_dim=self.embedding_dim,
    )

  def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
    """Batch embed multiple texts with a single model invocation when possible.
    
    Note: Current USE Lite models typically expect single-sequence input.
    This method attempts true batching but falls back to per-item if unsupported.
    """
    text_list = list(texts)
    if not text_list:
      return []
    
    # If no model loaded, process individually with hash fallback
    if self.interpreter is None or self.tokenizer is None:
      return [self.embed_text(t) for t in text_list]
    
    # Attempt batch inference
    try:
      input_details = self.interpreter.get_input_details()
      output_details = self.interpreter.get_output_details()
      input_shape = input_details[0]["shape"]
      
      # Check if model supports batch dimension > 1
      if len(input_shape) < 2 or input_shape[0] != 1:
        # Model expects single input; process individually
        return [self.embed_text(t) for t in text_list]
      
      # Tokenize all texts
      tokenized = [_tokenize(t.strip() if t else "", self.tokenizer, self.max_seq_len) for t in text_list]
      
      # Stack into batch (shape: [batch_size, max_seq_len])
      batch_input = np.vstack(tokenized)
      
      # Resize input tensor if needed
      self.interpreter.resize_tensor_input(input_details[0]["index"], batch_input.shape)
      self.interpreter.allocate_tensors()
      
      # Single invocation for entire batch
      self.interpreter.set_tensor(input_details[0]["index"], batch_input)
      self.interpreter.invoke()
      
      # Extract outputs
      output_data = self.interpreter.get_tensor(output_details[0]["index"])
      
      # Split back into per-text embeddings
      embeddings = []
      for i in range(len(text_list)):
        vec = np.array(output_data[i], dtype=np.float32).flatten()
        if vec.shape[0] > self.embedding_dim:
          vec = vec[:self.embedding_dim]
        elif vec.shape[0] < self.embedding_dim:
          padding = np.zeros(self.embedding_dim - vec.shape[0], dtype=np.float32)
          vec = np.concatenate([vec, padding])
        norm = np.linalg.norm(vec)
        if norm > 0:
          vec = vec / norm
        embeddings.append(vec.astype(float).tolist())
      
      return embeddings
    
    except Exception as exc:
      # Batch inference failed; fall back to individual processing
      logger.warning("Batch inference failed, processing individually: %s", exc)
      return [self.embed_text(t) for t in text_list]

  # ------------------------------------------------------------------
  # Storage helpers (using init_db.py schema)
  # ------------------------------------------------------------------
  def _ensure_db(self) -> sqlite3.Connection:
    """Open connection to database initialized by init_db.py.
    
    Schema from init_db.py:
      embeddings (id INTEGER PK, vector BLOB, entity_type TEXT, entity_id INTEGER, created_at TIMESTAMP)
    """
    if self._db_conn is None:
      self._db_conn = sqlite3.connect(self.db_path)
    return self._db_conn

  def store_embedding(self, chunk_id: str, text: str, embedding: List[float], entity_type: str = "chunk") -> int:
    """Store embedding in init_db.py schema and return the row id.
    
    Args:
      chunk_id: Identifier for the chunk (stored as entity_id if numeric, else hashed)
      text: Original text (not stored in embeddings table; stored separately if needed)
      embedding: Vector to store
      entity_type: Type classification (default: 'chunk')
    
    Returns:
      int: The id of the inserted row
    """
    conn = self._ensure_db()
    vec_array = np.array(embedding, dtype=np.float32)
    vec_blob = vec_array.tobytes()
    
    # Try to parse chunk_id as integer for entity_id, else use hash
    try:
      if chunk_id.startswith("chunk-"):
        entity_id = int(chunk_id.split("-")[1])
      else:
        entity_id = int(chunk_id)
    except (ValueError, IndexError):
      # Use hash of chunk_id as entity_id
      entity_id = abs(hash(chunk_id)) % (10 ** 9)
    
    cursor = conn.execute(
      "INSERT INTO embeddings(vector, entity_type, entity_id) VALUES(?, ?, ?)",
      (vec_blob, entity_type, entity_id),
    )
    conn.commit()
    return cursor.lastrowid

  def fetch_embedding(self, embedding_id: int) -> Optional[List[float]]:
    """Fetch embedding by its database id.
    
    Args:
      embedding_id: The id (primary key) returned by store_embedding
    
    Returns:
      List[float] or None if not found
    """
    conn = self._ensure_db()
    row = conn.execute("SELECT vector FROM embeddings WHERE id=?", (embedding_id,)).fetchone()
    if row is None:
      return None
    vec_blob = row[0]
    vec_array = np.frombuffer(vec_blob, dtype=np.float32)
    return vec_array.tolist()
  
  def fetch_embedding_by_entity(self, entity_type: str, entity_id: int) -> Optional[List[float]]:
    """Fetch embedding by entity_type and entity_id."""
    conn = self._ensure_db()
    row = conn.execute(
      "SELECT vector FROM embeddings WHERE entity_type=? AND entity_id=? ORDER BY id DESC LIMIT 1",
      (entity_type, entity_id)
    ).fetchone()
    if row is None:
      return None
    vec_blob = row[0]
    vec_array = np.frombuffer(vec_blob, dtype=np.float32)
    return vec_array.tolist()

  def embed_and_store_chunks(self, chunks: List[dict]) -> List[EmbeddingResult]:
    """Embed and store chunks, returning results with embedding_id in metadata."""
    results: List[EmbeddingResult] = []
    for idx, chunk in enumerate(chunks):
      text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
      chunk_id = chunk.get("id") if isinstance(chunk, dict) else None
      chunk_id = chunk_id or f"chunk-{idx}"
      emb = self.embed_text(text)
      embedding_id = self.store_embedding(chunk_id, text, emb, entity_type="chunk")
      
      # Add embedding_id to metadata
      metadata = chunk.get("metadata") if isinstance(chunk, dict) else None
      if metadata is None:
        metadata = {}
      else:
        metadata = metadata.copy()
      metadata["embedding_id"] = embedding_id
      
      results.append(EmbeddingResult(
        chunk_id=chunk_id,
        text=text,
        embedding=emb,
        metadata=metadata
      ))
    return results

  def similarity_search(self, query_text: str, top_k: int = 5) -> List[dict]:
    """Search for similar embeddings using cosine similarity.
    
    Returns:
      List of dicts with keys: id, entity_type, entity_id, score
    """
    query_vec = self.embed_text(query_text)
    conn = self._ensure_db()
    rows = conn.execute("SELECT id, entity_type, entity_id, vector FROM embeddings").fetchall()
    scored = []
    for row_id, entity_type, entity_id, vec_blob in rows:
      vec_array = np.frombuffer(vec_blob, dtype=np.float32)
      vec = vec_array.tolist()
      score = cosine_similarity(query_vec, vec)
      scored.append({
        "id": row_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "score": score
      })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

  def close(self) -> None:
    if self._db_conn is not None:
      self._db_conn.close()
      self._db_conn = None

