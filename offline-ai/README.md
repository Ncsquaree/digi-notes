# Offline AI - Free & Open-Source Flashcard & Knowledge Graph System

Fully offline, edge-friendly system for generating flashcards and knowledge graphs from academic text using TensorFlow Lite models.

## 🎯 Features

- **Text Preprocessing:** ✅ Complete (Phase 2)
  - Semantic text cleaning and normalization
  - Intelligent paragraph/sentence chunking
  - Metadata extraction (topic, type, academic level, formulas)
  - Format support: plain text, markdown, PDF-extracted
  
- **Embeddings:** ✅ Complete (Phase 3)
  - USE Lite-compatible pipeline with TFLite/SentencePiece
  - Hash-based fallback for offline/dev environments
  - SQLite storage for semantic search

- **Entity Extraction (NER):** ✅ Complete (Phase 4)
  - MobileBERT-SQuAD TFLite inference for entity extraction
  - NLTK rule-based fallback when model unavailable
  - Entity classification: Concept, Topic, Entity, Formula
  - Deduplication with fuzzy matching
  
- **Flashcard Generation:** ✅ Complete (Phase 5)
  - Rule-based question templates (definition, why/how, difference, formula, application)
  - MobileBERT-SQuAD extractive QA for answer generation
  - Difficulty mapping based on academic level and formula presence
  - Deduplication using Levenshtein similarity
  - SQLite storage with embedding integration
  
- **Knowledge Graph:** ✅ Complete (Phase 6)
  - Build concept/topic/entity nodes in SQLite
  - Infer relationships (RELATED_TO, PART_OF, EXPLAINS)
  - Link flashcards via DERIVED_FROM
  
  - Connect flashcards to KG nodes via cosine similarity (>0.7 threshold)
  - Fallback linking ensures every flashcard has ≥1 connection
  - K-means clustering for grouping similar flashcards
  - Query API for retrieving related flashcards/nodes
  
- **End-to-End CLI & Export:** ✅ Complete (Phase 8)
  - Export flashcards to JSON with metadata (difficulty, source type distribution)
  - Export knowledge graph to JSON (nodes, edges)
  - RAM usage monitoring (<1GB target on edge devices)
  - Production-ready summary with file paths and statistics
- **100% Offline:** No API calls, no internet required
- **Edge-Friendly:** Runs on CPU with <1GB RAM

## 🏗️ Architecture

```
Input Text → Preprocess → Extract Entities → Generate Flashcards
                ↓              ↓                    ↓
            Chunk Text    Build KG Nodes      Embed Cards
                ↓              ↓                    ↓
            Embeddings ← Link Semantically → Store in SQLite

  ## 🔬 Formula Detection
```

### Models Used


### Database Schema
- `edges`: Relationships (RELATED_TO, PART_OF, EXPLAINS)
- `embeddings`: 512-dim vectors for semantic search

### Prerequisites
- Python 3.10+
- ~200MB disk space for models
- ~100MB for dependencies

### Installation

```bash
pip install -r requirements.txt

# Download NLTK data (required for text tokenization)
python scripts/download_nltk_data.py
python scripts/download_models.py

# Initialize SQLite database
python scripts/init_db.py
```

### Verify Setup

```bash
# Check models
ls -lh models/
# Expected: mobilebert-squad.tflite (25MB), use-lite.tflite (50MB)

# Check database
sqlite3 offline_ai.db ".tables"
# Expected: nodes, edges, embeddings, flashcards
```

## 📖 Usage

### Phase 2: Text Preprocessing (✅ Complete)

The preprocessing module cleanses, chunks, and extracts metadata from academic text:

```python
from utils.preprocess import preprocess_text

# Load text from file
text = open('academic_paper.txt').read()

# Preprocess with semantic chunking
chunks = preprocess_text(text, max_chunk_size=500, input_format='plain')

# Each chunk contains text and metadata
for chunk in chunks:
    print(f"Type: {chunk['metadata']['type']}")        # heading, formula, list, paragraph
    print(f"Topic: {chunk['metadata']['topic']}")      # Extracted topic or heading
    print(f"Text: {chunk['text'][:100]}...")
    print(f"Academic Level: {chunk['metadata']['academic_level']}")  # basic, intermediate, advanced
```

**Supported Formats:**
[
    {
        'text': 'Chunk of text...',
        'metadata': {
            'type': 'heading|formula|list|paragraph',
            'topic': 'Extracted topic name',
            'char_count': 152,
            'word_count': 24,
            'has_formula': False,
            'academic_level': 'intermediate'
        }
    },
    # ... more chunks
]
```
**Advanced Options:**

```python
from utils.preprocess import clean_text, chunk_text, preprocess_text

# Fine-grained control over preprocessing behavior

# Option 1: Remove markdown syntax while cleaning
cleaned = clean_text(text, preserve_markdown=False)
# Removes: #, -, *, list markers, link syntax

# Option 2: Remove mathematical symbols
cleaned = clean_text(text, preserve_formulas=False)
# Removes: =, +, ×, ÷, →, subscripts, superscripts

# Option 3: Aggressive chunking (ignore paragraph structure)
chunks = chunk_text(cleaned, max_chunk_size=500, preserve_structure=False)
# Combines paragraphs for more aggressive chunking
# Useful when structure is less important than size optimization

# Example: Strip all formatting and do flow-based chunking
text = open('file.txt').read()
cleaned = clean_text(text, preserve_markdown=False, preserve_formulas=False)
chunks = chunk_text(cleaned, max_chunk_size=500, preserve_structure=False)
```

### CLI (Phase 8)

Run the production CLI documented below in Phase 8.

### Phase 3: Embeddings (In progress)

The embeddings module offers a TFLite inference path and a deterministic hash-based fallback when the model or tokenizer is missing. Results are normalized and can be stored in SQLite for similarity search.

```python
from utils.embeddings import EmbeddingGenerator
from utils.preprocess import preprocess_text

emb = EmbeddingGenerator(db_path="offline_ai.db")
embeddings = emb.embed_and_store_chunks(preprocess_text("Sample text"))
print(embeddings[0].embedding[:5])  # First 5 dims
```

**Similarity search:**
```python
results = emb.similarity_search("search phrase", top_k=3)
for hit in results:
  print(hit["chunk_id"], hit["score"])
```

### Python API

```python
from flashcards.generate import generate_flashcards
from knowledge_graph.build import GraphBuilder
from utils.embeddings import generate_embedding

# Generate flashcards
text = "Photosynthesis is the process by which plants..."
flashcards = generate_flashcards(text, count=10)
gb = GraphBuilder(chunks_with_entities, embedding_generator=emb, db_path="offline_ai.db", flashcard_ids=ids)
graph_result = gb.build_graph()
print(f"Created {graph_result['nodes_created']} nodes, {graph_result['edges_created']} edges")

from knowledge_graph.visualize import visualize_graph, export_graph_json
visualize_graph("offline_ai.db", output_path="offline-ai/data/knowledge_graph.png")
export_graph_json("offline_ai.db", output_path="offline-ai/data/graph.json")

# Generate embeddings
embedding = generate_embedding("What is photosynthesis?")
```

## 🧪 Testing

Run the preprocessing test suite to verify Phase 2 implementation:

```bash
cd offline-ai

# Install test dependencies (already in requirements.txt)
pip install pytest pytest-cov

# Run unit tests
pytest tests/test_preprocess.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Run all tests with coverage report
pytest tests/ -v --cov=utils.preprocess --cov-report=term-missing --cov-report=html
```

**Expected test results:**
- Unit tests: 25+ tests for clean_text, chunk_text, extract_metadata
- Integration tests: 10+ end-to-end tests
- Coverage: ≥80% for preprocess.py

Example test execution:
```
tests/test_preprocess.py::TestCleanText::test_clean_text_removes_extra_whitespace PASSED
tests/test_preprocess.py::TestChunkText::test_chunk_text_splits_by_paragraphs PASSED
tests/test_ner.py::TestEntityExtractor::test_extract_entities_with_fallback PASSED
tests/test_integration.py::TestPreprocessingIntegration::test_full_pipeline_with_academic_text PASSED
...
================================ 50+ passed in 3.5s ================================
```

### Phase 4: Entity Extraction (✅ Complete)

The NER module extracts academic entities using MobileBERT-SQuAD (repurposed for entity extraction) or NLTK fallback:

```python
from utils.ner import EntityExtractor

# Initialize with model auto-loading
ner = EntityExtractor(auto_load=True)

# Extract entities from text
text = "Photosynthesis converts CO2 and H2O into glucose."
entities = ner.extract_entities(text)

for entity in entities:
    print(f"{entity.text} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")

chunks_with_entities = ner.extract_from_chunks(chunks)

# Deduplicate entities
unique = ner.deduplicate_entities(all_entities)
```

- **Concept:** Academic terms (algorithm, theorem, process)
- **Topic:** Capitalized phrases (Quantum Mechanics, Machine Learning)
- **Entity:** Proper nouns (Einstein, Newton, DNA)

**Fallback Behavior:**
- If MobileBERT model unavailable → NLTK NER + noun phrase extraction
- If NLTK fails → returns empty list with warning log
- Confidence: MobileBERT (0.8-1.0), NLTK (0.6-0.7)

**Troubleshooting:**
- If "MobileBERT model not found" → Run `python scripts/download_models.py`
- If extraction seems incomplete → Check if model downloaded correctly (~25MB)
- If using fallback automatically → This is expected when model unavailable

### Phase 5: Flashcard Generation (✅ Complete)
### Phase 6: Knowledge Graph Construction (✅ Complete)

Build a knowledge graph from extracted entities and chunks:

```python
from knowledge_graph.build import GraphBuilder
from utils.embeddings import EmbeddingGenerator
from utils.ner import EntityExtractor

emb_gen = EmbeddingGenerator(db_path="offline_ai.db", auto_load=True)
ner = EntityExtractor(auto_load=True)
chunks_with_entities = ner.extract_from_chunks(chunks)

gb = GraphBuilder(
  chunks_with_entities=chunks_with_entities,
  embedding_generator=emb_gen,
  db_path="offline_ai.db",
  flashcard_ids=ids  # from Phase 5 storage
)
result = gb.build_graph()
print(f"Created {result['nodes_created']} nodes, {result['edges_created']} edges")

from knowledge_graph.visualize import visualize_graph, export_graph_json
visualize_graph("offline_ai.db", output_path="offline-ai/data/knowledge_graph.png")
export_graph_json("offline_ai.db", output_path="offline-ai/data/graph.json")
```

**Node Types:**
- Concept, Topic, Entity, Flashcard

**Edge Types:**
- RELATED_TO, PART_OF, EXPLAINS, DERIVED_FROM

### Phase 7: Semantic Linking (✅ Complete)

Links flashcards to knowledge graph nodes using embedding-based cosine similarity:

```python
from knowledge_graph.link import link_flashcards_to_graph, cluster_flashcards
from utils.embeddings import EmbeddingGenerator

emb_gen = EmbeddingGenerator(db_path="offline_ai.db", auto_load=True)

# Link flashcards to KG nodes
result = link_flashcards_to_graph(
    db_path="offline_ai.db",
    similarity_threshold=0.7,
    embedding_generator=emb_gen
)

print(f"Created {result['links_created']} semantic links")
print(f"Processed {result['flashcards_processed']} flashcards")
print(f"Average similarity: {result['avg_similarity']:.3f}")

# Optional: Cluster similar flashcards
clusters = cluster_flashcards(db_path="offline_ai.db", n_clusters=5)
print(f"Grouped flashcards into {len(set(clusters.values()))} clusters")
```

**Features:**
- **Threshold-based linking:** Edges created for similarity >0.7
- **Fallback linking:** Every flashcard guaranteed ≥1 connection (best match)
- **Relationship types:** `DERIVED_FROM` for Concept/Entity, `EXPLAINS` for Topic
- **Clustering:** K-means grouping of similar flashcards for spaced repetition

**Query API:**

```python
from knowledge_graph.link import find_similar_flashcards_for_node, find_similar_nodes_for_flashcard

# Find flashcards related to a concept
flashcards = find_similar_flashcards_for_node("offline_ai.db", "photosynthesis", top_k=5)
for fc in flashcards:
    print(f"{fc['question']} (similarity: {fc['similarity']:.3f})")

# Find nodes related to a flashcard
nodes = find_similar_nodes_for_flashcard("offline_ai.db", flashcard_id=1, top_k=5)
for node in nodes:
    print(f"{node['label']} ({node['node_type']}) - {node['similarity']:.3f}")
### Phase 8: End-to-End CLI & Export (✅ Complete)

Run the full pipeline from input text to structured JSON outputs:

```bash
# Initialize database (one-time setup)
python scripts/init_db.py

# Run full pipeline: preprocess → embed → NER → flashcards → graph → linking → export
python main.py data/raw/sample_input.txt

# Or with custom input file
python main.py path/to/your/academic_text.txt
```

Verify outputs exist and review metadata:

```bash
ls -lh data/flashcards.json data/graph.json
```

Expected RAM usage example (printed in summary):

```
Edge-Device Stats:
  RAM Usage: 420 MB (✓ <1GB target)
  CPU-only inference: ✓ (TFLite no GPU)
  Offline: ✓ (100% - no API calls)
```

**Generated Outputs:**
- `data/flashcards.json`: Q&A pairs with metadata (difficulty distribution, source types)
- `data/graph.json`: Knowledge graph structure (nodes, edges, relationships)
- `offline_ai.db`: SQLite database (flashcards, nodes, edges, embeddings tables)

**Output Example (flashcards.json snippet):**
```json
{
  "metadata": {
    "generated_at": "2025-12-27T10:30:45.123456",
    "source_text": "data/raw/sample_input.txt",
    "total_flashcards": 15,
    "by_difficulty": {"1": 2, "2": 5, "3": 8},
    "by_source_type": {"rule_based": 10, "qa_extracted": 5}
  },
  "flashcards": [
    {
      "id": 1,
      "question": "What is photosynthesis?",
      "answer": "Process of converting light energy to chemical energy",
      "difficulty": 2,
      "topic": "Biology",
      "source_type": "rule_based",
      "embedding_id": 42
    }
  ]
}
```

**Edge-Device Verified:**
- RAM Usage: <1GB (auto-monitored, warning if exceeded)
- CPU-only inference (no GPU required, TFLite architecture)
- 100% offline (no API calls, all local processing)
- Processing time: ~2-5 seconds for ~5000-word academic text on standard CPU


```


The flashcard generator creates Q&A pairs using rule-based templates and MobileBERT extractive QA:

```python
from flashcards.generate import FlashcardGenerator
from utils.ner import EntityExtractor
from utils.preprocess import preprocess_text

# Initialize
ner = EntityExtractor(auto_load=True)
gen = FlashcardGenerator(db_path="offline_ai.db", ner_extractor=ner)

# Preprocess and extract entities
chunks = preprocess_text(text)
chunks_with_entities = ner.extract_from_chunks(chunks)

# Generate flashcards
flashcards = gen.generate_from_chunks(chunks_with_entities)

# Store in SQLite
ids = gen.store_flashcards(flashcards)

# Display samples
for fc in flashcards[:3]:
    print(f"Q: {fc.question}")
    print(f"A: {fc.answer}")
    print(f"Difficulty: {fc.difficulty}/5")
    print(f"Entities: {', '.join(fc.entities)}")
    print()
```

**Question Types Generated:**
- **Definition:** "What is X?" → "X is a process/concept that..."
- **Why/How:** "Why/How does X?" → Causal explanations
- **Difference:** "What is the difference between X and Y?"
- **Formula:** "What is the formula for X?" → Mathematical/chemical expressions
- **Application:** "How is X applied in...?" → Real-world use cases

**Difficulty Mapping:**
- **Level 1-2:** Basic concepts, simple definitions (academic_level: basic)
- **Level 2-3:** Intermediate explanations (academic_level: intermediate)
- **Level 4-5:** Advanced topics, formulas (academic_level: advanced + has_formula: true)

**Output Format:**
```json
[
  {
    "question": "What is photosynthesis?",
    "answer": "Process by which plants convert light to chemical energy",
    "topic": "Photosynthesis",
    "entities": ["photosynthesis", "light energy"],
    "embedding_id": 42,
    "difficulty": 2,
    "source_type": "rule_based"
  }
]
```

**Deduplication:**
- Removes identical questions (exact match)
- Merges similar questions (Levenshtein similarity > 0.8)
- Keeps highest-quality answer for each unique question

**Troubleshooting:**
- If few flashcards generated → Check if chunks have sufficient text (>100 chars)
- If MobileBERT model missing → Falls back to rule-based answers
- If answers seem incomplete → Ensure entities were extracted in Phase 4

## 📊 Output Formats

### Flashcards JSON
```json
{
  "flashcards": [
    {
      "question": "What is photosynthesis?",
      "answer": "Process by which plants convert light to chemical energy",
      "difficulty": 1,
      "context": "Biology - Plant Processes",
      "embedding_id": 42
    }
  ],
  "metadata": {
    "model_used": "mobilebert-squad.tflite",
    "flashcard_count": 10
  }
}
```

### Knowledge Graph JSON
```json
{
  "nodes": [
    {"id": 1, "type": "Concept", "label": "Photosynthesis", "properties": {...}}
  ],
  "edges": [
    {"source": 1, "target": 2, "type": "RELATED_TO", "weight": 0.85}
  ]
}
```

## 🔧 Extending

### Add New Models
1. Download TFLite model to `models/`
2. Update `scripts/download_models.py`
3. Create loader in `utils/`

### Custom Flashcard Types
1. Modify `flashcards/generate.py`
2. Update `flashcards` table schema
3. Add new difficulty mapping

## 📚 References

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileBERT Paper](https://arxiv.org/abs/2004.02984)
- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- [NLTK Documentation](https://www.nltk.org/)
- [Phase 2: Text Preprocessing Implementation](utils/preprocess.py)

## 📋 Implementation Status

### Phase 1: Setup & Dependencies ✅ Complete
- Directory structure
- requirements.txt (TFLite, NLTK, SQLite)
- Model download script
- Database initialization
- Verification script

### Phase 2: Text Preprocessing ✅ Complete
- Text cleaning and normalization
- Semantic text chunking (paragraph + sentence)
- Metadata extraction (type, topic, academic level, formulas)
- Format handlers (markdown, PDF, plain text)
- Comprehensive test suite (35+ tests)
- Integration tests with downstream modules

### Phase 3: Embeddings ✅ Complete
- TFLite USE Lite model integration
- SentencePiece tokenization
- Hash-based fallback for testing
- BLOB storage in SQLite
- Batch processing support
- Similarity search

### Phase 4: Entity Extraction (NER) ✅ Complete
- MobileBERT-SQuAD TFLite integration
- QA repurposing for entity extraction
- NLTK fallback (NER + noun phrases)
- Entity classification (Concept, Topic, Entity, Formula)
- Deduplication with fuzzy matching
- Comprehensive test suite (20+ tests)
- Integration with preprocessing pipeline

### Phase 5: Flashcard Generation ✅ Complete
- Rule-based question templates (5 types: definition, why/how, difference, formula, application)
- MobileBERT-SQuAD extractive QA for answer generation
- Difficulty mapping (1-5 scale based on academic level + formula presence)
- Deduplication using Levenshtein similarity (>0.8 threshold)
- SQLite storage with embedding integration (foreign key to embeddings table)
- Comprehensive test suite (15+ tests)
- Integration with NER pipeline
- Sample output (15 flashcards from photosynthesis text)

### Phase 6: Knowledge Graph Construction ✅ Complete
- Node creation (Concept, Topic, Entity, Flashcard types)
- Relationship inference (RELATED_TO, PART_OF, EXPLAINS, DERIVED_FROM)
- Embedding integration for semantic search
- Graph visualization and JSON export
- Query API for traversing relationships

### Phase 7: Semantic Linking ✅ Complete
- Embedding-based cosine similarity linking (>0.7 threshold)
- Fallback linking ensures every flashcard has ≥1 connection
- Relationship type selection (DERIVED_FROM for Concept/Entity, EXPLAINS for Topic)
- K-means clustering for grouping similar flashcards
- Query API: find_similar_flashcards_for_node(), find_similar_nodes_for_flashcard()
- Comprehensive test suite (10+ tests)

### Phase 8: End-to-End CLI & Export ✅ Complete
- Full pipeline orchestration: `python main.py data/raw/sample_input.txt`
- Exports: `data/flashcards.json`, `data/graph.json`, `offline_ai.db`
- RAM monitoring with psutil (<1GB verified)
- Edge-device stats and final summary printed

## 📄 License

MIT License - Free for commercial and personal use.

## 🤝 Contributing

See main project `CONTRIBUTING.md` for guidelines.
