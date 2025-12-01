# AI Pipeline

Pipeline stages:

1. Ingestion
	- Receives image (handwritten notes). Store original to S3 `ai/data/raw/`.

2. OCR
	- Use TrOCR (local or HuggingFace) or AWS Textract for handwriting recognition.
	- Output: raw text stored in `ai/data/processed/`.

3. Preprocessing
	- Clean text, segment lines, identify formula blocks, code blocks, and lists.

4. Semantic Parsing (LLM)
	- Use `ai/modules/semantic/llm_parser.py` to extract topics, subtopics, formulas, Q&A.
	- Store structured JSON for downstream consumption.

5. Knowledge Graph
	- Build nodes/edges representing concept relationships using `ai/modules/knowledge_graph/graph_builder.py`.
	- Push to AWS Neptune via `neptune_connector.py`.

6. Summarization & Flashcards
	- `summarizer.py` produces a concise summary.
	- `flashcards/spaced_repetition.py` implements SM-2 to schedule reviews.

