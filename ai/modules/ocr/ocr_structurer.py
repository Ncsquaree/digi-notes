"""Structured OCR Output

Produces a structured schema from raw OCR text with:
- `document_metadata` including source_type, language, page_count, confidence
- `sections` with stable `section_id`, `title`, and typed `content` blocks
- Top-level `raw_text` copy of the input

Heuristics preserve tables and math/code as markdown and provide an optional
LLM fallback that returns the same schema, validated with Pydantic.
"""
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source_type: Optional[str] = Field(default=None, description="e.g., image|pdf|text")
    language: Optional[str] = Field(default=None)
    page_count: Optional[int] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    ocr_method: Optional[str] = Field(default=None)
    line_count: int = 0
    word_count: int = 0


class ContentParagraph(BaseModel):
    type: Literal['paragraph'] = 'paragraph'
    text: str


class ContentList(BaseModel):
    type: Literal['list'] = 'list'
    items: List[str] = Field(default_factory=list)


class ContentTable(BaseModel):
    type: Literal['table'] = 'table'
    markdown: str


class ContentMath(BaseModel):
    type: Literal['math'] = 'math'
    markdown: str


class ContentCode(BaseModel):
    type: Literal['code'] = 'code'
    markdown: str


ContentBlock = ContentParagraph | ContentList | ContentTable | ContentMath | ContentCode


class Section(BaseModel):
    section_id: str
    title: Optional[str] = None
    content: List[ContentBlock] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    document_metadata: DocumentMetadata
    sections: List[Section] = Field(default_factory=list)
    raw_text: str


HEADING_MAX_LEN = 120


def _normalize_whitespace(text: str) -> str:
    # Normalize common OCR artifacts and whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse excessive spaces but preserve single spaces
    text = re.sub(r"[\t\f\v]+", " ", text)
    # Trim trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def _is_heading(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if len(s) == 0 or len(s) > HEADING_MAX_LEN:
        return False
    # All caps or Title Case with few words
    words = s.split()
    if len(words) <= 10 and (s.isupper() or _looks_title_case(words)):
        return True
    # Ends with colon and short
    if s.endswith(":") and len(words) <= 12:
        return True
    # Surrounded by dashes/underscores (common OCR heading style)
    if re.match(r"^[\-_]+\s*[^\-\_].*", s):
        return True
    return False


def _looks_title_case(words: List[str]) -> bool:
    # Consider title case if most words start uppercase
    if not words:
        return False
    cap = sum(1 for w in words if w[:1].isupper())
    return cap >= max(1, int(0.6 * len(words)))


def _is_list_item(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^(?:[-*•]\s+|\d+[\.)]\s+|[a-zA-Z][\.)]\s+)", s))


def _clean_line(line: str) -> str:
    s = line.strip()
    # Remove leading bullet/number markers for storage
    s = re.sub(r"^(?:[-*•]\s+|\d+[\.)]\s+|[a-zA-Z][\.)]\s+)", "", s)
    # Normalize spaces inside line
    s = re.sub(r"\s+", " ", s)
    return s


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _detect_language(text: str) -> Optional[str]:
    try:
        from langdetect import detect  # optional dependency
        return detect(text)
    except Exception:
        # naive heuristic: default to 'unknown'
        return 'unknown'


def _is_table_row(line: str) -> bool:
    s = line.strip()
    if s.count('|') >= 2:
        return True
    return False


def _is_table_divider(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^\s*\|?\s*:?\-+(\s*\|\s*:?\-+)*\s*$", s))


def _is_code_fence(line: str) -> bool:
    return bool(re.match(r"^\s*```", line.strip()))


def _extract_math_blocks(line: str) -> List[str]:
    blocks: List[str] = []
    # $$...$$ or \[...\] or inline $...$
    for pat in [r"\$\$(.+?)\$\$", r"\\\[(.+?)\\\]", r"\$(.+?)\$"]:
        for m in re.finditer(pat, line):
            blocks.append(m.group(0))
    return blocks
    return len(re.findall(r"\b\w+\b", text))


def _new_section(section_counter: int, title: Optional[str]) -> Section:
    return Section(section_id=f"sec-{section_counter}", title=title, content=[])


def _llm_structure(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Call an LLM (OpenAI) to produce structured JSON matching provided schema."""
    import os
    import time
    import json
    try:
        import openai
        model = os.getenv('OPENAI_MODEL', 'gpt-4')
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY not set')
        openai.api_key = api_key
        system = (
            "You convert OCR text into a structured JSON with document_metadata, sections, and raw_text. "
            "Preserve tables and math as markdown. Output only valid JSON matching the parameters schema."
        )
        user = f"Text:\n{text}"
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            functions=[{"name": "structured_document", "parameters": schema}],
            function_call={"name": "structured_document"},
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.3')),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
            timeout=int(os.getenv('OPENAI_TIMEOUT', '60')),
        )
        choices = resp.get('choices', [])
        message = choices[0].get('message', {})
        args = message.get('function_call', {}).get('arguments', '{}')
        return json.loads(args)
    except Exception as e:
        raise RuntimeError(f"LLM structuring failed: {e}")


def structure_document(
    text: str,
    source_type: Optional[str] = None,
    page_count: Optional[int] = None,
    ocr_confidence: Optional[float] = None,
    llm_fallback: bool = False,
    llm_threshold: float = 0.6,
    ocr_method: Optional[str] = None,
) -> StructuredDocument:
    """Produce a structured document from raw OCR text using heuristics, with optional LLM fallback."""
    if not text:
        return StructuredDocument(
            document_metadata=DocumentMetadata(
                source_type=source_type,
                language=None,
                page_count=page_count,
                confidence=ocr_confidence,
                ocr_method=ocr_method,
                line_count=0,
                word_count=0,
            ),
            sections=[],
            raw_text="",
        )

    # Determine if LLM fallback should run
    use_llm = llm_fallback or (ocr_confidence is not None and ocr_confidence < llm_threshold)

    # Build schema from Pydantic
    schema = StructuredDocument.model_json_schema(ref_template='#/definitions/{model}')

    if use_llm:
        try:
            data = _llm_structure(text, schema)
            # Validate
            doc = StructuredDocument.model_validate(data)
            return doc
        except Exception:
            # fall through to heuristics on failure
            pass

    norm = _normalize_whitespace(text)
    lines = [l for l in norm.split("\n")]
    sections: List[Section] = []
    section_counter = 1
    current: Section = _new_section(section_counter, title=None)

    def _commit_section():
        nonlocal current, section_counter
        if current.title or current.content:
            sections.append(current)
        section_counter += 1
        current = _new_section(section_counter, title=None)

    in_code = False
    table_buffer: List[str] = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            # Section boundary only for blank lines if there is content; otherwise ignore
            if current.content:
                # end any pending table buffer
                if table_buffer:
                    current.content.append(ContentTable(markdown="\n".join(table_buffer)))
                    table_buffer = []
            continue

        # Code fences
        if _is_code_fence(line):
            in_code = not in_code
            if not in_code:
                # closing fence ends code block; ensure not to include fence markers
                pass
            else:
                # opening fence; start a code block
                current.content.append(ContentCode(markdown=line.strip()))
            continue

        if in_code:
            # append lines to last code block markdown
            if current.content and isinstance(current.content[-1], ContentCode):
                cb = current.content[-1]
                cb.markdown = cb.markdown + "\n" + line
            else:
                current.content.append(ContentCode(markdown=line))
            continue

        # Headings
        if _is_heading(line):
            # Flush table buffer
            if table_buffer:
                current.content.append(ContentTable(markdown="\n".join(table_buffer)))
                table_buffer = []
            _commit_section()
            current.title = _clean_line(line).rstrip(":")
            continue

        # Tables (markdown style)
        if _is_table_row(line) or _is_table_divider(line):
            table_buffer.append(line)
            continue

        # Lists
        if _is_list_item(line):
            # If previous block is a list, append; else start a new list
            cleaned = _clean_line(line)
            if current.content and isinstance(current.content[-1], ContentList):
                lst = current.content[-1]
                lst.items.append(cleaned)
            else:
                current.content.append(ContentList(items=[cleaned]))
            continue

        # Math blocks
        math_parts = _extract_math_blocks(line)
        if math_parts:
            for m in math_parts:
                current.content.append(ContentMath(markdown=m))
            # Also store remaining text (without math markers) as paragraph if any
            cleaned = _clean_line(re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$", "", line))
            if cleaned:
                if current.content and isinstance(current.content[-1], ContentParagraph) and len(cleaned) < 60:
                    prev = current.content[-1]
                    prev.text = (prev.text + " " + cleaned).strip()
                else:
                    current.content.append(ContentParagraph(text=cleaned))
            continue

        # Default: paragraph
        cleaned = _clean_line(line)
        if current.content and isinstance(current.content[-1], ContentParagraph) and len(cleaned) < 60:
            prev = current.content[-1]
            prev.text = (prev.text + " " + cleaned).strip()
        else:
            current.content.append(ContentParagraph(text=cleaned))

    # Finalize pending table
    if table_buffer:
        current.content.append(ContentTable(markdown="\n".join(table_buffer)))
        table_buffer = []

    _commit_section()

    md = DocumentMetadata(
        source_type=source_type,
        language=_detect_language(norm),
        page_count=page_count,
        confidence=ocr_confidence,
        ocr_method=ocr_method,
        line_count=len(lines),
        word_count=_word_count(norm),
    )
    return StructuredDocument(document_metadata=md, sections=sections, raw_text=text)


def flatten_structured_text(doc: StructuredDocument) -> str:
    """Flatten the structured document to plain text for downstream models."""
    parts: List[str] = []
    for sec in doc.sections:
        if sec.title:
            parts.append(sec.title)
        for block in sec.content:
            if isinstance(block, ContentParagraph):
                parts.append(block.text)
            elif isinstance(block, ContentList):
                parts.extend([f"- {it}" for it in block.items])
            elif isinstance(block, (ContentTable, ContentMath, ContentCode)):
                parts.append(block.markdown)
    return "\n".join(parts)
