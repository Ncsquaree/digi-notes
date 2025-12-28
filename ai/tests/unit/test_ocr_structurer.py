import pytest

from modules.ocr import structure_document, flatten_structured_text, StructuredDocument


def test_structured_schema_includes_metadata_sections_and_raw_text():
    text = """
    INTRODUCTION
    This is a paragraph.

    Items:
    - First item
    - Second item

    Table:
    | Col A | Col B |
    | ----- | ----- |
    |  1    |   2   |

    Math: $E=mc^2$ and $$\\int_0^1 x^2 dx = 1/3$$
    """.strip()

    sd: StructuredDocument = structure_document(text, source_type='text', page_count=None, ocr_confidence=0.9)
    # Top-level fields
    assert sd.raw_text == text
    assert sd.document_metadata is not None
    assert isinstance(sd.document_metadata.word_count, int)
    assert isinstance(sd.document_metadata.line_count, int)
    # Sections
    assert len(sd.sections) >= 1
    assert sd.sections[0].section_id.startswith('sec-')
    # Content blocks include paragraph, list, table, math
    types = []
    for s in sd.sections:
        types.extend([c.type for c in s.content])
    assert 'paragraph' in types
    assert 'list' in types
    assert 'table' in types
    assert 'math' in types

    flat = flatten_structured_text(sd)
    assert isinstance(flat, str) and len(flat) > 0


def test_no_shared_state_between_runs():
    text1 = "Section A\n- item 1"
    text2 = "Section B\n- item 2"
    sd1: StructuredDocument = structure_document(text1)
    sd2: StructuredDocument = structure_document(text2)
    # Ensure lists/sections are independent
    assert sd1.sections[0].title != sd2.sections[0].title


def test_llm_fallback_is_used_when_low_confidence(monkeypatch):
    # Monkeypatch the internal LLM structuring to return a minimal valid schema
    from modules.ocr.ocr_structurer import _llm_structure

    def fake_llm(text, schema):
        return {
            "document_metadata": {
                "source_type": "text",
                "language": "en",
                "page_count": None,
                "confidence": 0.2,
                "line_count": 2,
                "word_count": 3
            },
            "sections": [
                {
                    "section_id": "sec-1",
                    "title": "LLM Title",
                    "content": [
                        {"type": "paragraph", "text": "Hello world"}
                    ]
                }
            ],
            "raw_text": text
        }

    monkeypatch.setattr('modules.ocr.ocr_structurer._llm_structure', fake_llm)

    text = "Hello world"
    sd: StructuredDocument = structure_document(text, source_type='text', ocr_confidence=0.2, llm_fallback=True, llm_threshold=0.6)
    # Validated model should reflect the fake LLM output
    assert sd.sections[0].title == "LLM Title"
    assert sd.sections[0].content[0].type == 'paragraph'
    assert sd.document_metadata.confidence == 0.2


def test_sample_image_like_text_to_json_contract():
    # Simulate a small OCR text with headings, list, and table
    text = """
    CHEMISTRY
    Periodic Table Overview
    | Element | Symbol |
    | ------- | ------ |
    | Oxygen  |   O    |

    Key Points:
    - Atomic number defines element identity
    - Periods are horizontal rows
    """.strip()

    sd: StructuredDocument = structure_document(text, source_type='image', page_count=1, ocr_confidence=0.85)
    assert sd.document_metadata.source_type == 'image'
    assert sd.document_metadata.page_count == 1
    # Contract checks
    assert isinstance(sd.raw_text, str)
    assert len(sd.sections) >= 1
    # Ensure at least one table and one list present
    types = []
    for s in sd.sections:
        types.extend([c.type for c in s.content])
    assert 'table' in types
    assert 'list' in types
