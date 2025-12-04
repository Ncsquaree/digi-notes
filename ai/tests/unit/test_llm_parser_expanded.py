import json
import pytest
from modules.semantic.llm_parser import LLMParser, ParsedContent, LLMParserError, LLMValidationError


def test_parser_function_call_path(monkeypatch, mock_openai_client):
    parser = LLMParser.get_instance()
    text = 'Photosynthesis is...'  # short text
    pc = parser.parse(text)
    assert isinstance(pc, ParsedContent)


def test_parser_empty_text():
    p = LLMParser.get_instance()
    with pytest.raises(LLMParserError):
        p.parse('  ')
