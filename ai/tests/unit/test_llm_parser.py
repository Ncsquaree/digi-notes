import pytest

from modules.semantic.llm_parser import LLMParser, LLMParserError, ParsedContent, LLMValidationError, LLM_PARSER_MAX_TEXT_LENGTH
import openai


@pytest.mark.unit
def test_get_instance_singleton():
    a = LLMParser.get_instance()
    b = LLMParser.get_instance()
    assert a is b


@pytest.mark.unit
def test_parse_success(sample_parsed_content, mock_openai_client):
    parser = LLMParser.get_instance()
    parsed = parser.parse('Photosynthesis is the process...')
    assert isinstance(parsed, ParsedContent)
    assert parsed.topics is not None


@pytest.mark.unit
def test_parse_empty_text_raises():
    parser = LLMParser.get_instance()
    with pytest.raises(LLMParserError):
        parser.parse('')


@pytest.mark.unit
def test_parse_too_long_text_raises():
    parser = LLMParser.get_instance()
    long_text = 'x' * (LLM_PARSER_MAX_TEXT_LENGTH + 1)
    with pytest.raises(LLMParserError):
        parser.parse(long_text)


@pytest.mark.unit
def test_parse_invalid_json_raises(monkeypatch):
    # simulate OpenAI returning a function_call with invalid JSON
    def fake_create(*args, **kwargs):
        return {'choices': [{'message': {'function_call': {'name': 'parsed_content', 'arguments': '{invalid json'}}}]}, 'usage': {}}

    monkeypatch.setattr(openai.ChatCompletion, 'create', staticmethod(fake_create))
    parser = LLMParser.get_instance()
    with pytest.raises(LLMValidationError):
        parser.parse('Some text that triggers invalid json')
