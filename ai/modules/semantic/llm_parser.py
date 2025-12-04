"""LLM-based semantic parser for academic content.

Provides:
- Pydantic models describing structured academic content
- LLMParser singleton wrapping OpenAI calls with retries
- parse_academic_content convenience function

Custom exceptions: LLMParserError, LLMAPIError, LLMValidationError, LLMTimeoutError
"""
from __future__ import annotations

import os
import time
import json
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ValidationError
from modules.utils import get_logger, log_llm_call

LOG = get_logger()

# tenacity for retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# OpenAI
import openai
from openai import error as openai_error

# Exceptions
class LLMParserError(Exception):
    pass

class LLMAPIError(LLMParserError):
    pass

class LLMValidationError(LLMParserError):
    pass

class LLMTimeoutError(LLMParserError):
    pass

# Pydantic models for structured output
class Question(BaseModel):
    question: str
    answer: str
    difficulty: str = Field(..., description='easy|medium|hard')

class Formula(BaseModel):
    latex: str
    description: Optional[str] = None
    variables: List[str] = []

class Concept(BaseModel):
    name: str
    definition: str
    examples: List[str] = []

class Subtopic(BaseModel):
    name: str
    content: str
    key_points: List[str] = []

class Topic(BaseModel):
    name: str
    description: str
    subtopics: List[Subtopic] = []

class Metadata(BaseModel):
    word_count: int
    estimated_reading_time_minutes: float
    academic_level: str = Field(..., description='high school|undergraduate|graduate')


class ParsedContent(BaseModel):
    topics: List[Topic] = []
    formulas: List[Formula] = []
    key_concepts: List[Concept] = []
    questions: List[Question] = []
    metadata: Metadata

# Helper constants from env
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
OPENAI_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', '60'))
OPENAI_RETRY_ATTEMPTS = int(os.getenv('OPENAI_RETRY_ATTEMPTS', '3'))
OPENAI_RETRY_MULTIPLIER = int(os.getenv('OPENAI_RETRY_MULTIPLIER', '2'))
OPENAI_RETRY_MAX_WAIT = int(os.getenv('OPENAI_RETRY_MAX_WAIT', '10'))
LLM_PARSER_MAX_TEXT_LENGTH = int(os.getenv('LLM_PARSER_MAX_TEXT_LENGTH', '50000'))
LLM_PARSER_ENABLE_COST_TRACKING = os.getenv('LLM_PARSER_ENABLE_COST_TRACKING', 'true').lower() in ('1','true','yes')

# Singleton parser
class LLMParser:
    _instance = None

    def __init__(self):
        # prevent direct instantiation
        self.model = OPENAI_MODEL
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise LLMParserError('OPENAI_API_KEY not set')
        openai.api_key = key
        self.timeout = OPENAI_TIMEOUT
        LOG.info('LLMParser initialized', extra={'model': self.model})

    @classmethod
    def get_instance(cls) -> 'LLMParser':
        if cls._instance is None:
            cls._instance = LLMParser()
        return cls._instance

    def _build_system_prompt(self) -> str:
        return (
            "You are an academic content analyst. Given raw OCR text from handwritten or typed notes, "
            "extract structured academic content in JSON. Preserve LaTeX formulas exactly, identify topics and subtopics, "
            "extract formulas with descriptions and variables, key concepts with concise definitions and examples, "
            "and produce comprehension questions with answers and difficulty levels (easy/medium/hard). "
            "Treat bullet or numbered lists as sources of key points or subtopics: convert list items into `key_points` or subtopic content where appropriate. "
            "When the text references diagrams or figures (e.g., 'see diagram above'), infer a brief textual description and associate it with the nearest topic or subtopic under a field such as an additional key point or short `content` sentence. "
            "Output only valid JSON that conforms to the provided schema. Do not add extra commentary or explanation outside the JSON object."
        )

    def _build_user_prompt(self, text: str) -> str:
        return (
            "Analyze the following OCR-extracted text and produce JSON matching the ParsedContent schema. "
            "Include metadata fields exactly: word_count (integer), estimated_reading_time_minutes (float, minutes), academic_level (one of: high school, undergraduate, graduate).\n\n"
            "Treat lists as key_points or subtopics and infer brief diagram descriptions when diagrams are referenced. "
            f"Text:\n{text}"
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        # crude estimation based on model
        # prices per 1000 tokens (approx): gpt-4 prompt 0.03, completion 0.06; gpt-3.5-turbo prompt 0.0015, completion 0.002
        if 'gpt-4' in model:
            return (prompt_tokens / 1000.0) * 0.03 + (completion_tokens / 1000.0) * 0.06
        else:
            return (prompt_tokens / 1000.0) * 0.0015 + (completion_tokens / 1000.0) * 0.002

    @retry(stop=stop_after_attempt(OPENAI_RETRY_ATTEMPTS), wait=wait_exponential(multiplier=OPENAI_RETRY_MULTIPLIER, max=OPENAI_RETRY_MAX_WAIT), retry=retry_if_exception_type((LLMAPIError, LLMTimeoutError)))
    def _call_openai(self, messages: List[Dict[str, str]], function_def: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        start = time.time()
        try:
            # Use function-calling to request structured JSON output matching ParsedContent
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
                timeout=self.timeout,
                functions=[function_def],
                function_call={"name": function_def.get('name')},
            )
            duration_ms = int((time.time() - start) * 1000)
            usage = resp.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            # log llm call
            if request_id:
                log_llm_call(request_id, self.model, prompt_tokens, completion_tokens, duration_ms, cost=self._estimate_cost(prompt_tokens, completion_tokens, self.model) if LLM_PARSER_ENABLE_COST_TRACKING else None)
            return resp
        except openai_error.Timeout as e:
            LOG.exception('openai_timeout', exc_info=True)
            raise LLMTimeoutError(str(e))
        except openai_error.OpenAIError as e:
            LOG.exception('openai_api_error', exc_info=True)
            raise LLMAPIError(str(e))
        except Exception as e:
            # Do not wrap LLMAPIError/LLMTimeoutError here; allow them to propagate to trigger retry
            if isinstance(e, (LLMAPIError, LLMTimeoutError)):
                raise
            LOG.exception('openai_unknown_error', exc_info=True)
            raise LLMAPIError(str(e))

    def parse(self, text: str, request_id: Optional[str] = None) -> ParsedContent:
        if not text or not text.strip():
            raise LLMParserError('Empty text')
        if len(text) > LLM_PARSER_MAX_TEXT_LENGTH:
            raise LLMParserError(f'Text too long ({len(text)} > {LLM_PARSER_MAX_TEXT_LENGTH})')
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        # Build function schema from ParsedContent JSON Schema
        pcs = ParsedContent.model_json_schema(ref_template='#/definitions/{model}')
        function_def = {
            'name': 'parsed_content',
            'description': 'Return the parsed academic content matching the ParsedContent schema as JSON',
            'parameters': pcs,
        }
        try:
            resp = self._call_openai(messages, function_def=function_def, request_id=request_id)
            # extract content
            choices = resp.get('choices', [])
            if not choices:
                raise LLMAPIError('No choices returned')
            message = choices[0].get('message', {})
            # When using function calling, parse the function_call arguments
            func_call = message.get('function_call')
            if not func_call:
                # fallback: attempt to read content and parse JSON
                content = message.get('content')
                try:
                    parsed_json = json.loads(content)
                except Exception:
                    raise LLMValidationError('Response did not include function_call and is not valid JSON')
            else:
                args_str = func_call.get('arguments')
                try:
                    parsed_json = json.loads(args_str)
                except Exception:
                    raise LLMValidationError('Function call arguments are not valid JSON')
            # validate with Pydantic
            try:
                pc = ParsedContent.model_validate(parsed_json)
            except ValidationError as e:
                LOG.exception('parsed_content_validation_failed', exc_info=True)
                raise LLMValidationError(str(e))
            return pc
        except LLMTimeoutError:
            raise
        except LLMAPIError:
            raise
        except LLMValidationError:
            raise
        except Exception as e:
            LOG.exception('llm_parser_unknown', exc_info=True)
            raise LLMParserError(str(e))

# Convenience function
def parse_academic_content(text: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    parser = LLMParser.get_instance()
    pc = parser.parse(text, request_id=request_id)
    return pc.model_dump()
