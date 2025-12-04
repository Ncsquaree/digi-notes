from __future__ import annotations

import os
import time
import json
import random
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from modules.utils import get_logger, log_llm_call, log_flashcard_generation

LOG = get_logger()

# tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import openai
from openai import error as openai_error

# Import ParsedContent type if available
try:
    from modules.semantic import ParsedContent
except Exception:
    ParsedContent = dict


class FlashcardGeneratorError(Exception):
    pass


class FlashcardAPIError(FlashcardGeneratorError):
    pass


class FlashcardValidationError(FlashcardGeneratorError):
    pass


class FlashcardTimeoutError(FlashcardGeneratorError):
    pass


class FlashcardItem(BaseModel):
    question: str
    answer: str
    difficulty: int = Field(0, ge=0, le=5)
    context: Optional[str] = None
    source_type: str = Field('generated')


class GenerateFlashcardsRequest(BaseModel):
    parsed_content: ParsedContent
    count: Optional[int] = None
    min_difficulty: Optional[int] = Field(0, ge=0, le=5)
    max_difficulty: Optional[int] = Field(5, ge=0, le=5)
    include_formulas: bool = True
    include_concepts: bool = True
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GenerateFlashcardsResponse(BaseModel):
    flashcards: List[FlashcardItem]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Config
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
FLASHCARD_MAX_TOKENS = int(os.getenv('FLASHCARD_MAX_TOKENS', '1500'))
FLASHCARD_TEMPERATURE = float(os.getenv('FLASHCARD_TEMPERATURE', '0.5'))
OPENAI_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', '60'))
OPENAI_RETRY_ATTEMPTS = int(os.getenv('OPENAI_RETRY_ATTEMPTS', '3'))
OPENAI_RETRY_MULTIPLIER = int(os.getenv('OPENAI_RETRY_MULTIPLIER', '2'))
OPENAI_RETRY_MAX_WAIT = int(os.getenv('OPENAI_RETRY_MAX_WAIT', '10'))
FLASHCARD_MIN_COUNT = int(os.getenv('FLASHCARD_MIN_COUNT', '5'))
FLASHCARD_MAX_COUNT = int(os.getenv('FLASHCARD_MAX_COUNT', '50'))


class FlashcardGenerator:
    _instance = None

    def __init__(self):
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise FlashcardGeneratorError('OPENAI_API_KEY not set')
        openai.api_key = key
        self.model = OPENAI_MODEL
        self.timeout = OPENAI_TIMEOUT
        LOG.info('FlashcardGenerator initialized', extra={'model': self.model})

    @classmethod
    def get_instance(cls) -> 'FlashcardGenerator':
        if cls._instance is None:
            cls._instance = FlashcardGenerator()
        return cls._instance

    def _map_text_difficulty(self, txt: str) -> int:
        # Accept numeric input (int or numeric string) and clamp to 0-5
        if txt is None or txt == '':
            return 0
        # if it's already an int
        try:
            if isinstance(txt, int):
                return max(0, min(5, int(txt)))
            # numeric string
            s = str(txt).strip()
            if s.isdigit():
                v = int(s)
                return max(0, min(5, v))
        except Exception:
            pass

        t = str(txt).lower().strip()
        if 'easy' in t:
            return 1
        if 'hard' in t:
            return 5
        if 'medium' in t or 'moderate' in t:
            return 3
        # fallback heuristic: short questions -> easier
        return 1 if len(t.split()) < 8 else 3

    def _build_generation_prompt(self, parsed: Dict[str, Any], target_count: int, options: Dict[str, Any]) -> str:
        # Instruction prompt to ask the LLM to produce JSON array of QA items
        include_formulas = bool(options.get('include_formulas', True))
        include_concepts = bool(options.get('include_concepts', True))
        instruction_lines = [
            'You are an academic assistant. Given parsed academic content (topics, subtopics, concepts, formulas),',
            f'generate up to {target_count} high-quality question-answer flashcards.',
            'Preserve LaTeX formulas exactly and return a JSON array under the key "flashcards" where each item has "question", "answer", "difficulty" (easy|medium|hard), and optional "context".',
            'Do not invent unrelated facts. Prioritize conceptual and formula understanding.'
        ]
        if include_formulas and not include_concepts:
            instruction_lines.append('Include formula-based questions where relevant; avoid concept-only questions.')
        elif include_concepts and not include_formulas:
            instruction_lines.append('Include concept-based questions and avoid formula-only questions.')
        # otherwise include both (default)

        prompt = {
            'instruction': ' '.join(instruction_lines),
            'parsed_content': parsed,
            'constraints': {
                'max_questions': target_count,
                'output_format': 'JSON',
                'include_formulas': include_formulas,
                'include_concepts': include_concepts,
            },
        }
        # return as string to include in LLM messages
        return json.dumps(prompt)

    @retry(stop=stop_after_attempt(OPENAI_RETRY_ATTEMPTS), wait=wait_exponential(multiplier=OPENAI_RETRY_MULTIPLIER, max=OPENAI_RETRY_MAX_WAIT), retry=retry_if_exception_type((FlashcardAPIError, FlashcardTimeoutError)))
    def _call_openai(self, messages: List[Dict[str, Any]], max_tokens: int, request_id: Optional[str] = None) -> Dict[str, Any]:
        start = time.time()
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=FLASHCARD_TEMPERATURE,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            duration_ms = int((time.time() - start) * 1000)
            usage = resp.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            if request_id:
                log_llm_call(request_id, self.model, prompt_tokens, completion_tokens, duration_ms)
            return resp
        except openai_error.Timeout as e:
            LOG.exception('flashcard_timeout', exc_info=True)
            raise FlashcardTimeoutError(str(e))
        except openai_error.OpenAIError as e:
            LOG.exception('flashcard_api_error', exc_info=True)
            raise FlashcardAPIError(str(e))
        except Exception as e:
            LOG.exception('flashcard_unknown_error', exc_info=True)
            raise FlashcardAPIError(str(e))

    def _extract_from_parsed_questions(self, parsed: Dict[str, Any]) -> List[FlashcardItem]:
        out: List[FlashcardItem] = []
        try:
            questions = parsed.get('questions') or []
            for q in questions:
                question = q.get('question') or q.get('q') or ''
                answer = q.get('answer') or q.get('a') or ''
                diff = q.get('difficulty') or q.get('difficulty_label') or ''
                context = q.get('context') or q.get('topic')
                item = FlashcardItem(question=question, answer=answer, difficulty=self._map_text_difficulty(diff), context=context, source_type='parsed_question')
                out.append(item)
        except Exception:
            LOG.exception('extract_from_parsed_questions_failed', exc_info=True)
        return out

    def _generate_additional_flashcards(self, parsed: Dict[str, Any], target_count: int, options: Dict[str, Any] = None, request_id: Optional[str] = None) -> List[FlashcardItem]:
        options = options or {}
        prompt = self._build_generation_prompt(parsed, target_count, options)
        messages = [
            {'role': 'system', 'content': 'You generate study flashcards.'},
            {'role': 'user', 'content': prompt}
        ]
        resp = self._call_openai(messages, max_tokens=FLASHCARD_MAX_TOKENS, request_id=request_id)
        choices = resp.get('choices', [])
        if not choices:
            return []
        text = choices[0].get('message', {}).get('content') if choices else None
        if not text:
            return []
        # try parse JSON from model
        try:
            parsed_resp = json.loads(text)
            cards = parsed_resp.get('flashcards') or []
        except Exception:
            # fallback: try to extract arrays from text
            try:
                parsed_resp = json.loads(text[text.find('{'):])
                cards = parsed_resp.get('flashcards') or []
            except Exception:
                LOG.exception('flashcard_generation_parse_failed', exc_info=True)
                raise FlashcardValidationError('Could not parse flashcard generation output')

        out: List[FlashcardItem] = []
        for c in cards:
            q = c.get('question')
            a = c.get('answer')
            d_label = c.get('difficulty', '')
            ctx = c.get('context')
            if not q or not a:
                continue
            try:
                # allow numeric difficulties directly
                if isinstance(d_label, int) or (isinstance(d_label, str) and str(d_label).strip().isdigit()):
                    diff = int(d_label)
                else:
                    diff = self._map_text_difficulty(d_label if isinstance(d_label, str) else '')
                diff = max(0, min(5, int(diff)))
            except Exception:
                diff = 0
            out.append(FlashcardItem(question=q, answer=a, difficulty=diff, context=ctx, source_type='generated'))
        return out

    def generate(self, parsed_content: Dict[str, Any], count: Optional[int] = None, options: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None) -> GenerateFlashcardsResponse:
        if not isinstance(parsed_content, dict):
            raise FlashcardValidationError('parsed_content must be a dict')
        options = options or {}
        start = time.time()
        extracted = self._extract_from_parsed_questions(parsed_content)
        generated: List[FlashcardItem] = []
        total_needed = count or max(FLASHCARD_MIN_COUNT, len(extracted))
        total_needed = min(total_needed, FLASHCARD_MAX_COUNT)
        if len(extracted) < total_needed:
            try:
                generated = self._generate_additional_flashcards(parsed_content, total_needed - len(extracted), options=options, request_id=request_id)
            except FlashcardGeneratorError:
                LOG.exception('generate_additional_flashcards_failed', exc_info=True)

        all_cards = extracted + generated
        # filter by difficulty range
        min_d = int(options.get('min_difficulty', 0))
        max_d = int(options.get('max_difficulty', 5))
        filtered = [c for c in all_cards if min_d <= int(c.difficulty) <= max_d]
        # shuffle and limit
        random.shuffle(filtered)
        final = filtered[:total_needed] if count else filtered
        duration_ms = int((time.time() - start) * 1000)
        # count sources
        source_counts = {
            'parsed_questions': len([c for c in final if c.source_type == 'parsed_question']),
            'generated': len([c for c in final if c.source_type == 'generated']),
            'concepts': 0,
            'formulas': 0,
        }
        # structured logging
        try:
            log_flashcard_generation(request_id or '', len(final), source_counts, duration_ms, cache_hit=False)
        except Exception:
            LOG.exception('log_flashcard_generation_failed', exc_info=True)

        metadata = {'processing_time_ms': duration_ms, 'model_used': self.model, 'source_counts': source_counts, 'flashcard_count': len(final)}
        return GenerateFlashcardsResponse(flashcards=final, metadata=metadata)


def generate_flashcards(parsed_content: Dict[str, Any], count: Optional[int] = None, options: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    gen = FlashcardGenerator.get_instance()
    resp = gen.generate(parsed_content, count=count, options=options, request_id=request_id)
    return resp.model_dump()
