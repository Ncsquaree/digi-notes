from __future__ import annotations

import os
import time
import json
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from modules.utils import get_logger, log_llm_call, log_summarization

LOG = get_logger()

# tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import openai
from openai import error as openai_error

from .cache_manager import CacheManager


class SummarizerError(Exception):
    pass


class SummarizerAPIError(SummarizerError):
    pass


class SummarizerTimeoutError(SummarizerError):
    pass


class SummaryMode(str, Enum):
    BRIEF = 'brief'
    DETAILED = 'detailed'
    BOTH = 'both'


class SummaryResult(BaseModel):
    brief: Optional[str] = None
    detailed: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Config
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
SUMMARIZER_BRIEF_MAX_TOKENS = int(os.getenv('SUMMARIZER_BRIEF_MAX_TOKENS', '500'))
SUMMARIZER_DETAILED_MAX_TOKENS = int(os.getenv('SUMMARIZER_DETAILED_MAX_TOKENS', '1500'))
SUMMARIZER_TEMPERATURE = float(os.getenv('SUMMARIZER_TEMPERATURE', '0.3'))
OPENAI_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', '60'))
OPENAI_RETRY_ATTEMPTS = int(os.getenv('OPENAI_RETRY_ATTEMPTS', '3'))
OPENAI_RETRY_MULTIPLIER = int(os.getenv('OPENAI_RETRY_MULTIPLIER', '2'))
OPENAI_RETRY_MAX_WAIT = int(os.getenv('OPENAI_RETRY_MAX_WAIT', '10'))


class Summarizer:
    _instance = None

    def __init__(self):
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise SummarizerError('OPENAI_API_KEY not set')
        openai.api_key = key
        self.model = OPENAI_MODEL
        self.timeout = OPENAI_TIMEOUT
        LOG.info('Summarizer initialized', extra={'model': self.model})

    @classmethod
    def get_instance(cls) -> 'Summarizer':
        if cls._instance is None:
            cls._instance = Summarizer()
        return cls._instance

    def _build_brief_prompt(self, parsed: Dict[str, Any]) -> str:
        # Build a concise prompt preserving LaTeX
        return (
            "You are an academic summarizer. Produce a concise 2-3 sentence summary of the provided parsed academic content. "
            "Preserve LaTeX formulas exactly (e.g., $E=mc^2$). Highlight the main concepts and their relationships. "
            "Do not add new concepts. Respond as plain text.") + "\n\nContent:" + json.dumps(parsed)

    def _build_detailed_prompt(self, parsed: Dict[str, Any]) -> str:
        return (
            "You are an academic summarizer. Produce a comprehensive, structured paragraph summary preserving all formulas in LaTeX, including examples and key points from subtopics. "
            "Keep an academic tone appropriate for study notes. Respond as plain text.") + "\n\nContent:" + json.dumps(parsed)

    @retry(stop=stop_after_attempt(OPENAI_RETRY_ATTEMPTS), wait=wait_exponential(multiplier=OPENAI_RETRY_MULTIPLIER, max=OPENAI_RETRY_MAX_WAIT), retry=retry_if_exception_type((SummarizerAPIError, SummarizerTimeoutError)))
    def _call_openai(self, messages, max_tokens: int, request_id: Optional[str] = None) -> Dict[str, Any]:
        start = time.time()
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=SUMMARIZER_TEMPERATURE,
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
            LOG.exception('summarizer_timeout', exc_info=True)
            raise SummarizerTimeoutError(str(e))
        except openai_error.OpenAIError as e:
            LOG.exception('summarizer_api_error', exc_info=True)
            raise SummarizerAPIError(str(e))
        except Exception as e:
            LOG.exception('summarizer_unknown_error', exc_info=True)
            raise SummarizerAPIError(str(e))

    def summarize(self, parsed_content: Dict[str, Any], mode: SummaryMode = SummaryMode.BOTH, request_id: Optional[str] = None) -> SummaryResult:
        if not isinstance(parsed_content, dict):
            raise SummarizerError('parsed_content must be a dict')

        # Normalize mode argument: accept either SummaryMode or str; reject invalid strings
        try:
            if isinstance(mode, str):
                mode_enum = SummaryMode(mode)
            elif isinstance(mode, SummaryMode):
                mode_enum = mode
            else:
                # attempt to coerce other values (e.g., passed enum-like)
                mode_enum = SummaryMode(mode)
        except Exception as e:
            raise SummarizerError(f'invalid summarization mode: {mode}') from e

        cache = CacheManager.get_instance()
        mode_value = mode_enum.value
        cached = cache.get_summary(parsed_content, mode_value)
        if cached:
            res = SummaryResult(**cached)
            # tag cache hit in metadata
            res.metadata = res.metadata or {}
            res.metadata['cache_hit'] = True
            # compute brief/detailed word counts for observability
            brief_wc = len(res.brief.split()) if res.brief else 0
            detailed_wc = len(res.detailed.split()) if res.detailed else 0
            # structured single-event logging for summarization (cache hit)
            try:
                log_summarization(request_id or '', mode_value, brief_wc, detailed_wc, 0, cache_hit=True)
            except Exception:
                LOG.exception('log_summarization_failed_on_cache_hit', exc_info=True)
            LOG.info('summarization_cache_hit', extra={'request_id': request_id, 'mode': mode_value})
            return res

        start = time.time()
        brief_text = None
        detailed_text = None

        try:
            if mode_enum in (SummaryMode.BRIEF, SummaryMode.BOTH):
                prompt = self._build_brief_prompt(parsed_content)
                messages = [{'role': 'system', 'content': 'You produce concise academic summaries.'}, {'role': 'user', 'content': prompt}]
                resp = self._call_openai(messages, max_tokens=SUMMARIZER_BRIEF_MAX_TOKENS, request_id=request_id)
                choices = resp.get('choices', [])
                brief_text = choices[0].get('message', {}).get('content') if choices else None

            if mode_enum in (SummaryMode.DETAILED, SummaryMode.BOTH):
                prompt = self._build_detailed_prompt(parsed_content)
                messages = [{'role': 'system', 'content': 'You produce detailed academic summaries.'}, {'role': 'user', 'content': prompt}]
                resp = self._call_openai(messages, max_tokens=SUMMARIZER_DETAILED_MAX_TOKENS, request_id=request_id)
                choices = resp.get('choices', [])
                detailed_text = choices[0].get('message', {}).get('content') if choices else None

            duration_ms = int((time.time() - start) * 1000)
            metadata = {
                'processing_time_ms': duration_ms,
                'mode': mode_value,
                'model_used': self.model,
                'cache_hit': False,
            }
            brief_wc = len(brief_text.split()) if brief_text else 0
            detailed_wc = len(detailed_text.split()) if detailed_text else 0
            # estimate cost not implemented here
            result = SummaryResult(brief=brief_text, detailed=detailed_text, metadata=metadata)
            # store in cache
            try:
                cache.set_summary(parsed_content, mode_value, result.model_dump(), ttl=None)
            except Exception:
                LOG.warning('cache_set_failed')

            # log summarization
            log_summarization(request_id or '', mode_value, brief_wc, detailed_wc, duration_ms, cache_hit=False)
            return result
        except SummarizerTimeoutError:
            raise
        except SummarizerAPIError:
            raise
        except Exception as e:
            LOG.exception('summarization_failed', exc_info=True)
            raise SummarizerError(str(e))


def generate_summary(parsed_content: Dict[str, Any], mode: str = 'both', request_id: Optional[str] = None) -> Dict[str, Any]:
    sm = Summarizer.get_instance()
    try:
        mode_enum = SummaryMode(mode) if isinstance(mode, str) else mode
    except Exception:
        mode_enum = SummaryMode.BOTH
    res = sm.summarize(parsed_content, mode=mode_enum, request_id=request_id)
    return res.model_dump()
