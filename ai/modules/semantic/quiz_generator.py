import os
import time
import json
from typing import List, Optional, Dict, Any
from enum import Enum

import tenacity
from pydantic import BaseModel, Field, validator

from modules.utils import get_logger, log_llm_call
from .llm_parser import OpenAIError

LOG = get_logger()

# Exceptions
class QuizGeneratorError(Exception):
    pass

class QuizAPIError(QuizGeneratorError):
    pass

class QuizValidationError(QuizGeneratorError):
    pass

class QuizTimeoutError(QuizGeneratorError):
    pass

# Models
class QuestionType(str, Enum):
    MCQ = 'mcq'
    TRUE_FALSE = 'true_false'
    SHORT_ANSWER = 'short_answer'


class MCQOption(BaseModel):
    option: str
    is_correct: bool = False


class QuizQuestion(BaseModel):
    question: str
    type: QuestionType
    options: Optional[List[MCQOption]] = None
    correct_answer: str
    explanation: Optional[str] = None
    difficulty: str = Field('medium')
    points: int = Field(1)

    @validator('difficulty')
    def clamp_difficulty(cls, v):
        if v not in ('easy', 'medium', 'hard'):
            return 'medium'
        return v


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    total_points: int
    metadata: Dict[str, Any]


# Env
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
QUIZ_MAX_TOKENS = int(os.getenv('QUIZ_MAX_TOKENS', '2000'))
QUIZ_TEMPERATURE = float(os.getenv('QUIZ_TEMPERATURE', '0.5'))
QUIZ_MAX_QUESTIONS = int(os.getenv('QUIZ_MAX_QUESTIONS', '20'))
OPENAI_TIMEOUT = float(os.getenv('OPENAI_TIMEOUT', '30'))
OPENAI_RETRY_ATTEMPTS = int(os.getenv('OPENAI_RETRY_ATTEMPTS', '3'))


# Singleton
class QuizGenerator:
    _instance = None

    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise QuizGeneratorError('OPENAI_API_KEY not set')
        self.model = OPENAI_MODEL
        self.timeout = OPENAI_TIMEOUT
        self.temperature = QUIZ_TEMPERATURE
        LOG.info('QuizGenerator initialized', extra={'model': self.model})

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = QuizGenerator()
        return cls._instance

    def _build_system_prompt(self) -> str:
        return (
            "You are an assistant that generates high-quality academic quiz questions from structured parsed content. "
            "Return a JSON object matching the requested schema. Preserve LaTeX formulas. For MCQs produce exactly 4 options, one correct. "
            "Include a short explanation and a difficulty (easy|medium|hard) and points for each question."
        )

    def _build_user_prompt(self, parsed_content: dict, question_count: int, question_types: Optional[List[str]]):
        types = question_types or ['mcq', 'true_false', 'short_answer']
        return json.dumps({
            'parsed_content': parsed_content,
            'question_count': question_count,
            'question_types': types,
        })

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # crude estimation using gpt-4/gpt-3.5 pricing assumptions
        # Defaults chosen to be conservative; exact pricing may vary.
        if 'gpt-4' in self.model:
            return (prompt_tokens + completion_tokens) / 1000.0 * 0.06
        return (prompt_tokens + completion_tokens) / 1000.0 * 0.002

    @tenacity.retry(stop=tenacity.stop_after_attempt(OPENAI_RETRY_ATTEMPTS),
                    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
                    reraise=True)
    def _call_openai(self, messages: List[dict], function_def: dict = None, request_id: str = None) -> dict:
        # Import locally to avoid hard dependency at module import time
        try:
            import openai
            from openai.error import OpenAIError, Timeout
        except Exception as e:
            raise QuizAPIError('OpenAI client not available') from e

        start = time.time()
        try:
            if function_def:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    functions=[function_def],
                    temperature=self.temperature,
                    max_tokens=QUIZ_MAX_TOKENS,
                    timeout=self.timeout,
                )
            else:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=QUIZ_MAX_TOKENS,
                    timeout=self.timeout,
                )
        except Timeout as e:
            LOG.exception('quiz_openai_timeout', exc_info=True)
            raise QuizTimeoutError(str(e)) from e
        except OpenAIError as e:
            LOG.exception('quiz_openai_error', exc_info=True)
            raise QuizAPIError(str(e)) from e
        duration = int((time.time() - start) * 1000)
        # normalize resp to plain dict to be resilient across SDK versions
        try:
            if not isinstance(resp, dict):
                if hasattr(resp, 'to_dict_recursive'):
                    resp = resp.to_dict_recursive()
                elif hasattr(resp, 'to_dict'):
                    resp = resp.to_dict()
                elif hasattr(resp, 'to_json'):
                    resp = json.loads(resp.to_json())
                else:
                    # fallback: try json serialization
                    resp = json.loads(json.dumps(resp, default=lambda o: getattr(o, '__dict__', str(o))))
        except Exception:
            # if normalization fails, keep original but wrap in dict where possible
            try:
                resp = dict(resp)
            except Exception:
                pass

        usage = resp.get('usage', {}) if isinstance(resp, dict) else {}
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        log_llm_call(request_id=request_id, duration_ms=duration, model=self.model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost)
        return resp

    def generate_quiz(self, parsed_content: dict, question_count: int = 10, question_types: Optional[List[str]] = None, options: Optional[dict] = None, request_id: str = None) -> QuizResponse:
        if question_count < 1 or question_count > QUIZ_MAX_QUESTIONS:
            raise QuizValidationError(f'question_count must be 1-{QUIZ_MAX_QUESTIONS}')
        if question_types:
            for t in question_types:
                if t not in (QuestionType.MCQ.value, QuestionType.TRUE_FALSE.value, QuestionType.SHORT_ANSWER.value):
                    raise QuizValidationError('Invalid question type: ' + str(t))
        messages = [
            {'role': 'system', 'content': self._build_system_prompt()},
            {'role': 'user', 'content': self._build_user_prompt(parsed_content, question_count, question_types)},
        ]
        # Provide a simple function schema to ask for structured JSON output
        function_def = {
            'name': 'quiz_response',
            'description': 'Return quiz as JSON matching schema',
            'parameters': {
                'type': 'object',
                'properties': {
                    'questions': {
                        'type': 'array',
                        'items': {'type': 'object'}
                    },
                    'total_points': {'type': 'integer'},
                    'metadata': {'type': 'object'}
                },
                'required': ['questions', 'total_points']
            }
        }
        # pass options through to prompt handling if present (currently ignored by LLM prompt builder)
        options = options or {}
        resp = self._call_openai(messages, function_def=function_def, request_id=request_id)
        # parse out function_call arguments if present
        try:
            # new OpenAI responses may include choices[0].message.function_call.arguments
            choices = resp.get('choices', []) if isinstance(resp, dict) else []
            if choices and 'message' in choices[0] and choices[0]['message'].get('function_call'):
                args_text = choices[0]['message']['function_call'].get('arguments', '{}')
            elif choices and 'message' in choices[0] and choices[0]['message'].get('content'):
                # fallback to content
                args_text = choices[0]['message'].get('content', '{}')
            else:
                args_text = '{}'
            data = json.loads(args_text)
        except Exception as e:
            LOG.exception('quiz_response_parse_failed', exc_info=True)
            raise QuizAPIError('Failed to parse model response') from e

        try:
            # Validate questions individually
            questions_raw = data.get('questions', [])
            questions = []
            total = 0
            for q in questions_raw:
                # Normalize type
                qtype = q.get('type')
                if qtype in ('mcq', 'MCQ'):
                    q['type'] = QuestionType.MCQ.value
                elif qtype in ('true_false', 'TRUE_FALSE', 'tf'):
                    q['type'] = QuestionType.TRUE_FALSE.value
                else:
                    q['type'] = QuestionType.SHORT_ANSWER.value
                qq = QuizQuestion(**q)
                questions.append(qq)
                total += qq.points
            metadata = data.get('metadata', {})
            # ensure metadata includes requested/requested types and effective count
            types_requested = question_types or [QuestionType.MCQ.value, QuestionType.TRUE_FALSE.value, QuestionType.SHORT_ANSWER.value]
            metadata = data.get('metadata', {}) or {}
            metadata.setdefault('question_count', len(questions))
            metadata.setdefault('types_requested', types_requested)
            metadata.setdefault('model_used', self.model)

            # validate produced questions count
            if len(questions) < 1:
                raise QuizValidationError('No valid questions generated by model')
            if len(questions) > QUIZ_MAX_QUESTIONS:
                LOG.warning('quiz_generated_exceeds_max', extra={'requested': question_count, 'generated': len(questions), 'max': QUIZ_MAX_QUESTIONS})
                questions = questions[:QUIZ_MAX_QUESTIONS]
            total = sum(q.points for q in questions)
            # construct QuizResponse
            qr = QuizResponse(questions=questions, total_points=total, metadata=metadata)
            return qr
        except QuizValidationError:
            raise
        except Exception as e:
            LOG.exception('quiz_validation_failed', exc_info=True)
            raise QuizValidationError(str(e)) from e

    def validate_answers(self, quiz_response: QuizResponse, user_answers: Dict[int, str]) -> Dict[str, Any]:
        """Validate user answers against a QuizResponse.

        Returns a dict with overall `score`, `max_score`, and `results` list with per-question details.
        """
        if not isinstance(quiz_response, QuizResponse):
            raise QuizValidationError('quiz_response must be a QuizResponse instance')

        results = []
        score = 0
        max_score = quiz_response.total_points
        for idx, q in enumerate(quiz_response.questions):
            user_ans = None
            if isinstance(user_answers, dict):
                user_ans = user_answers.get(idx)
            correct = False
            points_earned = 0

            try:
                if q.type == QuestionType.MCQ:
                    if user_ans is not None:
                        # compare normalized strings
                        ua = str(user_ans).strip().lower()
                        ca = str(q.correct_answer).strip().lower()
                        if ua == ca:
                            correct = True
                elif q.type == QuestionType.TRUE_FALSE:
                    if user_ans is not None:
                        ua = str(user_ans).strip().lower()
                        ca = str(q.correct_answer).strip().lower()
                        # allow t/f, true/false, y/n
                        synonyms = {'t': 'true', 'f': 'false', 'y': 'true', 'n': 'false'}
                        ua_norm = synonyms.get(ua, ua)
                        ca_norm = synonyms.get(ca, ca)
                        if ua_norm == ca_norm:
                            correct = True
                else:
                    # short answer: keyword or case-insensitive containment
                    if user_ans:
                        ua = str(user_ans).strip().lower()
                        ca = str(q.correct_answer).strip().lower()
                        # check if any keyword from correct_answer appears in user's answer
                        keywords = [k for k in ca.split() if len(k) > 2]
                        if any(kw in ua for kw in keywords):
                            correct = True
            except Exception:
                correct = False

            if correct:
                points_earned = q.points
                score += points_earned

            results.append({
                'question_index': idx,
                'correct': correct,
                'user_answer': user_ans,
                'correct_answer': q.correct_answer,
                'points_earned': points_earned,
                'points_available': q.points,
            })

        return {'score': score, 'max_score': max_score, 'results': results}


# convenience
def generate_quiz(parsed_content: dict, question_count: int = 10, question_types: Optional[List[str]] = None, options: Optional[dict] = None, request_id: str = None) -> Dict[str, Any]:
    g = QuizGenerator.get_instance()
    res = g.generate_quiz(parsed_content, question_count=question_count, question_types=question_types, options=options, request_id=request_id)
    return res.model_dump()


def validate_quiz_answers(quiz: Any, user_answers: Dict[int, str]) -> Dict[str, Any]:
    """Validate a user's answers against a quiz produced by the generator.

    `quiz` may be a `QuizResponse` instance or a dict matching the response shape.
    `user_answers` is a mapping from question index (0-based) to the user's answer string.
    """
    g = QuizGenerator.get_instance()
    # normalize to QuizResponse
    if isinstance(quiz, dict):
        qr = QuizResponse(**quiz)
    elif isinstance(quiz, QuizResponse):
        qr = quiz
    else:
        raise QuizValidationError('Invalid quiz object for validation')
    return g.validate_answers(qr, user_answers)

