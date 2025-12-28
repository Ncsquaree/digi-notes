import os
import sys
import logging
import pathlib
import contextvars
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

_request_ctx_var = contextvars.ContextVar('request_ctx', default={})


def set_request_context(request_id: str, user_id: str = None):
    _request_ctx_var.set({'request_id': request_id, 'user_id': user_id})


def get_request_context():
    return _request_ctx_var.get()


def _inject_request_context(record):
    ctx = get_request_context()
    record.request_id = ctx.get('request_id')
    record.user_id = ctx.get('user_id')
    return True


def get_logger(name: str = 'ai_service'):
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')
    # default to a relative logs directory so local dev doesn't require /app
    LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'logs')
    LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', str(10 * 1024 * 1024)))
    LOG_MAX_FILES = int(os.getenv('LOG_MAX_FILES', '7'))

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL.upper())
    # resolve relative paths against current working directory for local dev
    log_path = pathlib.Path(LOG_FILE_PATH)
    if not log_path.is_absolute():
        log_path = pathlib.Path(os.getcwd()) / log_path
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    if LOG_FORMAT == 'json':
        fmt = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s')
    else:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handlers
    combined = RotatingFileHandler(pathlib.Path(log_path) / 'combined.log', maxBytes=LOG_MAX_SIZE, backupCount=LOG_MAX_FILES)
    combined.setFormatter(fmt)
    logger.addHandler(combined)

    errors = RotatingFileHandler(pathlib.Path(log_path) / 'error.log', maxBytes=LOG_MAX_SIZE, backupCount=LOG_MAX_FILES)
    errors.setLevel(logging.ERROR)
    errors.setFormatter(fmt)
    logger.addHandler(errors)

    # inject context
    f = logging.Filter()
    f.filter = _inject_request_context
    logger.addFilter(f)

    # capture warnings
    logging.captureWarnings(True)

    return logger


def log_request(request_id: str, method: str, path: str, status_code: int, duration_ms: float, ip: str = None):
    logger = get_logger()
    logger.info('http_request', extra={'request_id': request_id, 'method': method, 'path': path, 'status_code': status_code, 'duration_ms': duration_ms, 'ip': ip})


def log_error(error: Exception, context: dict = None):
    logger = get_logger()
    logger.exception('error', exc_info=True, extra=context or {})


def log_model_load(model_name: str, device: str, load_time_ms: float):
    logger = get_logger()
    logger.info('model_load', extra={'model': model_name, 'device': device, 'load_time_ms': load_time_ms})


def log_ocr_result(request_id: str, image_size_mb: float, text_length: int, confidence: float, duration_ms: float):
    logger = get_logger()
    logger.info('ocr_result', extra={'request_id': request_id, 'image_size_mb': image_size_mb, 'text_length': text_length, 'confidence': confidence, 'duration_ms': duration_ms})


def log_llm_call(request_id: str, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: float, cost: float = None):
    logger = get_logger()
    logger.info('llm_call', extra={'request_id': request_id, 'model': model, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'duration_ms': duration_ms, 'cost': cost})


def log_graph_operation(request_id: str, operation: str, node_count: int, edge_count: int, duration_ms: float, user_id: str = None):
    logger = get_logger()
    logger.info('graph_operation', extra={
        'request_id': request_id,
        'operation': operation,
        'node_count': node_count,
        'edge_count': edge_count,
        'duration_ms': duration_ms,
        'user_id': user_id,
    })


def log_summarization(request_id: str, mode: str, brief_length: int, detailed_length: int, duration_ms: float, cache_hit: bool = False, cost: float = None):
    logger = get_logger()
    logger.info('summarization', extra={
        'request_id': request_id,
        'mode': mode,
        'brief_word_count': brief_length,
        'detailed_word_count': detailed_length,
        'duration_ms': duration_ms,
        'cache_hit': cache_hit,
        'cost': cost,
    })


def log_flashcard_generation(request_id: str, flashcard_count: int, source_counts: dict, duration_ms: float, cache_hit: bool = False, cost: float = None):
    logger = get_logger()
    logger.info('flashcard_generation', extra={
        'request_id': request_id,
        'flashcard_count': flashcard_count,
        'source_counts': source_counts,
        'duration_ms': duration_ms,
        'cache_hit': cache_hit,
        'cost': cost,
    })


def log_quiz_generation(request_id: str, question_count: int, question_types: list, duration_ms: float, cost: float = None):
    logger = get_logger()
    logger.info('quiz_generation', extra={
        'request_id': request_id,
        'question_count': question_count,
        'question_types': question_types,
        'duration_ms': duration_ms,
        'cost': cost,
    })


def log_mindmap_generation(request_id: str, node_count: int, edge_count: int, duration_ms: float, cost: float = None):
    logger = get_logger()
    logger.info('mindmap_generation', extra={
        'request_id': request_id,
        'node_count': node_count,
        'edge_count': edge_count,
        'duration_ms': duration_ms,
        'cost': cost,
    })


def log_ocr_fallback(task_id: str, chain: list, final_service: str, final_confidence: float):
    logger = get_logger()
    logger.info('ocr_fallback_chain', extra={
        'task_id': task_id,
        'chain': chain,
        'final_service': final_service,
        'final_confidence': final_confidence,
    })
