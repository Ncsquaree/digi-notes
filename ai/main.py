import os
import sys
import os
import os
import sys
import time
import signal
import asyncio
import traceback
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
import tempfile
import pathlib

from modules.ocr import (
    TrOCRHandler,
    TextractHandler,
    PixtralOCR,
    preprocess_image,
    TrOCRModelError,
    TrOCRInferenceError,
    PixtralModelError,
    PixtralInferenceError,
    TextractError,
    ImagePreprocessingError,
    StructuredDocument,
    structure_document,
    flatten_structured_text,
)
from modules.ocr.preprocess import PREPROCESSING_ENABLED
from modules.semantic import (
    parse_academic_content,
    ParsedContent,
    LLMParserError,
    LLMAPIError,
    LLMValidationError,
    LLMTimeoutError,
    Summarizer,
    generate_summary,
    SummaryResult,
    SummaryMode,
    SummarizerError,
    SummarizerAPIError,
    SummarizerTimeoutError,
    QuizGenerator,
    generate_quiz,
    QuizResponse,
    QuizQuestion,
    QuestionType,
    QuizGeneratorError,
    QuizAPIError,
    QuizValidationError,
    QuizTimeoutError,
    MindmapGenerator,
    generate_mindmap,
    MindmapResponse,
    MindmapNode,
    MindmapEdge,
    MindmapGeneratorError,
    MindmapAPIError,
    MindmapValidationError,
    MindmapTimeoutError,
)
from modules.flashcards import (
    generate_flashcards,
    FlashcardItem,
    GenerateFlashcardsRequest,
    GenerateFlashcardsResponse,
    FlashcardGeneratorError,
    FlashcardAPIError,
    FlashcardValidationError,
    FlashcardTimeoutError,
)
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseSettings
from modules.knowledge_graph import GraphBuilder, GraphQueries, NeptuneConnector, GraphBuildError, NeptuneConnectionError, NeptuneQueryError, GraphQueryError

from modules.utils import get_logger, set_request_context, get_request_context, FileHandler, FileDownloadError
from modules.utils import log_quiz_generation, log_mindmap_generation
from modules.utils import TaskManager, TaskStatus
from modules.utils.logger import log_ocr_fallback
import uuid
from PIL import Image
from modules.utils import TaskManager, TaskStatus
import uuid
from PIL import Image

LOG = get_logger()


class Settings(BaseSettings):
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '8000'))
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    BACKEND_URL: str = os.getenv('BACKEND_URL', 'http://backend:5000')
    CORS_ORIGIN: str = os.getenv('CORS_ORIGIN', os.getenv('BACKEND_URL', 'http://backend:5000'))
    SHUTDOWN_TIMEOUT_MS: int = int(os.getenv('SHUTDOWN_TIMEOUT_MS', '15000'))
    REDIS_REQUIRED_FOR_READY: bool = os.getenv('REDIS_REQUIRED_FOR_READY', 'false').lower() in ('1','true','yes')
    OPENAI_REQUIRED_FOR_READY: bool = os.getenv('OPENAI_REQUIRED_FOR_READY', 'false').lower() in ('1','true','yes')
    NEPTUNE_REQUIRED_FOR_READY: bool = os.getenv('NEPTUNE_REQUIRED_FOR_READY', 'false').lower() in ('1','true','yes')


settings = Settings()

app = FastAPI(title='Digi Notes AI Service', version='1.0.0', description='AI microservice for Digi Notes')

# CORS config
origins = [o.strip() for o in settings.CORS_ORIGIN.split(',') if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware('http')
async def add_request_id_and_logging(request: Request, call_next):
    # prefer incoming X-Request-ID header for cross-service tracing
    header_request_id = None
    try:
        header_request_id = request.headers.get('x-request-id') or request.headers.get('X-Request-ID')
    except Exception:
        header_request_id = None
    request_id = header_request_id or os.urandom(8).hex()
    request.state.request_id = request_id
    set_request_context(request_id)
    start = time.time()
    LOG.info('http_request_start', extra={'method': request.method, 'path': request.url.path, 'request_id': request_id, 'client': request.client.host if request.client else None})
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        LOG.exception('Unhandled exception in request', exc_info=True)
        body = {'success': False, 'error': {'message': 'Internal server error', 'request_id': request_id}}
        return JSONResponse(status_code=500, content=body)
    duration = int((time.time() - start) * 1000)
    LOG.info('http_request_end', extra={'method': request.method, 'path': request.url.path, 'status_code': response.status_code, 'duration_ms': duration, 'request_id': request_id})
    # echo back the request id for downstream tracing
    try:
        response.headers['X-Request-ID'] = request_id
    except Exception:
        pass
    return response


@app.get('/health')
async def health():
    return {'status': 'ok', 'timestamp': datetime.utcnow().isoformat() + 'Z', 'service': 'ai'}


def _check_postgres():
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            dbname=os.getenv('DB_NAME', 'digi_notes'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            connect_timeout=3,
        )
        conn.close()
        return 'ok'
    except Exception as e:
        return f'error: {str(e)}'


def _check_redis():
    try:
        import redis
        r = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=int(os.getenv('REDIS_PORT', '6379')), password=os.getenv('REDIS_PASSWORD') or None, socket_timeout=3)
        r.ping()
        return 'ok'
    except Exception as e:
        return f'error: {str(e)}'


def _check_s3():
    try:
        import boto3
        s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
        bucket = os.getenv('AWS_S3_BUCKET')
        if not bucket:
            return 'error: no bucket configured'
        s3.head_bucket(Bucket=bucket)
        return 'ok'
    except Exception as e:
        return f'error: {str(e)}'


def _check_openai():
    try:
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            # If OpenAI is required for readiness, treat missing key as error
            if settings.OPENAI_REQUIRED_FOR_READY:
                return 'error: no openai key'
            return 'warn: no openai key'

        # lightweight connectivity check using the public models list endpoint
        # use requests to avoid requiring a specific OpenAI client version here
        import requests
        resp = requests.get('https://api.openai.com/v1/models', headers={'Authorization': f'Bearer {key}'}, timeout=5)
        if resp.status_code == 200:
            return 'ok'
        else:
            return f'error: openai status {resp.status_code}'
    except Exception as e:
        return f'error: {str(e)}'


def _check_neptune():
    try:
        from modules.knowledge_graph import NeptuneConnector
        nc = NeptuneConnector.get_instance()
        healthy = nc.health_check()
        return 'ok' if healthy else 'error: neptune unreachable'
    except Exception as e:
        return f'error: {str(e)}'


@app.get('/ready')
async def ready():
    services = {}
    services['database'] = _check_postgres()
    redis_status = _check_redis()
    services['redis'] = redis_status
    services['s3'] = _check_s3()
    services['openai'] = _check_openai()
    services['neptune'] = _check_neptune()

    ready_ok = True
    if services['database'].startswith('error'):
        ready_ok = False
    if settings.REDIS_REQUIRED_FOR_READY and services['redis'].startswith('error'):
        ready_ok = False
    if services['s3'].startswith('error'):
        ready_ok = False
    # If OpenAI is required for readiness, consider its errors fatal
    if settings.OPENAI_REQUIRED_FOR_READY and services['openai'].startswith('error'):
        ready_ok = False
    # Neptune readiness can be gated by env var
    if settings.NEPTUNE_REQUIRED_FOR_READY and services.get('neptune', '').startswith('error'):
        ready_ok = False

    status_code = 200 if ready_ok else 503
    return JSONResponse(status_code=status_code, content={'status': 'ready' if ready_ok else 'not ready', 'services': services})


class OCRRequest(BaseModel):
    s3_key: Optional[str] = Field(None, description='S3 object key')
    s3_url: Optional[str] = Field(None, description='Full S3 URL')
    use_preprocessing: bool = Field(True, description='Apply image preprocessing')
    fallback_to_textract: bool = Field(True, description='Use Textract if TrOCR fails')

    @validator('s3_key', 's3_url', always=True)
    def at_least_one(cls, v, values, **kwargs):
        if not (values.get('s3_key') or values.get('s3_url') or v):
            raise ValueError('Either s3_key or s3_url must be provided')
        return v


class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    method: str
    preprocessing_applied: bool
    metadata: dict
    request_id: str


class MistralOCRRequest(BaseModel):
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    use_preprocessing: bool = True
    fallback_to_trocr: bool = True
    max_pdf_pages: Optional[int] = Field(None, description='Max pages for PDF (default 10)')

    @validator('s3_key', 's3_url', always=True)
    def at_least_one(cls, v, values, **kwargs):
        if not (values.get('s3_key') or values.get('s3_url') or v):
            raise ValueError('Either s3_key or s3_url must be provided')
        return v


class MistralOCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    method: str
    preprocessing_applied: bool
    page_count: Optional[int] = None
    pages: Optional[List[dict]] = None
    metadata: dict
    request_id: str


class OCRStructureRequest(BaseModel):
    text: str = Field(..., description='Raw OCR text to structure')
    options: Optional[dict] = Field(default_factory=dict)


class OCRStructureResponse(BaseModel):
    success: bool
    structured_document: StructuredDocument
    flattened_text: str
    metadata: dict
    request_id: str


class SemanticParseRequest(BaseModel):
    text: str = Field(..., description='OCR extracted text to parse', max_length=int(os.getenv('LLM_PARSER_MAX_TEXT_LENGTH', '50000')))
    options: Optional[dict] = Field({}, description='Parsing options')


class SemanticParseResponse(BaseModel):
    success: bool
    parsed_content: ParsedContent
    metadata: dict
    request_id: str


@app.post('/ocr/extract', response_model=OCRResponse)
async def ocr_extract(req: OCRRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    temp_files = []
    fh = FileHandler()
    local_path = None
    processed_pil = None
    preprocessing_steps = []
    start_total = time.time()
    try:
        # Download
        try:
            if req.s3_key:
                local_path = fh.download_from_s3_by_key(req.s3_key)
            else:
                local_path = fh.download_from_s3(req.s3_url)
        except FileDownloadError as e:
            LOG.warning('file_download_failed', extra={'request_id': request_id})
            return JSONResponse(status_code=400, content={'success': False, 'error': 'File download failed', 'details': str(e), 'request_id': request_id})

        file_info = fh.get_file_info(local_path)

        # Optional preprocessing
        pil_img = None
        preprocessing_requested_but_disabled = False
        if req.use_preprocessing:
            if PREPROCESSING_ENABLED:
                try:
                    pre = preprocess_image(local_path)
                    processed_pil = pre.get('processed')
                    preprocessing_steps = pre.get('steps_applied', [])
                    pil_img = processed_pil
                except ImagePreprocessingError as e:
                    LOG.warning('preprocessing_failed', extra={'request_id': request_id})
                    # parse structured error_type if present
                    msg = str(e)
                    error_type = 'other'
                    if msg.startswith('image_too_small'):
                        error_type = 'image_too_small'
                    elif msg.startswith('image_too_large'):
                        error_type = 'image_too_large'
                    return JSONResponse(status_code=400, content={'success': False, 'error': 'Image preprocessing failed', 'details': msg, 'request_id': request_id, 'metadata': {'preprocessing_error_type': error_type}})
            else:
                preprocessing_requested_but_disabled = True
                from PIL import Image
                pil_img = Image.open(local_path).convert('RGB')
        else:
            from PIL import Image
            pil_img = Image.open(local_path).convert('RGB')

        # Run OCR: Prefer Pixtral when enabled, fallback to TrOCR
        used_service = None
        text = ''
        confidence = 0.0
        model_name = None
        device = None
        pixtral_enabled = os.getenv('PIXTRAL_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        if pixtral_enabled:
            try:
                pix = PixtralOCR.get_instance()
                p_start = time.time()
                p_res = pix.extract_text(pil_img)
                p_ms = int((time.time() - p_start) * 1000)
                LOG.info('pixtral_result', extra={'request_id': request_id, 'duration_ms': p_ms})
                used_service = 'pixtral'
                text = p_res.get('text', '')
                confidence = float(p_res.get('confidence', 0.0))
                model_name = p_res.get('model')
                device = p_res.get('device')
            except (PixtralModelError, PixtralInferenceError) as e:
                LOG.warning('pixtral_unavailable_or_failed', extra={'request_id': request_id, 'details': str(e)})
                # fallback to TrOCR
        if used_service is None:
            try:
                trocr = TrOCRHandler.get_instance()
                t_start = time.time()
                t_res = trocr.extract_text(pil_img)
                t_ms = int((time.time() - t_start) * 1000)
                LOG.info('trocr_result', extra={'request_id': request_id, 'duration_ms': t_ms, 'confidence': t_res.get('confidence')})
                used_service = 'trocr'
                text = t_res.get('text', '')
                confidence = float(t_res.get('confidence', 0.0))
                model_name = t_res.get('model')
                device = t_res.get('device')
            except TrOCRModelError as e:
                LOG.exception('trocr_model_error', exc_info=True)
                return JSONResponse(status_code=500, content={'success': False, 'error': 'Model loading failed', 'details': str(e), 'request_id': request_id})
            except TrOCRInferenceError as e:
                LOG.exception('trocr_inference_error', exc_info=True)
                return JSONResponse(status_code=500, content={'success': False, 'error': 'OCR inference failed', 'details': str(e), 'request_id': request_id})

        # Fallback to Textract if confidence low
        try:
            threshold = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.7'))
        except Exception:
            threshold = 0.7

        textract_data = None
        if confidence < threshold and req.fallback_to_textract:
            th = TextractHandler()
            if not th.is_enabled():
                LOG.warning('textract_requested_but_unavailable', extra={'request_id': request_id})
                return JSONResponse(status_code=502, content={'success': False, 'error': 'Textract requested but not available', 'details': 'Textract disabled or credentials missing', 'request_id': request_id})
            try:
                    # prefer s3_key to call Textract on original S3 object
                    if req.s3_key:
                        textract_data = th.extract_text_from_s3(os.getenv('AWS_S3_BUCKET'), req.s3_key)
                    elif req.s3_url:
                        # parse bucket/key from url
                        try:
                            bucket, key = fh.parse_s3_url(req.s3_url)
                        except Exception:
                            bucket, key = os.getenv('AWS_S3_BUCKET'), None
                        if bucket and key:
                            textract_data = th.extract_text_from_s3(bucket, key)
                        else:
                            textract_data = th.extract_text_from_local(local_path)
                    else:
                        textract_data = th.extract_text_from_local(local_path)
                    if textract_data:
                        used_service = 'textract'
                        text = textract_data.get('text', '')
                        confidence = float(textract_data.get('confidence', 0.0))
            except TextractError as e:
                LOG.exception('textract_fallback_failed', exc_info=True)
                return JSONResponse(status_code=502, content={'success': False, 'error': 'Textract fallback failed', 'details': str(e), 'request_id': request_id})

        total_ms = int((time.time() - start_total) * 1000)
        metadata = {
            'processing_time_ms': total_ms,
            'file_info': file_info,
            'device': device if 'device' in locals() else None,
            'model_name': model_name if 'model_name' in locals() else None,
            'preprocessing_steps': preprocessing_steps,
            'preprocessing_requested_but_disabled': preprocessing_requested_but_disabled if 'preprocessing_requested_but_disabled' in locals() else False,
        }

        return OCRResponse(success=True, text=text, confidence=confidence, method=used_service, preprocessing_applied=bool(preprocessing_steps), metadata=metadata, request_id=request_id)

    finally:
        # cleanup temp files
        try:
            if local_path:
                fh.cleanup_temp_file(local_path)
        except Exception:
            LOG.exception('cleanup_local_failed', exc_info=True)


@app.post('/ocr/mistral', response_model=MistralOCRResponse)
async def ocr_mistral(req: MistralOCRRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    fh = FileHandler()
    local_path = None
    try:
        if req.s3_key:
            local_path = fh.download_from_s3_by_key(req.s3_key)
        else:
            local_path = fh.download_from_s3(req.s3_url)

        file_info = fh.get_file_info(local_path)
        is_pdf = file_info.get('mime_type') == 'application/pdf' or (file_info.get('extension', '').lower() == '.pdf')
        pix = PixtralOCR.get_instance()
        used_method = 'pixtral'

        if is_pdf:
            max_pages = req.max_pdf_pages or int(os.getenv('PIXTRAL_MAX_PDF_PAGES', '10'))
            result = await asyncio.to_thread(pix.extract_text_from_pdf, local_path, max_pages, req.use_preprocessing)
        else:
            result = await asyncio.to_thread(pix.extract_text_from_path, local_path, req.use_preprocessing)

        text = result.get('text', '')
        confidence = float(result.get('confidence', 0.0))

        threshold = float(os.getenv('MISTRAL_CONFIDENCE_THRESHOLD', '0.7'))
        if confidence < threshold and req.fallback_to_trocr:
            LOG.info('mistral_low_confidence_fallback', extra={'request_id': request_id, 'confidence': confidence, 'threshold': threshold})
            if not is_pdf:
                from PIL import Image
                trocr = TrOCRHandler.get_instance()
                pil_img = Image.open(local_path).convert('RGB')
                trocr_result = await asyncio.to_thread(trocr.extract_text, pil_img)
                text = trocr_result.get('text', text)
                confidence = float(trocr_result.get('confidence', confidence))
                used_method = 'trocr'
            else:
                LOG.warning('trocr_pdf_fallback_skipped', extra={'request_id': request_id})

        return MistralOCRResponse(
            success=True,
            text=text,
            confidence=confidence,
            method=used_method,
            preprocessing_applied=bool(result.get('preprocessing_applied', False)),
            page_count=result.get('page_count'),
            pages=result.get('pages'),
            metadata={
                'file_info': file_info,
                'preprocessing_steps': result.get('preprocessing_steps', []),
                'model': result.get('model'),
                'device': result.get('device'),
            },
            request_id=request_id,
        )
    except PixtralModelError as e:
        LOG.exception('pixtral_model_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Pixtral model loading failed', 'details': str(e), 'request_id': request_id})


@app.post('/ocr/structure', response_model=OCRStructureResponse)
async def ocr_structure(req: OCRStructureRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    try:
        if not req.text or not req.text.strip():
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Empty text', 'request_id': request_id})
        source_type = (req.options or {}).get('source_type')
        page_count = (req.options or {}).get('page_count')
        ocr_confidence = (req.options or {}).get('ocr_confidence')
        llm_fallback = (req.options or {}).get('llm_fallback', False)
        llm_threshold = float((req.options or {}).get('llm_threshold') or os.getenv('STRUCTURER_LLM_THRESHOLD', '0.6'))
        sd: StructuredDocument = await asyncio.to_thread(structure_document, req.text, source_type, page_count, ocr_confidence, llm_fallback, llm_threshold, (req.options or {}).get('ocr_method'))
        flattened = await asyncio.to_thread(flatten_structured_text, sd)
        meta = {'sections': len(sd.sections), 'word_count': sd.document_metadata.word_count, 'language': sd.document_metadata.language}
        return OCRStructureResponse(success=True, structured_document=sd, flattened_text=flattened, metadata=meta, request_id=request_id)
    except Exception as e:
        LOG.exception('ocr_structure_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Structuring failed', 'details': str(e), 'request_id': request_id})
    except PixtralInferenceError as e:
        LOG.exception('pixtral_inference_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Pixtral OCR failed', 'details': str(e), 'request_id': request_id})
    except FileDownloadError as e:
        return JSONResponse(status_code=400, content={'success': False, 'error': 'File download failed', 'details': str(e), 'request_id': request_id})
    finally:
        if local_path:
            try:
                fh.cleanup_temp_file(local_path)
            except Exception:
                LOG.exception('cleanup_failed', exc_info=True)


class ProcessNoteRequest(BaseModel):
    user_id: str
    task_id: Optional[str] = None
    note_id: str
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    options: Optional[dict] = Field(default_factory=dict)

    @validator('s3_key', 's3_url', always=True)
    def at_least_one_source(cls, v, values, **kwargs):
        if not (values.get('s3_key') or values.get('s3_url') or v):
            raise ValueError('Either s3_key or s3_url must be provided')
        return v


class ProcessNoteResponse(BaseModel):
    success: bool
    task_id: str
    message: Optional[str]
    request_id: str


@app.post('/process-note', response_model=ProcessNoteResponse, status_code=202)
async def process_note(req: ProcessNoteRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    task_id = req.task_id or (req.options or {}).get('task_id') or uuid.uuid4().hex
    tm = TaskManager.get_instance()
    tm.create_task(task_id, req.user_id, req.note_id, metadata={'s3_key': req.s3_key, 's3_url': req.s3_url, **(req.options or {})})

    # pipeline core - run inside wait_for when awaiting
    async def _run_pipeline():
        tm.update_status(task_id, TaskStatus.PROCESSING, progress_pct=0, current_step='start')

        # Step weights (percent allocation)
        weights = {
            'download': 5,
            'ocr': 25,
            'structure': 10,
            'parse': 15,
            'summarize': 15,
            'graph': 15,
            'flashcards': 10,
            'finalize': 5,
        }

        total_done = 0
        result_bundle = {}
        fh = None

        try:
            fh = FileHandler()

            # 1) Download
            step = 'download'
            try:
                if req.s3_key:
                    local_path = await asyncio.to_thread(fh.download_from_s3_by_key, req.s3_key)
                else:
                    local_path = await asyncio.to_thread(fh.download_from_s3, req.s3_url)
                result_bundle['local_path'] = local_path
                total_done += weights[step]
                tm.update_progress(task_id, step, total_done, {'local_path': local_path})
                tm.mark_step_complete(task_id, step, {'local_path': local_path})
            except Exception as e:
                tm.mark_step_failed(task_id, 'download', str(e))
                tm.fail_task(task_id, f'download failed: {e}')
                return result_bundle

            # 2) OCR
            step = 'ocr'
            try:
                pil_img = Image.open(local_path).convert('RGB')
                pixtral_enabled = os.getenv('PIXTRAL_ENABLED', 'true').lower() in ('1', 'true', 'yes')
                try:
                    pixtral_threshold = float(os.getenv('PIXTRAL_CONFIDENCE_THRESHOLD', '0.75'))
                except Exception:
                    pixtral_threshold = 0.75
                try:
                    textract_threshold = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.7'))
                except Exception:
                    textract_threshold = 0.7

                used_service = None
                text = ''
                confidence = 0.0
                preprocessing_steps = []
                fallback_chain = []

                # Primary: Pixtral
                if pixtral_enabled:
                    try:
                        pix = PixtralOCR.get_instance()
                        pix_res = await asyncio.to_thread(pix.extract_text, pil_img)
                        text = pix_res.get('text', '')
                        confidence = float(pix_res.get('confidence', 0.0))
                        used_service = 'pixtral'
                        preprocessing_steps = pix_res.get('preprocessing_steps', [])
                        fallback_chain.append('pixtral')
                        LOG.info('pixtral_ocr_success', extra={'task_id': task_id, 'confidence': confidence})
                    except (PixtralModelError, PixtralInferenceError) as e:
                        fallback_chain.append('pixtral_failed')
                        LOG.warning('pixtral_ocr_failed', extra={'task_id': task_id, 'error': str(e)})

                # Fallback 1: TrOCR
                if used_service is None or confidence < pixtral_threshold:
                    try:
                        trocr = TrOCRHandler.get_instance()
                        trocr_res = await asyncio.to_thread(trocr.extract_text, pil_img)
                        trocr_text = trocr_res.get('text', '')
                        trocr_conf = float(trocr_res.get('confidence', 0.0))
                        fallback_chain.append('trocr')
                        if trocr_conf > confidence:
                            text = trocr_text
                            confidence = trocr_conf
                            used_service = 'trocr'
                        LOG.info('trocr_fallback_used', extra={'task_id': task_id, 'confidence': trocr_conf})
                    except (TrOCRModelError, TrOCRInferenceError) as e:
                        fallback_chain.append('trocr_failed')
                        LOG.warning('trocr_fallback_failed', extra={'task_id': task_id, 'error': str(e)})

                # Fallback 2: Textract
                if confidence < textract_threshold and (req.options or {}).get('fallback_to_textract', True):
                    th = TextractHandler()
                    if th.is_enabled():
                        try:
                            textract_data = await asyncio.to_thread(th.extract_text_from_local, local_path)
                            fallback_chain.append('textract')
                            if textract_data:
                                tex_text = textract_data.get('text', '')
                                tex_conf = float(textract_data.get('confidence', 0.0))
                                if tex_conf > confidence:
                                    text = tex_text
                                    confidence = tex_conf
                                    used_service = 'textract'
                                LOG.info('textract_fallback_used', extra={'task_id': task_id, 'confidence': tex_conf})
                        except TextractError as e:
                            fallback_chain.append('textract_failed')
                            LOG.warning('textract_fallback_failed', extra={'task_id': task_id, 'error': str(e)})
                    else:
                        fallback_chain.append('textract_unavailable')

                if used_service is None:
                    raise RuntimeError('All OCR methods failed')

                result_bundle['ocr'] = {
                    'text': text,
                    'confidence': confidence,
                    'service': used_service,
                    'preprocessing_steps': preprocessing_steps,
                    'fallback_chain': fallback_chain,
                }
                total_done += weights[step]
                tm.update_progress(task_id, step, total_done, result_bundle['ocr'])
                tm.mark_step_complete(task_id, step, result_bundle['ocr'])
                try:
                    log_ocr_fallback(task_id, fallback_chain, used_service, confidence)
                except Exception:
                    LOG.warning('log_ocr_fallback_failed', exc_info=True)
            except Exception as e:
                tm.mark_step_failed(task_id, step, str(e))
                # OCR is critical; fail
                tm.fail_task(task_id, f'ocr failed: {e}')
                return result_bundle

            # 2b) Structure OCR (optional)
            step = 'structure'
            try:
                if (req.options or {}).get('enable_structurer', True):
                    sd: StructuredDocument = await asyncio.to_thread(
                        structure_document,
                        text,
                        (result_bundle.get('file_info') or {}).get('mime_type'),
                        None,
                        confidence,
                        (req.options or {}).get('llm_fallback', False),
                        float((req.options or {}).get('llm_threshold') or os.getenv('STRUCTURER_LLM_THRESHOLD', '0.6')),
                        (result_bundle.get('ocr') or {}).get('service'),
                    )
                    # store both structured and a flattened convenience text
                    flattened = await asyncio.to_thread(flatten_structured_text, sd)
                    result_bundle['structured'] = {
                        'document': sd.model_dump(),
                        'flattened_text': flattened,
                        'ocr_metadata': {
                            'service': (result_bundle.get('ocr') or {}).get('service'),
                            'confidence': (result_bundle.get('ocr') or {}).get('confidence'),
                            'preprocessing_steps': (result_bundle.get('ocr') or {}).get('preprocessing_steps', []),
                        },
                    }
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'sections': len(sd.sections), 'word_count': sd.document_metadata.word_count})
                    tm.mark_step_complete(task_id, step, {'sections': len(sd.sections)})
                else:
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
            except Exception as e:
                tm.mark_step_failed(task_id, step, str(e))
                # Structuring is non-critical; continue based on flag
                if os.getenv('PROCESS_NOTE_CONTINUE_ON_ERROR', 'true').lower() in ('1', 'true', 'yes'):
                    result_bundle['structured'] = None
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
                else:
                    tm.fail_task(task_id, f'structure failed: {e}')
                    return result_bundle

            # 3) Parse semantic
            step = 'parse'
            try:
                # Prefer flattened structured text if available
                parse_text = (result_bundle.get('structured') or {}).get('flattened_text') or text
                parsed = await asyncio.to_thread(parse_academic_content, parse_text, request_id)
                result_bundle['parsed'] = parsed.model_dump() if hasattr(parsed, 'model_dump') else dict(parsed)
                total_done += weights[step]
                tm.update_progress(task_id, step, total_done, {'parsed_summary': len(str(result_bundle['parsed']))})
                tm.mark_step_complete(task_id, step, {'parsed_length': len(str(result_bundle['parsed']))})
            except Exception as e:
                tm.mark_step_failed(task_id, step, str(e))
                # parsing is critical; fail
                tm.fail_task(task_id, f'parse failed: {e}')
                return result_bundle

            # 4) Summarize
            step = 'summarize'
            try:
                parsed_arg = result_bundle.get('parsed') or {}
                summary_res = await asyncio.to_thread(generate_summary, parsed_arg, 'both', request_id)
                result_bundle['summary'] = summary_res
                total_done += weights[step]
                tm.update_progress(task_id, step, total_done, {'summary_keys': list(summary_res.keys()) if isinstance(summary_res, dict) else []})
                tm.mark_step_complete(task_id, step, {'summary': summary_res})
            except Exception as e:
                tm.mark_step_failed(task_id, step, str(e))
                # summarization is non-critical; continue based on flag
                if os.getenv('PROCESS_NOTE_CONTINUE_ON_ERROR', 'true').lower() in ('1', 'true', 'yes'):
                    result_bundle['summary'] = None
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
                else:
                    tm.fail_task(task_id, f'summarize failed: {e}')
                    return result_bundle

            # 5) Graph building (optional)
            step = 'graph'
            try:
                if (req.options or {}).get('enable_graph', True):
                    builder = GraphBuilder(req.user_id, req.note_id, result_bundle.get('parsed') or {})
                    graph_res = await asyncio.to_thread(builder.build_graph)
                    result_bundle['graph'] = graph_res
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'nodes': graph_res.get('nodes_created'), 'edges': graph_res.get('edges_created')})
                    tm.mark_step_complete(task_id, step, graph_res)
                else:
                    # increment progress for skipped step
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
            except Exception as e:
                tm.mark_step_failed(task_id, step, str(e))
                if os.getenv('PROCESS_NOTE_CONTINUE_ON_ERROR', 'true').lower() in ('1', 'true', 'yes'):
                    result_bundle['graph'] = None
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
                else:
                    tm.fail_task(task_id, f'graph failed: {e}')
                    return result_bundle

            # 6) Flashcards (optional)
            step = 'flashcards'
            try:
                if (req.options or {}).get('enable_flashcards', True):
                    count = int((req.options or {}).get('flashcard_count') or os.getenv('PROCESS_NOTE_FLASHCARD_COUNT', '10'))
                    parsed_arg = result_bundle.get('parsed') or {}
                    fc_res = await asyncio.to_thread(generate_flashcards, parsed_arg, count, {'include_formulas': True}, request_id)
                    result_bundle['flashcards'] = fc_res
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'flashcard_count': len(fc_res.get('flashcards', [])) if isinstance(fc_res, dict) else 0})
                    tm.mark_step_complete(task_id, step, fc_res)
                else:
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
            except Exception as e:
                tm.mark_step_failed(task_id, step, str(e))
                if os.getenv('PROCESS_NOTE_CONTINUE_ON_ERROR', 'true').lower() in ('1', 'true', 'yes'):
                    result_bundle['flashcards'] = None
                    total_done += weights[step]
                    tm.update_progress(task_id, step, total_done, {'skipped': True})
                else:
                    tm.fail_task(task_id, f'flashcards failed: {e}')
                    return result_bundle

            # finalize
            total_done = 100
            tm.update_progress(task_id, 'finalize', total_done, {'result_keys': list(result_bundle.keys())})

            # determine completion vs partial
            task_state = tm.get_task(task_id) or {}
            steps_failed = task_state.get('steps_failed') or []
            # simple heuristic: check presence of 'ocr' and 'parse' in steps_completed list
            ocr_ok = any((isinstance(s, str) and s == 'ocr') or (isinstance(s, dict) and s.get('step') == 'ocr') for s in (task_state.get('steps_completed') or []))
            parse_ok = any((isinstance(s, str) and s == 'parse') or (isinstance(s, dict) and s.get('step') == 'parse') for s in (task_state.get('steps_completed') or []))

            if steps_failed and ocr_ok and parse_ok:
                tm.complete_partial(task_id, final_result=result_bundle)
                return result_bundle
            else:
                tm.complete_task(task_id, final_result=result_bundle)
                return result_bundle

        except Exception as e:
            LOG.exception('process_note_worker_error', exc_info=True)
            tm.fail_task(task_id, str(e))
            return result_bundle
        finally:
            # cleanup local file if present
            try:
                lp = result_bundle.get('local_path') if isinstance(result_bundle, dict) else None
                if lp and fh:
                    fh.cleanup_temp_file(lp)
            except Exception:
                LOG.exception('worker_cleanup_failed', exc_info=True)

    # decide whether to await full pipeline or run in background
    await_result = (req.options or {}).get('await_result', True)
    timeout = float(os.getenv('PROCESS_NOTE_TIMEOUT', '300'))

    if await_result:
        try:
            # run pipeline bounded by timeout
            res = await asyncio.wait_for(_run_pipeline(), timeout=timeout)
            # after completion, return flattened task info
            task = tm.get_task(task_id) or {}
            return JSONResponse(status_code=200, content={'success': True, 'task_id': task_id, 'status': task.get('status'), 'progress_pct': task.get('progress_pct'), 'current_step': task.get('current_step'), 'steps_completed': task.get('steps_completed'), 'steps_failed': task.get('steps_failed'), 'result': task.get('result'), 'error_message': task.get('error_message'), 'created_at': task.get('created_at'), 'updated_at': task.get('updated_at'), 'request_id': request_id})
        except asyncio.TimeoutError:
            tm.fail_task(task_id, f'pipeline timeout after {timeout} seconds')
            return JSONResponse(status_code=504, content={'success': False, 'error': 'Processing timeout', 'task_id': task_id, 'request_id': request_id})
    else:
        # background: schedule with its own timeout guard
        async def _bg_wrapper():
            try:
                await asyncio.wait_for(_run_pipeline(), timeout=timeout)
            except asyncio.TimeoutError:
                tm.fail_task(task_id, f'pipeline timeout after {timeout} seconds')

        asyncio.create_task(_bg_wrapper())
        return ProcessNoteResponse(success=True, task_id=task_id, message='Processing started (background)', request_id=request_id)


@app.get('/process-note/status/{task_id}')
async def process_note_status(task_id: str, fastapi_request: Request = None):
    request_id = getattr(fastapi_request.state, 'request_id', None) if fastapi_request else os.urandom(8).hex()
    tm = TaskManager.get_instance()
    task = tm.get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={'success': False, 'error': 'Task not found', 'task_id': task_id, 'request_id': request_id})
    # Flattened status contract
    resp = {
        'success': True,
        'task_id': task_id,
        'status': task.get('status'),
        'progress_pct': task.get('progress_pct'),
        'current_step': task.get('current_step'),
        'steps_completed': task.get('steps_completed'),
        'steps_failed': task.get('steps_failed'),
        'result': task.get('result'),
        'error_message': task.get('error_message'),
        'created_at': task.get('created_at'),
        'updated_at': task.get('updated_at'),
        'request_id': request_id,
    }
    return JSONResponse(status_code=200, content=resp)

# Duplicate process-note implementation removed. The canonical `/process-note`
# and `/process-note/status/{task_id}` handlers are defined earlier in this
# file and return the flattened status shape expected by the backend.


class GraphBuildRequest(BaseModel):
    user_id: str
    note_id: str
    parsed_content: ParsedContent


class GraphBuildResponse(BaseModel):
    success: bool
    nodes_created: int
    edges_created: int
    metadata: dict
    request_id: str


class GraphVisualizeResponse(BaseModel):
    success: bool
    nodes: list
    edges: list
    metadata: dict
    request_id: str


class RelatedConceptsResponse(BaseModel):
    success: bool
    concepts: list
    metadata: dict
    request_id: str


@app.post('/graph/build', response_model=GraphBuildResponse)
async def graph_build(req: GraphBuildRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    LOG.info('graph_build_start', extra={'request_id': request_id, 'user_id': req.user_id, 'note_id': req.note_id})
    start = time.time()
    try:
        builder = GraphBuilder(req.user_id, req.note_id, req.parsed_content)
        result = builder.build_graph()
        duration_ms = int((time.time() - start) * 1000)
        from modules.utils import log_graph_operation
        log_graph_operation(request_id, 'build', result.get('nodes_created', 0), result.get('edges_created', 0), duration_ms, user_id=req.user_id)
        metadata = {'processing_time_ms': duration_ms}
        return GraphBuildResponse(success=True, nodes_created=result.get('nodes_created', 0), edges_created=result.get('edges_created', 0), metadata=metadata, request_id=request_id)
    except GraphBuildError as e:
        LOG.exception('graph_build_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Graph build failed', 'details': str(e), 'request_id': request_id})
    except NeptuneConnectionError as e:
        LOG.exception('graph_build_neptune_unavailable', exc_info=True)
        return JSONResponse(status_code=503, content={'success': False, 'error': 'Neptune unavailable', 'details': str(e), 'request_id': request_id})
    except Exception as e:
        LOG.exception('graph_build_unknown', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Unexpected error', 'details': str(e), 'request_id': request_id})


@app.get('/graph/visualize/{userId}', response_model=GraphVisualizeResponse)
async def graph_visualize(userId: str, depth: int = 2, fastapi_request: Request = None):
    request_id = getattr(fastapi_request.state, 'request_id', None) if fastapi_request else os.urandom(8).hex()
    try:
        depth = max(1, min(5, int(depth)))
    except Exception:
        depth = 2
    try:
        res = GraphQueries.visualize_user_graph(userId, depth)
        metadata = {'depth': depth}
        from modules.utils import log_graph_operation
        log_graph_operation(request_id, 'visualize', len(res.get('nodes', [])), len(res.get('edges', [])), 0, user_id=userId)
        return GraphVisualizeResponse(success=True, nodes=res.get('nodes', []), edges=res.get('edges', []), metadata=metadata, request_id=request_id)
    except GraphQueryError as e:
        LOG.exception('graph_visualize_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Visualization query failed', 'details': str(e), 'request_id': request_id})
    except NeptuneConnectionError as e:
        LOG.exception('graph_visualize_neptune_unavailable', exc_info=True)
        return JSONResponse(status_code=503, content={'success': False, 'error': 'Neptune unavailable', 'details': str(e), 'request_id': request_id})


@app.get('/graph/related-concepts/{conceptId}', response_model=RelatedConceptsResponse)
async def graph_related_concepts(conceptId: str, limit: int = 10, fastapi_request: Request = None):
    request_id = getattr(fastapi_request.state, 'request_id', None) if fastapi_request else os.urandom(8).hex()
    try:
        limit = max(1, min(50, int(limit)))
    except Exception:
        limit = 10
    try:
        res = GraphQueries.get_related_concepts(conceptId, limit)
        metadata = {'limit': limit}
        from modules.utils import log_graph_operation
        log_graph_operation(request_id, 'related_concepts', len(res), 0, 0)
        return RelatedConceptsResponse(success=True, concepts=res, metadata=metadata, request_id=request_id)
    except GraphQueryError as e:
        LOG.exception('graph_related_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Related concepts query failed', 'details': str(e), 'request_id': request_id})
    except NeptuneConnectionError as e:
        LOG.exception('graph_related_neptune_unavailable', exc_info=True)
        return JSONResponse(status_code=503, content={'success': False, 'error': 'Neptune unavailable', 'details': str(e), 'request_id': request_id})


@app.post('/parse/semantic', response_model=SemanticParseResponse)
async def parse_semantic(req: SemanticParseRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    if not req.text or not req.text.strip():
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Empty text', 'request_id': request_id})
    LOG.info('semantic_parse_start', extra={'request_id': request_id, 'text_length': len(req.text)})
    start = time.time()
    try:
        parsed = parse_academic_content(req.text, request_id)
        duration_ms = int((time.time() - start) * 1000)
        metadata = {'processing_time_ms': duration_ms, 'text_length': len(req.text), 'model_used': os.getenv('OPENAI_MODEL')}
        LOG.info('semantic_parse_complete', extra={'request_id': request_id, 'duration_ms': duration_ms})
        return SemanticParseResponse(success=True, parsed_content=parsed, metadata=metadata, request_id=request_id)
    except LLMValidationError as e:
        LOG.exception('semantic_validation_error', exc_info=True)
        return JSONResponse(status_code=422, content={'success': False, 'error': 'Schema validation failed', 'details': str(e), 'request_id': request_id})
    except LLMTimeoutError as e:
        LOG.exception('semantic_timeout', exc_info=True)
        return JSONResponse(status_code=504, content={'success': False, 'error': 'LLM request timeout', 'details': str(e), 'request_id': request_id})
    except LLMAPIError as e:
        LOG.exception('semantic_api_error', exc_info=True)
        return JSONResponse(status_code=502, content={'success': False, 'error': 'LLM API error', 'details': str(e), 'request_id': request_id})
    except LLMParserError as e:
        LOG.exception('semantic_parser_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Parsing failed', 'details': str(e), 'request_id': request_id})
    except Exception as e:
        LOG.exception('semantic_unknown_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Unexpected error', 'details': str(e), 'request_id': request_id})


class SummarizeRequest(BaseModel):
    parsed_content: ParsedContent
    mode: str = Field('both', description='brief|detailed|both')
    options: Optional[dict] = Field({}, description='Options (reserved)')
    # Mode validation is performed at request time in the handler to return
    # a 400-style validation error (instead of FastAPI/Pydantic's default 422).


class SummarizeResponse(BaseModel):
    success: bool
    summary_brief: Optional[str]
    summary_detailed: Optional[str]
    metadata: dict
    request_id: str


class FlashcardGenerateRequest(BaseModel):
    parsed_content: ParsedContent
    count: Optional[int] = Field(None, description='Target flashcard count (None=all available)')
    include_formulas: bool = Field(True, description='Include formula-based questions')
    include_concepts: bool = Field(True, description='Include concept-based questions')
    options: Optional[dict] = Field(default_factory=dict, description='Generation options: min_difficulty, max_difficulty')


class FlashcardGenerateResponse(BaseModel):
    success: bool
    flashcards: List[FlashcardItem]
    metadata: dict
    request_id: str




@app.post('/summarize', response_model=SummarizeResponse)
async def summarize_content(req: SummarizeRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    # Request-level validation for `mode` to return 400 on invalid values
    if req.mode not in ('brief', 'detailed', 'both'):
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid mode', 'details': 'mode must be one of brief|detailed|both', 'request_id': request_id})

    LOG.info('summarize_start', extra={'request_id': request_id, 'mode': req.mode})
    start = time.time()
    try:
        parsed_dict = req.parsed_content.model_dump() if hasattr(req.parsed_content, 'model_dump') else dict(req.parsed_content)
        result = generate_summary(parsed_dict, mode=req.mode, request_id=request_id)
        duration_ms = int((time.time() - start) * 1000)
        metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
        metadata.update({'processing_time_ms': duration_ms, 'mode': req.mode, 'model_used': os.getenv('OPENAI_MODEL')})
        brief = result.get('brief') if isinstance(result, dict) else None
        detailed = result.get('detailed') if isinstance(result, dict) else None
        LOG.info('summarize_complete', extra={'request_id': request_id, 'duration_ms': duration_ms})
        return SummarizeResponse(success=True, summary_brief=brief, summary_detailed=detailed, metadata=metadata, request_id=request_id)
    except SummarizerTimeoutError as e:
        LOG.exception('summarize_timeout', exc_info=True)
        return JSONResponse(status_code=504, content={'success': False, 'error': 'Summarization timeout', 'details': str(e), 'request_id': request_id})
    except SummarizerAPIError as e:
        LOG.exception('summarize_api_error', exc_info=True)
        return JSONResponse(status_code=502, content={'success': False, 'error': 'LLM API error', 'details': str(e), 'request_id': request_id})
    except SummarizerError as e:
        LOG.exception('summarize_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Summarization failed', 'details': str(e), 'request_id': request_id})
    except Exception as e:
        LOG.exception('summarize_unknown', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Unexpected error', 'details': str(e), 'request_id': request_id})


@app.post('/flashcards/generate', response_model=FlashcardGenerateResponse)
async def generate_flashcards_endpoint(req: FlashcardGenerateRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    max_count = int(os.getenv('FLASHCARD_MAX_COUNT', '50'))
    if req.count and (req.count < 1 or req.count > max_count):
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid count', 'details': f'count must be 1-{max_count}', 'request_id': request_id})

    LOG.info('flashcard_generation_start', extra={'request_id': request_id, 'count': req.count})
    start = time.time()
    try:
        parsed_dict = req.parsed_content.model_dump() if hasattr(req.parsed_content, 'model_dump') else dict(req.parsed_content)
        # Merge explicit include flags into options so generator receives them
        options = req.options or {}
        options['include_formulas'] = req.include_formulas
        options['include_concepts'] = req.include_concepts
        result = generate_flashcards(parsed_dict, count=req.count, options=options, request_id=request_id)
        duration_ms = int((time.time() - start) * 1000)
        metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
        metadata.update({'processing_time_ms': duration_ms, 'model_used': os.getenv('OPENAI_MODEL')})
        flashcards = result.get('flashcards', []) if isinstance(result, dict) else []
        LOG.info('flashcard_generation_complete', extra={'request_id': request_id, 'count': len(flashcards), 'duration_ms': duration_ms})
        return FlashcardGenerateResponse(success=True, flashcards=flashcards, metadata=metadata, request_id=request_id)
    except FlashcardValidationError as e:
        LOG.exception('flashcard_validation_error', exc_info=True)
        return JSONResponse(status_code=422, content={'success': False, 'error': 'Validation failed', 'details': str(e), 'request_id': request_id})
    except FlashcardTimeoutError as e:
        LOG.exception('flashcard_timeout', exc_info=True)
        return JSONResponse(status_code=504, content={'success': False, 'error': 'LLM timeout', 'details': str(e), 'request_id': request_id})
    except FlashcardAPIError as e:
        LOG.exception('flashcard_api_error', exc_info=True)
        return JSONResponse(status_code=502, content={'success': False, 'error': 'LLM API error', 'details': str(e), 'request_id': request_id})
    except FlashcardGeneratorError as e:
        LOG.exception('flashcard_generation_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Flashcard generation failed', 'details': str(e), 'request_id': request_id})
    except Exception as e:
        LOG.exception('flashcard_unknown_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Unexpected error', 'details': str(e), 'request_id': request_id})


@app.post('/tools/generate-quiz', response_model=QuizGenerateResponse)
async def generate_quiz_endpoint(req: QuizGenerateRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    max_q = int(os.getenv('QUIZ_MAX_QUESTIONS', '20'))
    if req.question_count and (req.question_count < 1 or req.question_count > max_q):
        return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid question_count', 'details': f'question_count must be 1-{max_q}', 'request_id': request_id})
    if req.question_types:
        allowed = {'mcq', 'true_false', 'short_answer'}
        for t in req.question_types:
            if t not in allowed:
                return JSONResponse(status_code=400, content={'success': False, 'error': 'Invalid question type', 'details': str(t), 'request_id': request_id})

    LOG.info('quiz_generation_start', extra={'request_id': request_id, 'question_count': req.question_count, 'question_types': req.question_types})
    start = time.time()
    try:
        parsed_dict = req.parsed_content.model_dump() if hasattr(req.parsed_content, 'model_dump') else dict(req.parsed_content)
        # pass options through to generator
        result = generate_quiz(parsed_dict, question_count=req.question_count or 10, question_types=req.question_types, options=req.options, request_id=request_id)
        duration_ms = int((time.time() - start) * 1000)
        # result is expected to be a dict from generate_quiz convenience function
        if not isinstance(result, dict):
            result = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        # preserve generator-provided metadata fields and add processing_time/model_used if missing
        metadata = result.get('metadata', {}) or {}
        metadata.setdefault('processing_time_ms', duration_ms)
        metadata.setdefault('model_used', os.getenv('OPENAI_MODEL'))

        # instantiate typed payload for consistent OpenAPI schema
        from pydantic import ValidationError as PydanticValidationError
        try:
            quiz_payload = QuizPayload(**result)
        except PydanticValidationError as e:
            LOG.exception('quiz_payload_validation_failed', exc_info=True)
            raise QuizValidationError('Generated quiz payload does not match schema') from e

        log_quiz_generation(request_id, question_count=len(quiz_payload.questions), question_types=req.question_types or [], duration_ms=duration_ms)
        LOG.info('quiz_generation_complete', extra={'request_id': request_id, 'question_count': len(quiz_payload.questions), 'duration_ms': duration_ms})
        return QuizGenerateResponse(success=True, quiz=quiz_payload, metadata=metadata, request_id=request_id)
    except QuizValidationError as e:
        LOG.exception('quiz_validation_error', exc_info=True)
        return JSONResponse(status_code=422, content={'success': False, 'error': 'Validation failed', 'details': str(e), 'request_id': request_id})
    except QuizTimeoutError as e:
        LOG.exception('quiz_timeout', exc_info=True)
        return JSONResponse(status_code=504, content={'success': False, 'error': 'LLM timeout', 'details': str(e), 'request_id': request_id})
    except QuizAPIError as e:
        LOG.exception('quiz_api_error', exc_info=True)
        return JSONResponse(status_code=502, content={'success': False, 'error': 'LLM API error', 'details': str(e), 'request_id': request_id})
    except QuizGeneratorError as e:
        LOG.exception('quiz_generation_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Quiz generation failed', 'details': str(e), 'request_id': request_id})
    except Exception as e:
        LOG.exception('quiz_unknown_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Unexpected error', 'details': str(e), 'request_id': request_id})


@app.post('/tools/generate-mindmap', response_model=MindmapGenerateResponse)
async def generate_mindmap_endpoint(req: MindmapGenerateRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, 'request_id', None) or os.urandom(8).hex()
    LOG.info('mindmap_generation_start', extra={'request_id': request_id})
    start = time.time()
    try:
        parsed_dict = req.parsed_content.model_dump() if hasattr(req.parsed_content, 'model_dump') else dict(req.parsed_content)
        # pass options through
        result = generate_mindmap(parsed_dict, options=req.options, request_id=request_id)
        duration_ms = int((time.time() - start) * 1000)
        if not isinstance(result, dict):
            result = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        metadata = result.get('metadata', {}) or {}
        metadata.setdefault('processing_time_ms', duration_ms)
        metadata.setdefault('model_used', os.getenv('OPENAI_MODEL'))

        nodes = result.get('nodes') or []
        edges = result.get('edges') or []
        root_node_id = result.get('root_node_id') or ''
        mindmap = {'nodes': nodes, 'edges': edges, 'root_node_id': root_node_id}

        node_count = len(nodes)
        edge_count = len(edges)
        log_mindmap_generation(request_id, node_count=node_count, edge_count=edge_count, duration_ms=duration_ms)
        LOG.info('mindmap_generation_complete', extra={'request_id': request_id, 'node_count': node_count, 'edge_count': edge_count, 'duration_ms': duration_ms})
        return MindmapGenerateResponse(success=True, mindmap=mindmap, metadata=metadata, request_id=request_id)
    except MindmapValidationError as e:
        LOG.exception('mindmap_validation_error', exc_info=True)
        return JSONResponse(status_code=422, content={'success': False, 'error': 'Validation failed', 'details': str(e), 'request_id': request_id})
    except MindmapTimeoutError as e:
        LOG.exception('mindmap_timeout', exc_info=True)
        return JSONResponse(status_code=504, content={'success': False, 'error': 'LLM timeout', 'details': str(e), 'request_id': request_id})
    except MindmapAPIError as e:
        LOG.exception('mindmap_api_error', exc_info=True)
        return JSONResponse(status_code=502, content={'success': False, 'error': 'LLM API error', 'details': str(e), 'request_id': request_id})
    except MindmapGeneratorError as e:
        LOG.exception('mindmap_generation_failed', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Mindmap generation failed', 'details': str(e), 'request_id': request_id})
    except Exception as e:
        LOG.exception('mindmap_unknown_error', exc_info=True)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Unexpected error', 'details': str(e), 'request_id': request_id})


class QuizGenerateRequest(BaseModel):
    parsed_content: ParsedContent
    question_count: Optional[int] = Field(10, description='Number of questions (1-20)')
    question_types: Optional[List[str]] = Field(None, description='Question types: mcq, true_false, short_answer')
    options: Optional[dict] = Field(default_factory=dict, description='Generation options')


class QuizPayload(BaseModel):
    questions: List[QuizQuestion]
    total_points: int
    metadata: dict


class QuizGenerateResponse(BaseModel):
    success: bool
    quiz: QuizPayload
    metadata: dict
    request_id: str


class MindmapGenerateRequest(BaseModel):
    parsed_content: ParsedContent
    options: Optional[dict] = Field(default_factory=dict, description='Generation options')


class MindmapGenerateResponse(BaseModel):
    success: bool
    mindmap: dict
    metadata: dict
    request_id: str


@app.on_event('startup')
async def on_startup():
    LOG.info('AI service starting', extra={'env': settings.ENVIRONMENT})
    # Validate env minimal
    if not os.getenv('AWS_S3_BUCKET'):
        LOG.warning('AWS_S3_BUCKET not set; S3 features will be unavailable')
    # warm file handler
    try:
        fh = FileHandler()
        LOG.info('FileHandler ready')
    except Exception:
        LOG.exception('FileHandler init failed', exc_info=True)
    # warm Summarizer (non-blocking)
    try:
        try:
            Summarizer.get_instance()
            LOG.info('Summarizer warmup triggered')
        except Exception as e:
            LOG.warning('Summarizer warmup failed', extra={'error': str(e)})
    except Exception:
        LOG.exception('summarizer_warmup_error', exc_info=True)
    # warm FlashcardGenerator (non-blocking)
    try:
        try:
            from modules.flashcards import FlashcardGenerator
            FlashcardGenerator.get_instance()
            LOG.info('FlashcardGenerator warmup triggered')
        except Exception as e:
            LOG.warning('FlashcardGenerator warmup failed', extra={'error': str(e)})
    except Exception:
        LOG.exception('flashcardgenerator_warmup_error', exc_info=True)
    # warm QuizGenerator (non-blocking)
    try:
        try:
            QuizGenerator.get_instance()
            LOG.info('QuizGenerator warmup triggered')
        except Exception as e:
            LOG.warning('QuizGenerator warmup failed', extra={'error': str(e)})
    except Exception:
        LOG.exception('quizgenerator_warmup_error', exc_info=True)
    # warm MindmapGenerator (non-blocking)
    try:
        try:
            MindmapGenerator.get_instance()
            LOG.info('MindmapGenerator warmup triggered')
        except Exception as e:
            LOG.warning('MindmapGenerator warmup failed', extra={'error': str(e)})
    except Exception:
        LOG.exception('mindmapgenerator_warmup_error', exc_info=True)
    # warm Neptune connection (non-blocking)
    try:
        try:
            nc = NeptuneConnector.get_instance()
            LOG.info('NeptuneConnector warmup triggered')
        except Exception as e:
            LOG.warning('Neptune warmup failed', extra={'error': str(e)})
    except Exception:
        LOG.exception('neptune_warmup_error', exc_info=True)


@app.on_event('shutdown')
async def on_shutdown():
    LOG.info('AI service shutting down')
    try:
        try:
            nc = NeptuneConnector.get_instance()
            nc.close()
            LOG.info('NeptuneConnector closed')
        except Exception:
            LOG.warning('Neptune close failed or not initialized')
    except Exception:
        LOG.exception('neptune_close_error', exc_info=True)


def _install_signal_handlers(loop: Optional[asyncio.AbstractEventLoop] = None):
    if loop is None:
        loop = asyncio.get_event_loop()

    def _handler(signum, frame):
        LOG.info('Received shutdown signal', extra={'signal': signum})
        # allow uvicorn to handle shutdown; ensure loop stops after timeout
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


if __name__ == '__main__':
    import uvicorn

    _install_signal_handlers()
    workers = int(os.getenv('WORKERS', '1'))
    # uvicorn does not support --reload with multiple workers. Ensure we never
    # start with reload=True while workers>1. For development, prefer reload
    # and force a single worker.
    if settings.ENVIRONMENT == 'development':
        workers = 1
    reload_enabled = (settings.ENVIRONMENT == 'development') and (workers == 1)

    uvicorn.run(
        'main:app',
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=reload_enabled,
        workers=workers,
    )
