"""AWS Textract handler with retry logic using tenacity.

Provides TextractHandler with methods:
- is_enabled()
- extract_text_from_s3(bucket, key)
- extract_text_from_local(image_path)

Raises TextractError on failures.
"""
from __future__ import annotations

import os
import time
import io
import statistics
from typing import Dict, Any, List

from botocore.exceptions import ClientError
from botocore.config import Config
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from modules.utils import get_logger

LOG = get_logger()

AWS_TEXTRACT_ENABLED = os.getenv('AWS_TEXTRACT_ENABLED', 'true').lower() in ('1', 'true', 'yes')
AWS_TEXTRACT_MAX_PAGES = int(os.getenv('AWS_TEXTRACT_MAX_PAGES', '10'))
AWS_TEXTRACT_TIMEOUT = int(os.getenv('AWS_TEXTRACT_TIMEOUT', '60'))
AWS_TEXTRACT_RETRY_ATTEMPTS = int(os.getenv('AWS_TEXTRACT_RETRY_ATTEMPTS', '3'))


class TextractError(Exception):
    pass


class TextractHandler:
    def __init__(self):
        self._enabled = AWS_TEXTRACT_ENABLED
        try:
            config = Config(read_timeout=AWS_TEXTRACT_TIMEOUT, connect_timeout=AWS_TEXTRACT_TIMEOUT)
            self._client = boto3.client('textract', region_name=os.getenv('AWS_REGION'), config=config)
        except Exception:
            LOG.exception('textract_client_init_failed', exc_info=True)
            self._client = None

    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    def _parse_blocks(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        lines = []
        confidences = []
        for b in blocks:
            if b.get('BlockType') == 'LINE':
                text = b.get('Text', '').strip()
                conf = b.get('Confidence', 100.0)
                geom = b.get('Geometry', {}).get('BoundingBox', {})
                lines.append({'text': text, 'confidence': float(conf), 'bbox': geom})
                confidences.append(float(conf))
        avg_conf = float(statistics.mean(confidences) / 100.0) if confidences else 0.0
        full_text = '\n'.join([l['text'] for l in lines])
        return {'text': full_text, 'confidence': avg_conf, 'blocks': lines}

    @retry(stop=stop_after_attempt(AWS_TEXTRACT_RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(ClientError))
    def extract_text_from_s3(self, bucket: str, key: str) -> Dict[str, Any]:
        start = time.time()
        if not bucket or not key:
            raise TextractError('Bucket and key required')
        if not self.is_enabled():
            raise TextractError('Textract not enabled or client unavailable')
        try:
            resp = self._client.detect_document_text(Document={'S3Object': {'Bucket': bucket, 'Name': key}})
            # check pages if available
            pages = resp.get('DocumentMetadata', {}).get('Pages')
            if pages and pages > AWS_TEXTRACT_MAX_PAGES:
                LOG.warning('textract_page_count_exceeded', extra={'bucket': bucket, 'key': key, 'pages': pages, 'max_pages': AWS_TEXTRACT_MAX_PAGES})
                raise TextractError(f'page_count_exceeded: {pages} > {AWS_TEXTRACT_MAX_PAGES}')
            parsed = self._parse_blocks(resp.get('Blocks', []))
            total_ms = int((time.time() - start) * 1000)
            LOG.info('textract_s3_call', extra={'bucket': bucket, 'key': key, 'duration_ms': total_ms})
            return {'text': parsed['text'], 'confidence': parsed['confidence'], 'blocks': parsed['blocks'], 'page_count': pages or 1, 'service': 'textract'}
        except ClientError as e:
            LOG.exception('textract_s3_client_error', exc_info=True)
            raise TextractError(str(e))
        except Exception as e:
            LOG.exception('textract_s3_error', exc_info=True)
            raise TextractError(str(e))

    @retry(stop=stop_after_attempt(AWS_TEXTRACT_RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(ClientError))
    def extract_text_from_local(self, image_path: str) -> Dict[str, Any]:
        start = time.time()
        if not self.is_enabled():
            raise TextractError('Textract not enabled or client unavailable')
        try:
            with open(image_path, 'rb') as fh:
                data = fh.read()
            # Textract synchronous has limits; warn if large
            if len(data) > 5 * 1024 * 1024:
                LOG.warning('textract_large_payload', extra={'size_bytes': len(data)})
            resp = self._client.detect_document_text(Document={'Bytes': data})
            pages = resp.get('DocumentMetadata', {}).get('Pages')
            if pages and pages > AWS_TEXTRACT_MAX_PAGES:
                LOG.warning('textract_page_count_exceeded_local', extra={'path': image_path, 'pages': pages, 'max_pages': AWS_TEXTRACT_MAX_PAGES})
                raise TextractError(f'page_count_exceeded: {pages} > {AWS_TEXTRACT_MAX_PAGES}')
            parsed = self._parse_blocks(resp.get('Blocks', []))
            total_ms = int((time.time() - start) * 1000)
            LOG.info('textract_local_call', extra={'path': image_path, 'duration_ms': total_ms})
            return {'text': parsed['text'], 'confidence': parsed['confidence'], 'blocks': parsed['blocks'], 'page_count': pages or 1, 'service': 'textract'}
        except ClientError as e:
            LOG.exception('textract_local_client_error', exc_info=True)
            raise TextractError(str(e))
        except Exception as e:
            LOG.exception('textract_local_error', exc_info=True)
            raise TextractError(str(e))
