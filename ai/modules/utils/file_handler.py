import os
import pathlib
import tempfile
import mimetypes
import hashlib
from urllib.parse import urlparse
import logging

import boto3
from botocore.exceptions import ClientError

LOG = logging.getLogger(__name__)

TEMP_DIR = os.getenv('S3_TEMP_DIR', '/app/data/temp')
MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = os.getenv('SUPPORTED_IMAGE_FORMATS', 'jpg,jpeg,png,pdf').split(',')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')


class FileDownloadError(Exception):
    """Raised when a file cannot be downloaded from S3."""


class FileHandler:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        pathlib.Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        LOG.info('FileHandler initialized', extra={'temp_dir': TEMP_DIR})

    def parse_s3_url(self, s3_url: str):
        if not s3_url:
            raise ValueError('Empty S3 URL')
        parsed = urlparse(s3_url)
        # s3://bucket/key
        if parsed.scheme == 's3':
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            return bucket, key
        # https://bucket.s3.region.amazonaws.com/key
        if parsed.scheme in ('http', 'https'):
            host_parts = parsed.netloc.split('.')
            if len(host_parts) >= 3 and host_parts[1] == 's3':
                bucket = host_parts[0]
                key = parsed.path.lstrip('/')
                return bucket, key
            # fallback: https://s3.region.amazonaws.com/bucket/key
            parts = parsed.path.lstrip('/').split('/', 1)
            if len(parts) == 2:
                bucket, key = parts[0], parts[1]
                return bucket, key
        raise ValueError('Unsupported S3 URL format')

    def download_from_s3_by_key(self, key: str, local_path: str = None) -> str:
        if not AWS_S3_BUCKET:
            raise ValueError('AWS_S3_BUCKET not configured')
        if not key:
            raise ValueError('Empty S3 key')
        if not local_path:
            # preserve extension from S3 key when creating temp file
            suffix = pathlib.Path(key).suffix or ''
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR)
            local_path = tmp.name
            tmp.close()
        try:
            self.s3.download_file(AWS_S3_BUCKET, key, local_path)
            size = os.path.getsize(local_path)
            LOG.info('Downloaded from s3', extra={'bucket': AWS_S3_BUCKET, 'key': key, 'size': size})
            valid, err = self.validate_file(local_path)
            if not valid:
                self.cleanup_temp_file(local_path)
                raise ValueError(err)
            return local_path
        except ClientError:
            LOG.exception('S3 download failed', exc_info=True)
            if os.path.exists(local_path):
                self.cleanup_temp_file(local_path)
            # raise a higher-level error for callers to handle
            raise FileDownloadError(f'Failed to download object from S3: s3://{AWS_S3_BUCKET}/{key}')

    def download_from_s3(self, s3_url: str, local_path: str = None) -> str:
        bucket, key = self.parse_s3_url(s3_url)
        if bucket != AWS_S3_BUCKET:
            # allow cross-bucket but warn
            LOG.warning('Downloading from non-default bucket', extra={'bucket': bucket})
        if not local_path:
            # preserve extension from S3 key when creating temp file
            suffix = pathlib.Path(key).suffix or ''
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR)
            local_path = tmp.name
            tmp.close()
        try:
            self.s3.download_file(bucket, key, local_path)
            size = os.path.getsize(local_path)
            LOG.info('Downloaded from s3 url', extra={'url': s3_url, 'key': key, 'size': size})
            valid, err = self.validate_file(local_path)
            if not valid:
                self.cleanup_temp_file(local_path)
                raise ValueError(err)
            return local_path
        except ClientError:
            LOG.exception('S3 download failed', exc_info=True)
            if os.path.exists(local_path):
                self.cleanup_temp_file(local_path)
            raise FileDownloadError(f'Failed to download object from S3 URL: {s3_url}')

    def validate_file(self, file_path: str):
        if not os.path.exists(file_path):
            return False, 'File does not exist'
        size = os.path.getsize(file_path)
        if size > MAX_IMAGE_SIZE_BYTES:
            return False, f'File too large (max {MAX_IMAGE_SIZE_MB} MB)'
        ext = pathlib.Path(file_path).suffix.lstrip('.').lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            return False, 'Unsupported file extension'
        mime, _ = mimetypes.guess_type(file_path)
        # optional mime check
        if mime and mime.split('/')[-1] not in SUPPORTED_IMAGE_FORMATS and mime not in ('application/pdf',):
            # allow application/pdf explicitly
            return False, 'Unsupported MIME type'
        return True, None

    def get_file_info(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        size = os.path.getsize(file_path)
        ext = pathlib.Path(file_path).suffix.lower()
        mime, _ = mimetypes.guess_type(file_path)
        return {
            'size_bytes': size,
            'size_mb': size / (1024 * 1024),
            'extension': ext,
            'mime_type': mime,
            'filename': pathlib.Path(file_path).name
        }

    def cleanup_temp_file(self, file_path: str):
        try:
            if file_path and os.path.exists(file_path) and os.path.commonpath([os.path.abspath(file_path), os.path.abspath(TEMP_DIR)]) == os.path.abspath(TEMP_DIR):
                os.remove(file_path)
                LOG.info('Removed temp file', extra={'file': file_path})
        except Exception:
            LOG.exception('Failed to cleanup temp file', exc_info=True)
