import os
import sys
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

errors = []
warnings = []

import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--strict', '-s', action='store_true', help='Convert warnings to failures')
args = parser.parse_args()
STRICT = args.strict

required = {
    'server': ['ENVIRONMENT', 'HOST', 'PORT'],
    'aws': ['AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_S3_BUCKET'],
    'openai': ['OPENAI_API_KEY', 'OPENAI_MODEL'],
    'trocr': ['TROCR_MODEL', 'TROCR_DEVICE'],
    'database': ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
}

def check_presence(cat, keys):
    for k in keys:
        if not os.getenv(k):
            errors.append(f"{cat}: Missing {k}")

for cat, keys in required.items():
    check_presence(cat, keys)

# Format checks
try:
    port = int(os.getenv('PORT', '0'))
    if port < 1 or port > 65535:
        errors.append('PORT must be integer between 1 and 65535')
except ValueError:
    errors.append('PORT must be an integer')

openai_key = os.getenv('OPENAI_API_KEY', '')
if openai_key and not openai_key.startswith('sk-'):
    warnings.append('OPENAI_API_KEY does not start with sk-; verify provider')

t_device = os.getenv('TROCR_DEVICE', 'cpu')
if t_device not in ('cpu', 'cuda'):
    errors.append("TROCR_DEVICE must be 'cpu' or 'cuda'")

# Optional: Pixtral vision model (local)
pixtral_enabled = os.getenv('PIXTRAL_ENABLED', 'false').lower() in ('1','true','yes')
if pixtral_enabled:
    model_id = os.getenv('PIXTRAL_MODEL', 'mistral-community/pixtral-12b')
    if not model_id:
        errors.append('pixtral: PIXTRAL_MODEL not set')
    device_hint = os.getenv('PIXTRAL_DEVICE', 'auto').lower()
    if device_hint not in ('cpu', 'cuda', 'auto'):
        errors.append("pixtral: PIXTRAL_DEVICE must be cpu|cuda|auto")
    quant = os.getenv('PIXTRAL_QUANTIZATION', 'none').lower()
    if quant not in ('none', '4bit', '8bit'):
        errors.append("pixtral: PIXTRAL_QUANTIZATION must be none|4bit|8bit")

# Validate numeric env ranges
try:
    max_image_mb = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))
    if max_image_mb < 1 or max_image_mb > 200:
        errors.append('MAX_IMAGE_SIZE_MB must be between 1 and 200')
except ValueError:
    errors.append('MAX_IMAGE_SIZE_MB must be an integer')

try:
    conf = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.7'))
    if conf < 0.0 or conf > 1.0:
        errors.append('OCR_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0')
except ValueError:
    errors.append('OCR_CONFIDENCE_THRESHOLD must be a float')

region = os.getenv('AWS_REGION', '')
if region and not re.match(r'^[a-z]{2}-[a-z]+-\d$', region):
    warnings.append('AWS_REGION may not match common region pattern; verify value')

# AWS connectivity check (S3 head-bucket) - best effort
try:
    import boto3
    s3 = boto3.client('s3',
                      region_name=os.getenv('AWS_REGION'),
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    bucket = os.getenv('AWS_S3_BUCKET')
    if bucket:
        try:
            s3.head_bucket(Bucket=bucket)
            print('S3: bucket accessible')
        except Exception as e:
            errors.append(f'S3 access failed: {e}')
except Exception:
    warnings.append('boto3 not installed or AWS credentials unavailable; skipping S3 check')

# OpenAI check - use 1.x client API (OpenAI)
try:
    from openai import OpenAI
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            client.models.list()
            print('OpenAI: API reachable')
        except Exception as e:
            warnings.append(f'OpenAI check failed: {e}')
except Exception:
    warnings.append('openai package not available or client error; skipping OpenAI check')

# DB check
try:
    import psycopg2
    conn = psycopg2.connect(host=os.getenv('DB_HOST'),
                            port=int(os.getenv('DB_PORT', '5432')),
                            dbname=os.getenv('DB_NAME'),
                            user=os.getenv('DB_USER'),
                            password=os.getenv('DB_PASSWORD'))
    cur = conn.cursor()
    cur.execute('SELECT 1')
    print('Postgres: OK')
    cur.close(); conn.close()
except Exception as e:
    errors.append(f'Postgres connection failed: {e}')

# Redis check (optional)
if os.getenv('REDIS_HOST'):
    try:
        import redis
        r = redis.Redis(host=os.getenv('REDIS_HOST'), port=int(os.getenv('REDIS_PORT', '6379')), password=os.getenv('REDIS_PASSWORD') or None)
        if r.ping():
            print('Redis: OK')
    except Exception as e:
        warnings.append(f'Redis check failed: {e}')

# Neptune health check (if configured)
if os.getenv('NEPTUNE_ENDPOINT'):
    try:
        # simple socket/connectivity check
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(os.getenv('NEPTUNE_ENDPOINT'))
        host = parsed.hostname
        port = parsed.port or 8182
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        print('Neptune endpoint reachable')
    except Exception as e:
        warnings.append(f'Neptune connectivity check failed: {e}')

# Model cache dir check
cache_dir = Path(os.getenv('TRANSFORMERS_CACHE', '/app/models'))
try:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(cache_dir, os.W_OK):
        errors.append(f'Model cache path not writable: {cache_dir}')
    else:
        print(f'Model cache: {cache_dir}')
        # check free disk space for model cache
        try:
            total, used, free = shutil.disk_usage(str(cache_dir))
            free_mb = free // (1024 * 1024)
            print(f'Model cache free space: {free_mb} MB')
            if free_mb < 1024:
                warnings.append('Model cache free space is low (<1GB)')
        except Exception:
            warnings.append('Unable to determine disk usage for model cache')
except Exception as e:
    errors.append(f'Failed to verify/create model cache dir: {e}')

if errors:
    print('\nENV validation failed:')
    for e in errors:
        print(' -', e)
    sys.exit(1)

if warnings:
    print('\nWarnings:')
    for w in warnings:
        print(' -', w)
    if STRICT:
        print('\nStrict mode enabled: treating warnings as errors')
        sys.exit(1)

print('\nAll critical validations passed')
sys.exit(0)
