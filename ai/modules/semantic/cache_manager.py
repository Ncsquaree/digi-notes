import os
import json
import hashlib
from typing import Optional, Dict, Any

from modules.utils import get_logger

LOG = get_logger()


class CacheManager:
    _instance = None

    def __init__(self):
        import redis
        host = os.getenv('REDIS_HOST', 'redis')
        port = int(os.getenv('REDIS_PORT', '6379'))
        password = os.getenv('REDIS_PASSWORD') or None
        enabled = os.getenv('REDIS_CACHE_ENABLED', 'true').lower() in ('1', 'true', 'yes')
        self.enabled = enabled
        self.ttl = int(os.getenv('REDIS_CACHE_TTL', '3600'))
        self._client = None
        if not self.enabled:
            LOG.info('redis_cache_disabled')
            return
        try:
            self._client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
            # quick ping
            self._client.ping()
            LOG.info('redis_cache_connected', extra={'host': host, 'port': port})
        except Exception as e:
            LOG.warning('redis_cache_unavailable', extra={'error': str(e)})
            self.enabled = False

    @classmethod
    def get_instance(cls) -> 'CacheManager':
        if cls._instance is None:
            cls._instance = CacheManager()
        return cls._instance

    def _key(self, parsed_content: Dict[str, Any], mode: str) -> str:
        try:
            j = json.dumps(parsed_content, sort_keys=True)
        except Exception:
            j = str(parsed_content)
        h = hashlib.sha256(j.encode()).hexdigest()[:16]
        return f"summary:{h}:{mode}"

    def get_summary(self, parsed_content: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self._client:
            return None
        key = self._key(parsed_content, mode)
        try:
            val = self._client.get(key)
            if val is None:
                LOG.info('cache_miss', extra={'key': key})
                return None
            LOG.info('cache_hit', extra={'key': key})
            return json.loads(val)
        except Exception as e:
            LOG.warning('cache_get_failed', extra={'error': str(e)})
            return None

    def set_summary(self, parsed_content: Dict[str, Any], mode: str, summary_result: Dict[str, Any], ttl: Optional[int] = None):
        if not self.enabled or not self._client:
            return
        key = self._key(parsed_content, mode)
        ttl = ttl or self.ttl
        try:
            self._client.setex(key, ttl, json.dumps(summary_result))
            LOG.info('cache_set', extra={'key': key, 'ttl': ttl})
        except Exception as e:
            LOG.warning('cache_set_failed', extra={'error': str(e)})

    def invalidate_summary(self, parsed_content: Dict[str, Any], mode: str):
        if not self.enabled or not self._client:
            return
        key = self._key(parsed_content, mode)
        try:
            self._client.delete(key)
            LOG.info('cache_invalidate', extra={'key': key})
        except Exception as e:
            LOG.warning('cache_invalidate_failed', extra={'error': str(e)})
