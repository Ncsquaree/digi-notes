import os
import json
import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, List

try:
    import redis
except Exception:
    redis = None

from modules.utils import get_logger

LOG = get_logger()

TASK_TTL_SECONDS = int(os.getenv('TASK_TTL_SECONDS', os.getenv('TASK_TTL_SECONDS', '86400')))
TASK_MANAGER_ENABLED = os.getenv('TASK_MANAGER_ENABLED', 'true').lower() in ('1', 'true', 'yes')
REDIS_URL = os.getenv('REDIS_URL', None)


class TaskStatus(str, Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    PARTIAL_SUCCESS = 'partial_success'


class TaskManager:
    _instance = None

    def __init__(self):
        self._use_redis = False
        self._client = None
        self._in_memory: Dict[str, Dict[str, Any]] = {}
        try:
            if redis is not None and REDIS_URL:
                self._client = redis.from_url(REDIS_URL, decode_responses=True)
                # test ping
                self._client.ping()
                self._use_redis = True
                LOG.info('TaskManager using Redis', extra={'redis_url': REDIS_URL})
            elif redis is not None:
                # try default localhost
                self._client = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=int(os.getenv('REDIS_PORT', '6379')), password=os.getenv('REDIS_PASSWORD') or None, decode_responses=True)
                self._client.ping()
                self._use_redis = True
                LOG.info('TaskManager using Redis default host', extra={})
            else:
                LOG.warning('redis library not available, falling back to in-memory TaskManager')
        except Exception as e:
            LOG.warning('Redis not available for TaskManager, using in-memory store', extra={'error': str(e)})
            self._use_redis = False
            self._client = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TaskManager()
        return cls._instance

    def _key(self, task_id: str) -> str:
        return f'task:{task_id}'

    def create_task(self, task_id: str, user_id: str, note_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = int(time.time())
        obj = {
            'task_id': task_id,
            'user_id': user_id,
            'note_id': note_id,
            'status': TaskStatus.PENDING.value,
            'progress_pct': 0,
            'current_step': '',
            'steps_completed': [],
            'steps_failed': [],
            'result': None,
            'error_message': None,
            'metadata': metadata or {},
            'created_at': now,
            'updated_at': now,
        }
        try:
            if self._use_redis and self._client:
                self._client.set(self._key(task_id), json.dumps(obj))
                self._client.expire(self._key(task_id), TASK_TTL_SECONDS)
            else:
                self._in_memory[task_id] = obj
        except Exception as e:
            LOG.warning('task_create_failed', extra={'task_id': task_id, 'error': str(e)})
        LOG.info('task_created', extra={'task_id': task_id, 'user_id': user_id, 'note_id': note_id})
        return obj

    def _save(self, task_id: str, obj: Dict[str, Any]):
        obj['updated_at'] = int(time.time())
        try:
            if self._use_redis and self._client:
                self._client.set(self._key(task_id), json.dumps(obj))
                self._client.expire(self._key(task_id), TASK_TTL_SECONDS)
            else:
                self._in_memory[task_id] = obj
        except Exception as e:
            LOG.warning('task_save_failed', extra={'task_id': task_id, 'error': str(e)})

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self._use_redis and self._client:
                raw = self._client.get(self._key(task_id))
                if not raw:
                    return None
                return json.loads(raw)
            return self._in_memory.get(task_id)
        except Exception as e:
            LOG.warning('task_get_failed', extra={'task_id': task_id, 'error': str(e)})
            return self._in_memory.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus, progress_pct: Optional[int] = None, current_step: Optional[str] = None, error_msg: Optional[str] = None):
        task = self.get_task(task_id)
        if not task:
            LOG.warning('update_status_task_not_found', extra={'task_id': task_id})
            return None
        task['status'] = status.value if isinstance(status, TaskStatus) else status
        if progress_pct is not None:
            task['progress_pct'] = int(progress_pct)
        if current_step is not None:
            task['current_step'] = current_step
        if error_msg is not None:
            task['error_message'] = error_msg
        self._save(task_id, task)
        LOG.info('task_status_updated', extra={'task_id': task_id, 'status': task['status'], 'progress_pct': task['progress_pct'], 'current_step': task['current_step']})
        return task

    def update_progress(self, task_id: str, step_name: str, progress_pct: int, step_result: Optional[Dict[str, Any]] = None):
        task = self.get_task(task_id)
        if not task:
            LOG.warning('update_progress_task_not_found', extra={'task_id': task_id})
            return None
        task['current_step'] = step_name
        task['progress_pct'] = int(progress_pct)
        if step_result is not None:
            task.setdefault('result', {})
            task['result'].setdefault(step_name, step_result)
        self._save(task_id, task)
        LOG.info('task_progress_updated', extra={'task_id': task_id, 'step': step_name, 'progress_pct': progress_pct})
        return task

    def mark_step_complete(self, task_id: str, step_name: str, result_data: Optional[Dict[str, Any]] = None):
        task = self.get_task(task_id)
        if not task:
            LOG.warning('mark_step_complete_task_not_found', extra={'task_id': task_id})
            return None
        if step_name not in task.get('steps_completed', []):
            task.setdefault('steps_completed', []).append(step_name)
        if result_data is not None:
            task.setdefault('result', {})
            task['result'][step_name] = result_data
        self._save(task_id, task)
        LOG.info('task_step_completed', extra={'task_id': task_id, 'step': step_name})
        return task

    def mark_step_failed(self, task_id: str, step_name: str, error_msg: str):
        task = self.get_task(task_id)
        if not task:
            LOG.warning('mark_step_failed_task_not_found', extra={'task_id': task_id})
            return None
        # Ensure steps_failed is a list of structured objects and avoid duplicates
        task.setdefault('steps_failed', [])
        exists = False
        for entry in task.get('steps_failed', []):
            try:
                if isinstance(entry, dict) and entry.get('step') == step_name:
                    exists = True
                    break
                if isinstance(entry, str) and entry == step_name:
                    exists = True
                    break
            except Exception:
                continue
        if not exists:
            task['steps_failed'].append({'step': step_name, 'error': error_msg})
        # store the latest error_message for quick access, and also keep per-step details
        task['error_message'] = error_msg
        self._save(task_id, task)
        LOG.warning('task_step_failed', extra={'task_id': task_id, 'step': step_name, 'error': error_msg})
        return task

    def complete_partial(self, task_id: str, final_result: Optional[Dict[str, Any]] = None):
        """Mark a task as partial success (some steps failed but overall pipeline produced usable output)."""
        task = self.get_task(task_id)
        if not task:
            LOG.warning('complete_partial_not_found', extra={'task_id': task_id})
            return None
        task['status'] = TaskStatus.PARTIAL_SUCCESS.value
        task['progress_pct'] = 100
        task['current_step'] = ''
        if final_result is not None:
            task['result'] = final_result
        self._save(task_id, task)
        LOG.info('task_partial_completed', extra={'task_id': task_id})
        return task

    def complete_task(self, task_id: str, final_result: Optional[Dict[str, Any]] = None):
        task = self.get_task(task_id)
        if not task:
            LOG.warning('complete_task_not_found', extra={'task_id': task_id})
            return None
        task['status'] = TaskStatus.COMPLETED.value
        task['progress_pct'] = 100
        task['current_step'] = ''
        if final_result is not None:
            task['result'] = final_result
        self._save(task_id, task)
        LOG.info('task_completed', extra={'task_id': task_id})
        return task

    def fail_task(self, task_id: str, error_msg: str):
        task = self.get_task(task_id)
        if not task:
            LOG.warning('fail_task_not_found', extra={'task_id': task_id})
            return None
        task['status'] = TaskStatus.FAILED.value
        task['error_message'] = error_msg
        task['progress_pct'] = task.get('progress_pct', 0)
        self._save(task_id, task)
        LOG.error('task_failed', extra={'task_id': task_id, 'error': error_msg})
        return task
