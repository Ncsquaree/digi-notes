import os
import time
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import tenacity

from modules.utils import get_logger, log_llm_call
from .llm_parser import OpenAIError

LOG = get_logger()

# Exceptions
class MindmapGeneratorError(Exception):
    pass

class MindmapAPIError(MindmapGeneratorError):
    pass

class MindmapValidationError(MindmapGeneratorError):
    pass

class MindmapTimeoutError(MindmapGeneratorError):
    pass

# Models
class MindmapNode(BaseModel):
    id: str
    label: str
    level: int = 0
    node_type: str = Field('concept')
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MindmapEdge(BaseModel):
    source: str
    target: str
    relationship: str = Field('relates_to')
    label: Optional[str] = None


class MindmapResponse(BaseModel):
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]
    root_node_id: str
    metadata: Dict[str, Any]


# Env
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
MINDMAP_MAX_TOKENS = int(os.getenv('MINDMAP_MAX_TOKENS', '2000'))
MINDMAP_TEMPERATURE = float(os.getenv('MINDMAP_TEMPERATURE', '0.4'))
OPENAI_TIMEOUT = float(os.getenv('OPENAI_TIMEOUT', '30'))
OPENAI_RETRY_ATTEMPTS = int(os.getenv('OPENAI_RETRY_ATTEMPTS', '3'))


# Singleton
class MindmapGenerator:
    _instance = None

    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise MindmapGeneratorError('OPENAI_API_KEY not set')
        self.model = OPENAI_MODEL
        self.timeout = OPENAI_TIMEOUT
        self.temperature = MINDMAP_TEMPERATURE
        LOG.info('MindmapGenerator initialized', extra={'model': self.model})

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MindmapGenerator()
        return cls._instance

    def _build_system_prompt(self) -> str:
        return (
            "You are an assistant that generates hierarchical mindmaps from structured parsed content. "
            "Return a JSON object with nodes and edges. Preserve LaTeX formulas in labels. Ensure edges reference valid node ids."
        )

    def _build_user_prompt(self, parsed_content: dict):
        return json.dumps({'parsed_content': parsed_content})

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if 'gpt-4' in self.model:
            return (prompt_tokens + completion_tokens) / 1000.0 * 0.06
        return (prompt_tokens + completion_tokens) / 1000.0 * 0.002

    @tenacity.retry(stop=tenacity.stop_after_attempt(OPENAI_RETRY_ATTEMPTS),
                    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
                    reraise=True)
    def _call_openai(self, messages: List[dict], function_def: dict = None, request_id: str = None) -> dict:
        try:
            import openai
            from openai.error import OpenAIError, Timeout
        except Exception as e:
            raise MindmapAPIError('OpenAI client not available') from e

        start = time.time()
        try:
            if function_def:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    functions=[function_def],
                    temperature=self.temperature,
                    max_tokens=MINDMAP_MAX_TOKENS,
                    timeout=self.timeout,
                )
            else:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=MINDMAP_MAX_TOKENS,
                    timeout=self.timeout,
                )
        except Timeout as e:
            LOG.exception('mindmap_openai_timeout', exc_info=True)
            raise MindmapTimeoutError(str(e)) from e
        except OpenAIError as e:
            LOG.exception('mindmap_openai_error', exc_info=True)
            raise MindmapAPIError(str(e)) from e
        duration = int((time.time() - start) * 1000)
        # normalize resp to dict for SDK compatibility
        try:
            if not isinstance(resp, dict):
                if hasattr(resp, 'to_dict_recursive'):
                    resp = resp.to_dict_recursive()
                elif hasattr(resp, 'to_dict'):
                    resp = resp.to_dict()
                elif hasattr(resp, 'to_json'):
                    resp = json.loads(resp.to_json())
                else:
                    resp = json.loads(json.dumps(resp, default=lambda o: getattr(o, '__dict__', str(o))))
        except Exception:
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

    def _assign_node_ids(self, nodes: List[dict]) -> List[dict]:
        for n in nodes:
            if 'id' not in n or not n['id']:
                n['id'] = os.urandom(6).hex()
        return nodes

    def _validate_edges(self, nodes: List[dict], edges: List[dict]) -> List[dict]:
        node_ids = {n['id'] for n in nodes}
        valid = []
        for e in edges:
            if e.get('source') in node_ids and e.get('target') in node_ids:
                valid.append(e)
            else:
                LOG.warning('mindmap_edge_invalid', extra={'edge': e})
        return valid

    def generate_mindmap(self, parsed_content: dict, options: Optional[dict] = None, request_id: str = None) -> MindmapResponse:
        messages = [
            {'role': 'system', 'content': self._build_system_prompt()},
            {'role': 'user', 'content': self._build_user_prompt(parsed_content)},
        ]
        function_def = {
            'name': 'mindmap_response',
            'description': 'Return mindmap JSON with nodes and edges',
            'parameters': {
                'type': 'object',
                'properties': {
                    'nodes': {'type': 'array', 'items': {'type': 'object'}},
                    'edges': {'type': 'array', 'items': {'type': 'object'}},
                    'root_node_id': {'type': 'string'},
                    'metadata': {'type': 'object'}
                },
                'required': ['nodes', 'edges', 'root_node_id']
            }
        }
        resp = self._call_openai(messages, function_def=function_def, request_id=request_id)
        try:
            choices = resp.get('choices', []) if isinstance(resp, dict) else []
            if choices and 'message' in choices[0] and choices[0]['message'].get('function_call'):
                args_text = choices[0]['message']['function_call'].get('arguments', '{}')
            elif choices and 'message' in choices[0] and choices[0]['message'].get('content'):
                args_text = choices[0]['message'].get('content', '{}')
            else:
                args_text = '{}'
            data = json.loads(args_text)
        except Exception as e:
            LOG.exception('mindmap_response_parse_failed', exc_info=True)
            raise MindmapAPIError('Failed to parse model response') from e

        try:
            nodes_raw = data.get('nodes', [])
            edges_raw = data.get('edges', [])
            nodes_assigned = self._assign_node_ids(nodes_raw)
            edges_valid = self._validate_edges(nodes_assigned, edges_raw)
            nodes = [MindmapNode(**n) for n in nodes_assigned]
            edges = [MindmapEdge(**e) for e in edges_valid]
            metadata = data.get('metadata', {}) or {}
            metadata.setdefault('model_used', self.model)
            root_node_id = data.get('root_node_id') or (nodes[0].id if nodes else '')
            mr = MindmapResponse(nodes=nodes, edges=edges, root_node_id=root_node_id, metadata=metadata)
            return mr
        except Exception as e:
            LOG.exception('mindmap_validation_failed', exc_info=True)
            raise MindmapValidationError(str(e)) from e


# convenience
def generate_mindmap(parsed_content: dict, options: Optional[dict] = None, request_id: str = None) -> Dict[str, Any]:
    g = MindmapGenerator.get_instance()
    res = g.generate_mindmap(parsed_content, options=options, request_id=request_id)
    return res.model_dump()
