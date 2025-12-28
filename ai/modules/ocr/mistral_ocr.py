"""Local Mistral Pixtral OCR handler (vision-language via transformers).

Loads the Pixtral model locally (no hosted API calls) and runs inference on PIL images.
Model weights are cached under TRANSFORMERS_CACHE (default ai/models/pixtral-12b).

Provides PixtralOCR singleton with methods:
- get_instance()
- extract_text(image: PIL.Image)
- extract_text_from_path(image_path: str)
- cleanup()

Custom exceptions: PixtralModelError, PixtralInferenceError
"""
from __future__ import annotations

import json
import os
import time
import threading
import tempfile
from typing import Any, Dict, List

from PIL import Image

from modules.utils import get_logger, log_model_load
from modules.ocr.preprocess import preprocess_image, ImagePreprocessingError

LOG = get_logger()

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
except Exception:
    torch = None
    AutoModelForCausalLM = None
    AutoProcessor = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None


class PixtralModelError(Exception):
    pass


class PixtralInferenceError(Exception):
    pass


class PixtralOCR:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        raise RuntimeError('Use get_instance() to obtain PixtralOCR')

    @classmethod
    def _resolve_device(cls) -> str:
        hint = os.getenv('PIXTRAL_DEVICE', os.getenv('TROCR_DEVICE', 'auto')).lower()
        if hint == 'cpu':
            return 'cpu'
        if hint == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if hint == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cpu'

    @classmethod
    def _load_model(cls):
        if torch is None or AutoModelForCausalLM is None or AutoProcessor is None:
            raise PixtralModelError('torch/transformers not installed')

        start = time.time()
        model_id = os.getenv('PIXTRAL_MODEL', 'mistral-community/pixtral-12b')
        cache_dir = os.getenv('TRANSFORMERS_CACHE') or os.path.join(os.getcwd(), 'models', 'pixtral-12b')
        device = cls._resolve_device()
        max_length = int(os.getenv('PIXTRAL_MAX_LENGTH', '512'))
        quant = os.getenv('PIXTRAL_QUANTIZATION', 'none').lower()

        torch_dtype = torch.float16 if device == 'cuda' else torch.float32
        load_kwargs: Dict[str, Any] = {
            'cache_dir': cache_dir,
            'trust_remote_code': True,
        }

        if quant in ('4bit', '8bit'):
            load_kwargs['device_map'] = 'auto'
            load_kwargs['low_cpu_mem_usage'] = True
            if quant == '4bit':
                load_kwargs['load_in_4bit'] = True
            else:
                load_kwargs['load_in_8bit'] = True
        else:
            load_kwargs['torch_dtype'] = torch_dtype

        try:
            proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.to(device)
            model.eval()
            load_ms = int((time.time() - start) * 1000)
            log_model_load(model_id, device, load_ms)
            LOG.info('pixtral_model_loaded', extra={'model': model_id, 'device': device, 'quantization': quant})
            return {
                'model': model,
                'processor': proc,
                'device': device,
                'model_name': model_id,
                'max_length': max_length,
            }
        except Exception as e:
            LOG.exception('pixtral_model_load_failed', exc_info=True)
            raise PixtralModelError(str(e))

    @classmethod
    def get_instance(cls) -> 'PixtralOCR':
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                data = cls._load_model()
                inst = object.__new__(cls)
                inst._model = data['model']
                inst._processor = data['processor']
                inst._device = data['device']
                inst._model_name = data['model_name']
                inst._max_length = data['max_length']
                cls._instance = inst
        return cls._instance

    def _calculate_confidence(self, text: str, image: Image.Image) -> float:
        if not text or len(text.strip()) < 5:
            return 0.3
        length = len(text)
        alpha_ratio = sum(c.isalnum() for c in text) / length if length else 0.0
        space_ratio = sum(c.isspace() for c in text) / length if length else 0.0
        base_conf = 0.9
        if alpha_ratio < 0.5:
            base_conf -= 0.2
        if space_ratio > 0.4:
            base_conf -= 0.1
        if 50 <= length <= 5000:
            base_conf += 0.05
        return max(0.3, min(1.0, base_conf))

    def _parse_blocks(self, decoded: str) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        text_out = decoded
        try:
            data = json.loads(decoded)
            if isinstance(data, list):
                texts = []
                for b in data:
                    if not isinstance(b, dict):
                        continue
                    t = b.get('text', '') if isinstance(b.get('text', ''), str) else ''
                    bbox = b.get('bbox') if isinstance(b.get('bbox'), dict) else {}
                    if t:
                        texts.append(t)
                    blocks.append({'text': t, 'bbox': bbox})
                if texts:
                    text_out = '\n'.join(texts)
        except Exception:
            blocks = []
        return {'text': text_out, 'blocks': blocks}

    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        if image is None:
            raise PixtralInferenceError('No image provided')
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            prompt = (
                'You are an OCR engine. Extract text from the image and return a JSON array with '
                'the format: [{"text": "...", "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.05}}]. '
                'Coordinates must be normalized (0-1). Return ONLY valid JSON.'
            )
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors='pt',
            ).to(self._device)

            gen_kwargs = {
                'max_new_tokens': self._max_length,
                'do_sample': False,
            }
            start = time.time()
            outputs = self._model.generate(**inputs, **gen_kwargs)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            raw = decoded[0] if decoded else ''
            parsed = self._parse_blocks(raw)
            text = parsed.get('text', raw) or raw
            blocks = parsed.get('blocks', [])
            confidence = self._calculate_confidence(text, image)
            duration = int((time.time() - start) * 1000)
            LOG.info('pixtral_inference', extra={'model': self._model_name, 'device': self._device, 'duration_ms': duration, 'text_len': len(text)})
            return {
                'text': text,
                'blocks': blocks,
                'confidence': confidence,
                'model': self._model_name,
                'device': self._device,
            }
        except Exception as e:
            LOG.exception('pixtral_inference_failed', exc_info=True)
            raise PixtralInferenceError(str(e))

    def _should_preprocess(self, use_preprocessing: bool) -> bool:
        if not use_preprocessing:
            return False
        return os.getenv('PIXTRAL_PREPROCESSING_ENABLED', 'true').lower() in ('1', 'true', 'yes')

    def extract_text_from_path(self, image_path: str, use_preprocessing: bool = True) -> Dict[str, Any]:
        preprocessing_applied = False
        preprocessing_steps: List[str] = []
        try:
            if self._should_preprocess(use_preprocessing):
                try:
                    pre = preprocess_image(image_path)
                    pil_img = pre.get('processed')
                    preprocessing_steps = pre.get('steps_applied', [])
                    preprocessing_applied = True
                except ImagePreprocessingError:
                    LOG.warning('pixtral_preprocess_failed', extra={'path': image_path})
                    pil_img = Image.open(image_path).convert('RGB')
            else:
                pil_img = Image.open(image_path).convert('RGB')

            result = self.extract_text(pil_img)
            result['preprocessing_applied'] = preprocessing_applied
            result['preprocessing_steps'] = preprocessing_steps
            return result
        except Exception as e:
            LOG.exception('pixtral_extract_from_path_failed', exc_info=True)
            raise PixtralInferenceError(str(e))

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = None, use_preprocessing: bool = True) -> Dict[str, Any]:
        if convert_from_path is None:
            raise PixtralInferenceError('pdf2image is not installed; install pdf2image and poppler-utils')
        try:
            limit = max_pages or int(os.getenv('PIXTRAL_MAX_PDF_PAGES', '10'))
            pages = convert_from_path(pdf_path)
            results = []
            preprocessing_any = False
            steps_all: List[str] = []
            for idx, page_img in enumerate(pages, start=1):
                if idx > limit:
                    LOG.warning('pixtral_pdf_page_limit', extra={'limit': limit, 'skipped_from': idx})
                    break
                img_to_use = page_img
                page_steps: List[str] = []
                if self._should_preprocess(use_preprocessing):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    try:
                        page_img.save(tmp.name, format='JPEG')
                        pre = preprocess_image(tmp.name)
                        img_to_use = pre.get('processed')
                        page_steps = pre.get('steps_applied', [])
                        preprocessing_any = True
                        steps_all.extend(page_steps)
                    except Exception:
                        LOG.warning('pixtral_pdf_preprocess_failed', extra={'page': idx})
                    finally:
                        try:
                            tmp.close()
                            os.remove(tmp.name)
                        except Exception:
                            pass
                page_res = self.extract_text(img_to_use)
                page_res['page_num'] = idx
                page_res['preprocessing_steps'] = page_steps
                results.append(page_res)

            if not results:
                raise PixtralInferenceError('No pages processed from PDF')

            full_text_parts = []
            confidences = []
            for i, res in enumerate(results):
                if i > 0:
                    full_text_parts.append('\n\n--- PAGE {} ---\n\n'.format(res.get('page_num')))
                full_text_parts.append(res.get('text', ''))
                confidences.append(float(res.get('confidence', 0.0)))
            full_text = ''.join(full_text_parts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                'text': full_text,
                'confidence': avg_conf,
                'pages': [{'page_num': r.get('page_num'), 'text': r.get('text'), 'confidence': r.get('confidence'), 'blocks': r.get('blocks', [])} for r in results],
                'page_count': len(results),
                'model': self._model_name,
                'device': self._device,
                'preprocessing_applied': preprocessing_any,
                'preprocessing_steps': steps_all,
            }
        except Exception as e:
            LOG.exception('pixtral_extract_from_pdf_failed', exc_info=True)
            raise PixtralInferenceError(str(e))

    def cleanup(self):
        try:
            if hasattr(self, '_model'):
                del self._model
            if hasattr(self, '_processor'):
                del self._processor
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            PixtralOCR._instance = None
            LOG.info('pixtral_cleanup')
        except Exception:
            LOG.exception('pixtral_cleanup_failed', exc_info=True)
