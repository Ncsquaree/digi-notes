"""TrOCR handler: load model once and run inference.

Provides TrOCRHandler singleton with methods:
- get_instance()
- extract_text(image: PIL.Image)
- extract_text_from_path(image_path: str)
- cleanup()

Custom exceptions: TrOCRModelError, TrOCRInferenceError
"""
from __future__ import annotations

import os
import time
import threading
import traceback
from typing import Dict, Any

from PIL import Image

from modules.utils import get_logger, log_model_load

LOG = get_logger()

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import numpy as np
except Exception:
    # Defer import errors until used; log on load
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None


class TrOCRModelError(Exception):
    pass


class TrOCRInferenceError(Exception):
    pass


class TrOCRHandler:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        raise RuntimeError('Use get_instance() to obtain TrOCRHandler')

    @classmethod
    def _load_model(cls):
        start = time.time()
        model_name = os.getenv('TROCR_MODEL', 'microsoft/trocr-large-handwritten')
        cache_dir = os.getenv('TRANSFORMERS_CACHE') or None
        device_hint = os.getenv('TROCR_DEVICE', 'auto')
        try:
            if torch is None:
                raise TrOCRModelError('torch or transformers not installed')
            # detect device
            use_cuda = False
            if device_hint != 'cpu' and torch.cuda.is_available():
                use_cuda = True
            device = 'cuda' if use_cuda else 'cpu'
            LOG.info('trocr_load_start', extra={'model': model_name, 'device': device})
            proc = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
            model.to(torch.device(device))
            model.eval()
            load_ms = int((time.time() - start) * 1000)
            log_model_load(model_name, device, load_ms)
            return {'model': model, 'processor': proc, 'device': device, 'model_name': model_name}
        except Exception as e:
            LOG.exception('trocr_model_load_failed', exc_info=True)
            raise TrOCRModelError(str(e))

    @classmethod
    def get_instance(cls) -> 'TrOCRHandler':
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
                cls._instance = inst
        return cls._instance

    def extract_text(self, image: Image.Image, max_length: int = None) -> Dict[str, Any]:
        start = time.time()
        try:
            if image is None:
                raise TrOCRInferenceError('No image provided')
            # ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            pixel_values = self._processor(images=image, return_tensors='pt').pixel_values
            if self._device == 'cuda':
                pixel_values = pixel_values.to('cuda')
            gen_kwargs = {}
            max_len = int(os.getenv('TROCR_MAX_LENGTH', '512')) if max_length is None else int(max_length)
            gen_kwargs['max_length'] = max_len
            # Request generation with scores so we can compute token-level confidences
            outputs = self._model.generate(pixel_values, **gen_kwargs, output_scores=True, return_dict_in_generate=True)
            # decoded text
            decoded = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True) if hasattr(outputs, 'sequences') else []
            text = decoded[0] if decoded else ''
            # compute confidence from per-token scores if available
            conf = 1.0
            try:
                # outputs.scores is a list of tensors (seq_len, batch_size, vocab)
                scores = getattr(outputs, 'scores', None)
                if scores and len(scores) > 0:
                    max_probs = []
                    for step_scores in scores:
                        # step_scores shape: (batch_size, vocab_size)
                        probs = torch.nn.functional.softmax(step_scores, dim=-1)
                        # take max prob for batch index 0
                        max_p = probs[0].max().item()
                        max_probs.append(max_p)
                    # average max probabilities across steps
                    conf = float(sum(max_probs) / len(max_probs)) if max_probs else 1.0
                else:
                    conf = 1.0
            except Exception:
                conf = 1.0
            dur = int((time.time() - start) * 1000)
            LOG.info('trocr_inference', extra={'model': self._model_name, 'device': self._device, 'duration_ms': dur, 'text_len': len(text)})
            return {'text': text, 'confidence': float(conf), 'model': self._model_name, 'device': self._device}
        except TrOCRInferenceError:
            raise
        except Exception as e:
            LOG.exception('trocr_inference_failed', exc_info=True)
            raise TrOCRInferenceError(str(e))

    def extract_text_from_path(self, image_path: str) -> Dict[str, Any]:
        try:
            img = Image.open(image_path)
            return self.extract_text(img)
        except Exception as e:
            LOG.exception('trocr_extract_from_path_failed', exc_info=True)
            raise TrOCRInferenceError(str(e))

    def cleanup(self):
        try:
            if hasattr(self, '_model'):
                del self._model
            if hasattr(self, '_processor'):
                del self._processor
            if 'torch' in globals() and torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            TrOCRHandler._instance = None
            LOG.info('trocr_cleanup')
        except Exception:
            LOG.exception('trocr_cleanup_failed', exc_info=True)
