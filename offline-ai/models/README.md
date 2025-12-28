# TFLite Models

This directory contains pre-trained TensorFlow Lite models for offline AI inference.

## Models

- **mobilebert-squad.tflite** (~25MB): MobileBERT fine-tuned on SQuAD v1.1 for extractive question answering
- **use-lite.tflite** (~50MB): Universal Sentence Encoder Lite for 512-dimensional embeddings

## Download

Run the automated downloader to fetch models:

```bash
python scripts/download_models.py
```

## Source & License

- MobileBERT-SQuAD: [TensorFlow Hub](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1) - Apache 2.0
- Universal Sentence Encoder Lite: [TensorFlow Hub](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1) - Apache 2.0

Both models are free and open-source for commercial and personal use.
