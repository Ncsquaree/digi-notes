import os
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
