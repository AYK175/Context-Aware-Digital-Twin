# src/utils.py
import re
from uuid import uuid4

INVISIBLE_CHARS = r"[\u200b\u200c\u200d\uFEFF]"

def remove_invisible_chars(text: str) -> str:
    return re.sub(INVISIBLE_CHARS, "", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def generate_ids(n: int):
    return [str(uuid4()) for _ in range(n)]
