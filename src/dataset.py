
# src/dataset.py
import json
from pathlib import Path

def build_instruction_pairs(messages, user_name: str):
    """
    Converts chat messages into instruction/response pairs
    suitable for SFT training.
    """
    pairs = []

    for i in range(len(messages) - 1):
        if messages[i]["sender"] == user_name:
            pairs.append({
                "instruction": messages[i]["text"],
                "response": messages[i + 1]["text"]
            })

    return pairs

def save_jsonl(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
