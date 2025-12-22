# src/cleaning.py
import re
from .utils import remove_invisible_chars, safe_strip

# WhatsApp-style timestamp pattern 
CHAT_PATTERN = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4},\s'
    r'\d{1,2}:\d{2}:\d{2}\s[APM]{2})\]\s([^:]+):\s(.*)'
)

def clean_text_content(text: str) -> str | None:
    if not text:
        return None
    text = remove_invisible_chars(text)
    text = safe_strip(text)
    return text if text else None

def process_chat(file_path: str):
    """
    Parses chat exports and merges multiline messages.
    """
    messages = []
    current_msg = None

    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            cleaned_line = clean_text_content(line)
            if not cleaned_line:
                continue

            match = CHAT_PATTERN.match(cleaned_line)

            if match:
                timestamp, sender, content = match.groups()
                current_msg = {
                    "timestamp": timestamp,
                    "sender": sender,
                    "text": content
                }
                messages.append(current_msg)
            else:
                # Multiline continuation 
                if current_msg:
                    current_msg["text"] += " " + cleaned_line

    return messages



