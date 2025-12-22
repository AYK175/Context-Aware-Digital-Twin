# src/inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.rag import MemoryRAG

MODEL_PATH = "outputs/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

rag = MemoryRAG()

def final(message: str) -> str:
    memories = rag.retrieve(message)

    context = "\n".join(memories)

    prompt = f"""
### Instruction:
You are a clone. Respond naturally using your personality and memories.

Relevant memories:
{context}

User: {message}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.4,
        repetition_penalty=1.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0].split("### Response:")[-1].strip()



