# scripts/build_rag_store.py
from src.cleaning import process_chat
from src.rag import MemoryRAG

CHAT_FILE = "data/chat.txt"

def main():
    print("Loading and cleaning chat...")
    messages = process_chat(CHAT_FILE)

    texts = []
    for m in messages:
        if m["text"] and len(m["text"]) > 10:
            texts.append(f"{m['sender']}: {m['text']}")

    print(f"Prepared {len(texts)} messages for embedding")

    rag = MemoryRAG(batch_size=128)
    rag.add_texts(texts)

    print("✅ ChromaDB memory build complete")

if __name__ == "__main__":
    main()
