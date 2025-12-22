
# 🧠 Context-Aware Digital Twin

A **personalized digital twin** built using **LLaMA 3.1**, **Unsloth fine-tuning**, and **Retrieval-Augmented Generation (RAG)**.  
The system learns a user’s communication style from historical conversations and augments responses with long-term semantic memory stored in a vector database.

This project transitions from a research notebook into a **production-ready, modular ML system**.

---

## 🚀 Key Features

- **Persona Fine-Tuning**
  - Fine-tuned LLaMA 3.1 using conversational instruction–response pairs
  - Preserves tone, style, and linguistic patterns

- **Long-Term Memory (RAG)**
  - Semantic memory storage using **ChromaDB**
  - Sentence-Transformer embeddings for similarity search
  - Contextual memory injection during inference

- **Efficient Training**
  - 4-bit quantization
  - Unsloth + TRL SFTTrainer
  - GPU-friendly training pipeline

- **Modular Architecture**
  - Clean separation of data processing, training, memory, and inference
  - Notebook-free core logic

---

## 🏗️ Project Structure

```text
digital-twin/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── chat_exports/
│   ├── processed/
│   │   ├── cleaned_messages.jsonl
│   │   └── training_pairs.jsonl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── cleaning.py
│   ├── dataset.py
│   ├── train.py
│   ├── rag.py
│   ├── inference.py
│   └── utils.py
│
├── notebooks/
│   └── GPUZZv2.ipynb
│
└── scripts/
    ├── build_rag_store.py
    └── run_chat.py


                ┌─────────────────────┐
                │   Raw Chat Logs     │
                └─────────┬───────────┘
                          │
                  Cleaning & Merging
                          │
                ┌─────────▼───────────┐
                │ Instruction Pairs   │
                └─────────┬───────────┘
                          │
                 Fine-Tuning (Unsloth)
                          │
          ┌───────────────▼───────────────┐
          │     Digital Twin LLM          │
          └───────────────┬───────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │   ChromaDB Semantic Memory (RAG)  │
        └─────────────────┬─────────────────┘
                          │
                    Final Response
