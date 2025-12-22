# src/config.py

# ------------------
# Paths
# ------------------
RAW_CHAT_PATH = "data/chat.txt"
PROCESSED_DIR = "data/processed"
CHROMA_DIR = "data/chroma"
MODEL_OUTPUT_DIR = "outputs/final"

# ------------------
# Model
# ------------------
BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# ------------------
# RAG
# ------------------
CHROMA_COLLECTION_NAME = "memory"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_BATCH_SIZE = 128
RAG_TOP_K = 5

# ------------------
# Training
# ------------------
TRAIN_EPOCHS = 20
LEARNING_RATE = 2e-4
BATCH_SIZE = 5
GRAD_ACCUM_STEPS = 4
