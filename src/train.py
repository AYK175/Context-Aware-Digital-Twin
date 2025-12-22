# src/train.py
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from .config import *

def train():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,
    )

    dataset = load_dataset(
        "json",
        data_files=f"{PROCESSED_DIR}/training_pairs.jsonl"
    )["train"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir="outputs",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=60,
            save_steps=500,
            save_total_limit=2,
            report_to="none"
        )
    )

    trainer.train()
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    train()
