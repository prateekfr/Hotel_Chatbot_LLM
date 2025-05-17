import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


def main():
    # ========== Settings ==========
    MODEL_NAME = "tiiuae/Falcon3-1B-Instruct"
    DATA_PATH = "data/processed/falcon_finetune_dataset.jsonl"
    OUTPUT_DIR = "model/falcon-lora-finetuned"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ========== 4-bit Quantization ==========
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ========== Model ==========
    print("Loading model to GPU...")
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # ========== LoRA ==========
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ========== Dataset ==========
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # ========== Data Collator ==========
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ========== Training Arguments ==========
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,  # Recommended for Windows
        dataloader_pin_memory=True,
        report_to="none",
        disable_tqdm=False
    )

    # ========== Trainer ==========
    print("🚀 Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # ========== Save ==========
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
