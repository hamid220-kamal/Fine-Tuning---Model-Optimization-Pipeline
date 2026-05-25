import os
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ==============================================================================
# 2. MEMORY-OPTIMIZED QLORA FINE-TUNING VIA UNSLOTH
# ==============================================================================
# This script fine-tunes Llama-3 (8B) to output strict JSON using Unsloth's
# optimized 4-bit loading and QLoRA, making it feasible for consumer GPUs.

# Configuration
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Pre-quantized for ultra-fast download and low VRAM
MAX_SEQ_LENGTH = 2048 # Adjust based on average transcript length
DTYPE = None # None auto-detects fp16 vs bf16 based on GPU architecture
LOAD_IN_4BIT = True # Enforce 4-bit loading

def get_bfloat16_support():
    """Auto-detect if GPU supports bfloat16 for better precision stability."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def main():
    print("Loading Dataset...")
    # Load dataset generated in dataset.py
    try:
        dataset = load_from_disk("./json_extraction_dataset")
    except FileNotFoundError:
        print("Dataset not found! Please run dataset.py first.")
        return

    # 1. Load Model & Tokenizer with Unsloth optimization
    print(f"Loading Base Model ({MODEL_NAME}) in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # 2. Apply QLoRA (Low-Rank Adaptation)
    # We target all projection modules for maximum learning capacity in the adapters
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0, # Unsloth optimizes dropout=0 heavily
        bias="none",
        use_gradient_checkpointing="unsloth", # 30% less VRAM usage
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Setup SFT Trainer (Supervised Fine-Tuning)
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False, # Set to True for speed if sequences are short, but JSON needs exact boundaries
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4, # Effective batch size = 2 * 4 = 8
            warmup_steps=10,
            max_steps=100, # Set max_steps for a quick run, or num_train_epochs=1 for full
            learning_rate=2e-4, # Standard QLoRA LR
            fp16=not get_bfloat16_support(),
            bf16=get_bfloat16_support(),
            logging_steps=1,
            optim="adamw_8bit", # 8-bit optimizer saves VRAM
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none" # Disable wandb/tensorboard for clean local runs
        ),
    )

    # 4. Train the Model
    print("Starting Fine-Tuning...")
    trainer_stats = trainer.train()
    print("Training Complete!")
    print(f"Time Taken: {trainer_stats.metrics.get('train_runtime', 0):.2f} seconds")

    # 5. Save the LoRA Adapters locally
    save_path = "./lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA adapters saved successfully to {save_path}")

if __name__ == "__main__":
    main()
