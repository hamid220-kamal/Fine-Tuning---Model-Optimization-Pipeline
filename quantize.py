import os
from unsloth import FastLanguageModel

# ==============================================================================
# 3. HARDWARE QUANTIZATION AND EXPORT
# ==============================================================================
# This script loads the fine-tuned LoRA adapters from train.py, merges them
# with the base model, and exports the final model into deployment-ready formats.

# Paths
BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
LORA_PATH = "./lora_model"
EXPORT_DIR = "./exported_models"

def main():
    if not os.path.exists(LORA_PATH):
        print(f"Error: {LORA_PATH} not found. Please run train.py first.")
        return

    os.makedirs(EXPORT_DIR, exist_ok=True)

    print(f"Loading Base Model and merging LoRA adapters from {LORA_PATH}...")
    # FastLanguageModel automatically handles merging if we pass the lora path
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_PATH, 
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False, # We want to export the merged model, so we load in higher precision (or unsloth handles it during export)
    )

    # --------------------------------------------------------------------------
    # Export 1: 16-bit Float Hugging Face Format
    # Useful for standard cloud deployments (e.g., vLLM, standard HF pipelines)
    # --------------------------------------------------------------------------
    hf_export_path = os.path.join(EXPORT_DIR, "hf_16bit")
    print(f"\nExporting to 16-bit Hugging Face format at {hf_export_path}...")
    
    # Save the merged model directly (Unsloth handles the LoRA merging under the hood)
    model.save_pretrained_merged(hf_export_path, tokenizer, save_method="merged_16bit")
    print("16-bit export complete.")


    # --------------------------------------------------------------------------
    # Export 2: 4-bit precision GGUF format (Q4_K_M)
    # Highly optimized for CPU/Edge/Low-VRAM local deployment (llama.cpp, Ollama)
    # --------------------------------------------------------------------------
    print("\nExporting to 4-bit GGUF format (Q4_K_M)...")
    # Unsloth has a highly convenient wrapper that automatically downloads llama.cpp
    # binaries if needed and converts the model.
    gguf_export_path = os.path.join(EXPORT_DIR, "model-q4_k_m.gguf")
    
    try:
        model.save_pretrained_gguf(
            EXPORT_DIR, 
            tokenizer, 
            quantization_method="q4_k_m"
        )
        print(f"GGUF 4-bit export complete. Saved in {EXPORT_DIR}")
    except Exception as e:
        print(f"Error during GGUF export: {e}")
        print("Note: GGUF export requires cmake and a C++ compiler to build llama.cpp locally if binaries fail.")

    print("\nAll export processes completed!")
    print("To run locally with Ollama/llama.cpp, use the .gguf file.")

if __name__ == "__main__":
    main()
