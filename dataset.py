import json
import random
import uuid
from datetime import datetime, timedelta
from datasets import Dataset

# ==============================================================================
# 1. SYNTHETIC DATASET GENERATION & PROMPT FORMATTING
# ==============================================================================
# This script programmatically mocks 1,000 conversational training pairs and
# maps them into standard ChatML / Hugging Face formats optimized for JSON extraction.

NUM_SAMPLES = 1000

# Simulated noisy data pools
NAMES = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown", "Charlie Davis", "Emily Clark", "Frank White", "Grace Lee", "UNKNOWN"]
ISSUES = [
    ("billing", ["I was charged twice on my card!", "Why is my invoice so high?", "Refund my money now.", "I see an extra charge on my statement."]),
    ("technical", ["The app keeps crashing when I login.", "I can't reset my password, no email arrives.", "Error 500 on the dashboard.", "My sync is broken."]),
    ("account", ["How do I change my email?", "Delete my account immediately.", "I want to upgrade my plan.", "Add a new user to my team."])
]
STATUSES = [("resolved", True), ("unresolved", False)]

def generate_mock_transcript() -> dict:
    """Generates a messy, unstructured chat transcript and its strict JSON target."""
    name = random.choice(NAMES)
    issue_type, issue_prompts = random.choice(ISSUES)
    transcript_text = random.choice(issue_prompts)
    status_label, is_resolved = random.choice(STATUSES)
    
    # Generate some random dates to embed in the messy text
    date1 = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
    date2 = (datetime.now() - timedelta(days=random.randint(31, 60))).strftime("%m/%d/%Y")
    
    # Add noise to the input
    messy_input = (
        f"User [{name}] joined the chat at {date1}. "
        f"They said: '{transcript_text}' "
        f"Previous ticket from {date2} was marked as {status_label}. "
        f"Agent notes: Issue seems to be {issue_type} related. "
        f"Customer sentiment: {'angry' if not is_resolved else 'neutral'}."
    )
    
    # Target exact JSON schema
    target_json = {
        "customer_name": name if name != "UNKNOWN" else None,
        "issue_priority": "high" if "now" in transcript_text.lower() or "crash" in transcript_text.lower() else "normal",
        "resolved": is_resolved,
        "extracted_dates": [date1, date2]
    }
    
    return {
        "messy_transcript": messy_input,
        "target_json": json.dumps(target_json, separators=(',', ':')) # Strict formatting, no spaces
    }

def format_prompt(sample: dict) -> dict:
    """
    Formats the sample into ChatML / Alpaca style suitable for instruction tuning.
    We use a system prompt to strictly enforce JSON adherence.
    """
    system_prompt = (
        "You are a highly specialized data extraction expert. "
        "Extract the required information from the messy transcript into a strict JSON object. "
        "Do not include any other text, markdown formatting, or explanations. "
        "Schema: {\"customer_name\": str|null, \"issue_priority\": \"high\"|\"normal\", \"resolved\": bool, \"extracted_dates\": list[str]}"
    )
    
    user_prompt = f"Extract JSON from this transcript:\n\n{sample['messy_transcript']}"
    
    # Hugging Face TRL expects a "text" field for standard causal LM training
    # Format: <|system|>...</s><|user|>...</s><|assistant|>...</s>
    formatted_text = (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\n{user_prompt}</s>\n"
        f"<|assistant|>\n{sample['target_json']}</s>"
    )
    
    return {"text": formatted_text}

def build_dataset() -> Dataset:
    """Builds and returns the Hugging Face Dataset."""
    print(f"Generating {NUM_SAMPLES} synthetic training samples...")
    raw_data = [generate_mock_transcript() for _ in range(NUM_SAMPLES)]
    
    dataset = Dataset.from_list(raw_data)
    
    print("Formatting dataset for Instruction Tuning...")
    formatted_dataset = dataset.map(format_prompt, remove_columns=["messy_transcript", "target_json"])
    
    print(f"Dataset ready. Sample entry:\n{formatted_dataset[0]['text']}")
    return formatted_dataset

if __name__ == "__main__":
    # Save the dataset to disk for the training script
    ds = build_dataset()
    ds.save_to_disk("./json_extraction_dataset")
    print("Dataset saved to ./json_extraction_dataset")
