import sqlite3
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import re
import os

DB_PATH = "rlaif_waiting_room.db"
OUTPUT_DIR = "./notebooks/data/rlaif_finetune"

def extract_and_format_data():
    print("🔍 Connecting to RLAIF Database...")
    
    if not os.path.exists(DB_PATH):
        print("⚠️ No database found. The Waiting Room is empty.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Fetch all logged interactions
    df = pd.read_sql_query("SELECT * FROM hard_negatives", conn)
    conn.close()

    if df.empty:
        print("📭 No hard negatives logged yet. Keep testing the UI!")
        return

    print(f"📥 Found {len(df)} flagged interactions. Processing...")

    training_texts = []
    training_labels = []
    task_ids = []

    for _, row in df.iterrows():
        original_text = row['user_input']
        feedback = row['gemini_feedback']

        # 1. The Negative Example (The Hallucination)
        # DeBERTa scored this high, but Gemini flagged it. We force the label to 0.2.
        training_texts.append(original_text)
        training_labels.append(0.2)
        task_ids.append(0) # Task 0 is Quality Score

        # 2. The Positive Example (The Coach's Rewrite)
        # We use Regex to extract the sentence Gemini wrote under "Coach's Rewrite"
        rewrite_match = re.search(r'### 💡 Coach\'s Rewrite\n(.*)', feedback)
        if rewrite_match:
            gold_standard_text = rewrite_match.group(1).strip()
            if gold_standard_text:
                training_texts.append(gold_standard_text)
                training_labels.append(0.9) # Force a high label for the good rewrite
                task_ids.append(0)

    print(f"✨ Generated {len(training_texts)} new training examples!")

    # =========================
    # Tokenization for PyTorch
    # =========================
    print("⚙️ Tokenizing data for DeBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Create a Hugging Face Dataset
    raw_dataset = Dataset.from_dict({
        "text": training_texts,
        "label": training_labels,
        "task_id": task_ids
    })

    # Tokenization function mapping
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=256
        )

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    
    # Remove the raw text column so it matches your train.py expected format
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenized_dataset.save_to_disk(OUTPUT_DIR)
    
    print(f"✅ RLAIF Dataset successfully saved to '{OUTPUT_DIR}'")
    print("🚀 Ready to merge with your main dataset and run train.py!")

if __name__ == "__main__":
    extract_and_format_data()