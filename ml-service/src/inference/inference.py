import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

# Make sure to import the PEFT components!
from peft import LoraConfig, get_peft_model, TaskType

# Assuming this is your custom module
from fallacy_detector import detect_fallacy

# =========================
# 0. Hugging Face Login
# =========================

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("Logging into Hugging Face Hub...")
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env file. Running unauthenticated.")


# =========================
# 1. Multi-Task Model
# =========================
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Must match the training script exactly!
        base_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.1,
            bias="none"
        )

        self.encoder = get_peft_model(base_model, lora_config)

        hidden = self.encoder.config.hidden_size

        self.quality_head = nn.Linear(hidden, 1)
        self.component_head = nn.Linear(hidden, 3)
        self.stance_head = nn.Linear(hidden, 2) # 2 classes for PRO/CON

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 🔥 Fixed: Mean Pooling matches train.py
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1)
        
        # Cast to float32 to prevent dtype mismatch errors
        pooled = pooled.to(torch.float32)

        # 🔥 Fixed: Added squeeze(-1)
        quality_logits = self.quality_head(pooled).squeeze(-1)
        component_logits = self.component_head(pooled)
        stance_logits = self.stance_head(pooled)

        return quality_logits, component_logits, stance_logits


# =========================
# 2. Debate Analyzer
# =========================
class DebateAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../evaluation/debate_model.pt")

        # Initialize model with the exact same architecture
        self.model = MultiTaskModel().to(self.device)
        
        # Load weights safely
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device), 
            strict=False
        )
        self.model.eval()

        # Fixed Tokenizer: DeBERTa instead of RoBERTa
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        # 🔥 Fixed: Mapped exactly to how we encoded the labels in preprocessing.ipynb
        self.component_map = {0: "MajorClaim", 1: "Claim", 2: "Premise"}
        self.stance_map = {0: "CON", 1: "PRO"}

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.inference_mode():
            quality_logits, component_logits, stance_logits = self.model(
                input_ids,
                attention_mask
            )

            # Ensure quality score stays cleanly within bounds (since it was trained 0.0 to 1.0)
            quality = torch.clamp(quality_logits, min=0.0, max=1.0).item()
            component = torch.argmax(component_logits, dim=1).item()
            stance = torch.argmax(stance_logits, dim=1).item()

            fallacy = detect_fallacy(text)

            # Reduce score if fallacy detected
            if fallacy != "None":
                quality = quality * 0.6

        return {
            "argument_quality_score": round(quality, 3), # Output is 0.0 to 1.0
            "component": self.component_map[component],
            "stance": self.stance_map[stance],
            "fallacy": fallacy
        }


# =========================
# 3. Main Execution
# =========================
if __name__ == "__main__":
    analyzer = DebateAnalyzer()

    text = input("\nEnter argument:\n")

    result = analyzer.predict(text)

    print("\nPrediction")
    print(result)