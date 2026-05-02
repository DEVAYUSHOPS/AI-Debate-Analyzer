import os
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

from peft import LoraConfig, get_peft_model, TaskType
from .fallacy_detector import detect_fallacy


NEGATION_CUES = {
    "not", "no", "never", "without", "against", "oppose", "opposes",
    "opposed", "opposing", "reject", "rejects", "rejected", "unwarranted",
    "unnecessary", "harmful"
}

STANCE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can",
    "could", "do", "does", "for", "from", "have", "in", "is", "it", "its",
    "may", "might", "more", "must", "of", "on", "or", "should", "than",
    "that", "the", "their", "them", "there", "this", "to", "too", "very",
    "was", "we", "were", "which", "will", "with", "would"
}


def tokenize_words(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())


def content_terms(tokens):
    return {
        token.strip("'")
        for token in tokens
        if len(token.strip("'")) > 2 and token.strip("'") not in STANCE_STOPWORDS
    }


def negated_terms(tokens):
    negated = set()

    for index, token in enumerate(tokens):
        token = token.strip("'")
        if len(token) <= 2:
            continue

        before = tokens[max(0, index - 4):index]
        after = tokens[index + 1:index + 4]

        if any(cue in NEGATION_CUES for cue in before + after):
            negated.add(token)

    return negated


# =========================
# HF Login
# =========================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)


# =========================
# Model
# =========================
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

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
        self.stance_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1)

        pooled = pooled.to(torch.float32)

        quality = self.quality_head(pooled).squeeze(-1)
        component = self.component_head(pooled)
        stance = self.stance_head(pooled)

        return quality, component, stance


# =========================
# Analyzer
# =========================
class DebateAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../evaluation/debate_model.pt")

        self.model = MultiTaskModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        self.component_map = {0: "MajorClaim", 1: "Claim", 2: "Premise"}
        self.stance_map = {0: "CON", 1: "PRO"}

    # 🔥 ADD THIS
    def add_prefix(self, text, task, topic=None):
        if task == "quality":
            return "[QUALITY] " + text
        elif task == "component":
            return "[COMPONENT] " + text
        elif task == "stance":
            if topic:
                return f"[STANCE] topic: {topic} argument: {text}"
            return "[STANCE] " + text

    def predict(self, text, topic=None):

        def run_task(task_name):
            stance_topic = topic if task_name == "stance" else None
            prefixed_text = self.add_prefix(text, task_name, stance_topic)

            inputs = self.tokenizer(
                prefixed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.inference_mode():
                return self.model(input_ids, attention_mask)

        # 🔥 Run tasks separately
        quality_logits, _, _ = run_task("quality")
        _, component_logits, _ = run_task("component")
        _, _, stance_logits = run_task("stance")

        # 🔥 FIXED: sigmoid instead of clamp
        quality = torch.sigmoid(quality_logits).item()

        component = torch.argmax(component_logits, dim=1).item()
        stance = torch.argmax(stance_logits, dim=1).item()
        raw_stance = self.stance_map[stance]
        corrected_stance, stance_reason = self.correct_stance_with_topic(
            raw_stance,
            text,
            topic
        )

        fallacy = detect_fallacy(text)

        if fallacy != "None":
            quality *= 0.6

        result = {
            "argument_quality": round(quality, 3),
            "component": self.component_map[component],
            "stance": corrected_stance,
            "fallacy": fallacy
        }

        if topic:
            result["raw_stance"] = raw_stance
            result["stance_reason"] = stance_reason

        return result

    def correct_stance_with_topic(self, raw_stance, text, topic=None):
        if not topic:
            return raw_stance, "model_only"

        topic_tokens = tokenize_words(topic)
        text_tokens = tokenize_words(text)

        shared_terms = content_terms(topic_tokens) & content_terms(text_tokens)
        if not shared_terms:
            return raw_stance, "model_only_no_topic_overlap"

        topic_negated = negated_terms(topic_tokens)
        text_negated = negated_terms(text_tokens)

        negation_conflicts = [
            term
            for term in shared_terms
            if (term in topic_negated) != (term in text_negated)
        ]

        if negation_conflicts:
            return "CON", f"topic_negation_conflict:{','.join(sorted(negation_conflicts))}"

        explicit_opposition = any(cue in text_tokens for cue in NEGATION_CUES)
        if explicit_opposition and len(shared_terms) >= 2 and raw_stance == "PRO":
            return "CON", "explicit_opposition_to_topic"

        return raw_stance, "model_only"


# =========================
# Run
# =========================
if __name__ == "__main__":
    analyzer = DebateAnalyzer()

    topic = input("\nEnter debate topic/motion (optional):\n").strip()
    text = input("\nEnter argument:\n")
    result = analyzer.predict(text, topic=topic or None)

    print("\nPrediction")
    print(result)
