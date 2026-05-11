"""
NPPE-2 multilingual speech recognition:
Kaggle-ready Whisper baseline upgrade.

Recommended Kaggle packages:
!pip install -q transformers datasets evaluate jiwer peft accelerate librosa soundfile num2words

This script is designed to be pasted into a Kaggle notebook cell or uploaded as a script.
It gives you two practical paths:

1. QUICK_INFERENCE_ONLY = True
   Use a much stronger zero-shot model than your current notebook.
   Best when you are short on time.

2. QUICK_INFERENCE_ONLY = False
   Fine-tune Whisper with LoRA on the 2,000 training examples, validate with WER,
   then generate a submission.
"""

import gc
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from num2words import num2words
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# =========================
# Config
# =========================

COMP_DIR = "/kaggle/input/competitions/multilingual-speech-recognition"
TRAIN_CSV = f"{COMP_DIR}/train.csv"
TEST_CSV = f"{COMP_DIR}/test.csv"
SAMPLE_SUB = f"{COMP_DIR}/sample_submission.csv"
TRAIN_AUDIO_DIR = f"{COMP_DIR}/competition_data/train"
TEST_AUDIO_DIR = f"{COMP_DIR}/competition_data/test"

SEED = 42
TARGET_SR = 16000
MAX_AUDIO_SECONDS = 30

# Fastest strong option:
# MODEL_NAME = "openai/whisper-large-v3-turbo"
#
# Better fine-tuning fit under Kaggle constraints:
MODEL_NAME = "openai/whisper-small"

# If you only have a little time left, set this to True and skip training.
QUICK_INFERENCE_ONLY = False

# Keep these capped if you want a faster dry run first.
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

OUTPUT_DIR = "/kaggle/working/whisper-nppe2"
SUBMISSION_PATH = "/kaggle/working/submission.csv"


# =========================
# Repro / device
# =========================

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = DEVICE == "cuda"

print("DEVICE:", DEVICE)
print("MODEL_NAME:", MODEL_NAME)
print("QUICK_INFERENCE_ONLY:", QUICK_INFERENCE_ONLY)


# =========================
# Text cleanup
# =========================

ABBREV_REPLACEMENTS = {
    "ph.d.": "phd",
    "ph. d.": "phd",
    "dr.": "dr",
    "mr.": "mr",
    "mrs.": "mrs",
    "ms.": "ms",
    "prof.": "prof",
}


def looks_mostly_latin(text: str) -> bool:
    if not text:
        return False
    latin_like = sum(ch.isascii() for ch in text)
    return latin_like / max(len(text), 1) > 0.85


def expand_english_numbers(text: str) -> str:
    def repl(match: re.Match) -> str:
        token = match.group(0)
        try:
            return num2words(int(token)).replace("-", " ")
        except Exception:
            return token

    return re.sub(r"\b\d+\b", repl, text)


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    for src, dst in ABBREV_REPLACEMENTS.items():
        text = text.replace(src, dst)

    if looks_mostly_latin(text):
        text = expand_english_numbers(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# Data loading
# =========================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["audio_path"] = train_df["audio"].apply(lambda x: os.path.join(TRAIN_AUDIO_DIR, x))
test_df["audio_path"] = test_df["audio"].apply(lambda x: os.path.join(TEST_AUDIO_DIR, x))
train_df["text_norm"] = train_df["text"].apply(normalize_text)

if MAX_TRAIN_SAMPLES:
    train_df = train_df.sample(MAX_TRAIN_SAMPLES, random_state=SEED).reset_index(drop=True)

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)


def script_bucket(text: str) -> str:
    text = str(text)
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "tamil"
    if re.search(r"[\u0900-\u097F]", text):
        return "devanagari"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "telugu"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "malayalam"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gujarati"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bengali"
    return "latin_or_other"


train_df["script_bucket"] = train_df["text"].apply(script_bucket)

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["script_bucket"],
)

train_split = train_split.reset_index(drop=True)
valid_split = valid_split.reset_index(drop=True)

if MAX_EVAL_SAMPLES:
    valid_split = valid_split.sample(min(MAX_EVAL_SAMPLES, len(valid_split)), random_state=SEED)

print(train_split["script_bucket"].value_counts(dropna=False))
print(valid_split["script_bucket"].value_counts(dropna=False))


# =========================
# Processor / model
# =========================

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


def load_audio_array(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=TARGET_SR)
    return audio.astype(np.float32)


def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
    audio = load_audio_array(batch["audio_path"])

    if len(audio) > TARGET_SR * MAX_AUDIO_SECONDS:
        audio = audio[: TARGET_SR * MAX_AUDIO_SECONDS]

    batch["input_features"] = processor.feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
    ).input_features[0]

    labels = processor.tokenizer(batch["text_norm"]).input_ids
    batch["labels"] = labels
    return batch


train_ds = Dataset.from_pandas(train_split[["audio_path", "text_norm"]], preserve_index=False)
valid_ds = Dataset.from_pandas(valid_split[["audio_path", "text_norm"]], preserve_index=False)

train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=1)
valid_ds = valid_ds.map(prepare_dataset, remove_columns=valid_ds.column_names, num_proc=1)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
wer_metric = load("wer")


def decode_predictions(pred_ids: np.ndarray) -> List[str]:
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    return [normalize_text(x) for x in texts]


def compute_metrics(pred) -> Dict[str, float]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    pred_str = decode_predictions(pred_ids)
    label_str = decode_predictions(label_ids)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def make_lora_model(base_model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(base_model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def transcribe_file(path: str, infer_model: WhisperForConditionalGeneration) -> str:
    audio = load_audio_array(path)
    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        generated = infer_model.generate(
            input_features,
            task="transcribe",
            max_new_tokens=225,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return normalize_text(text)


def evaluate_holdout(infer_model: WhisperForConditionalGeneration, df: pd.DataFrame, n_rows: int = None) -> float:
    subset = df.copy()
    if n_rows is not None:
        subset = subset.head(n_rows).copy()

    preds = [transcribe_file(path, infer_model) for path in subset["audio_path"]]
    refs = subset["text_norm"].tolist()
    wer = wer_metric.compute(predictions=preds, references=refs)
    return wer


if QUICK_INFERENCE_ONLY:
    model = model.to(DEVICE)
    quick_wer = evaluate_holdout(model, valid_split, n_rows=len(valid_split))
    print("quick validation wer:", quick_wer)
else:
    model = make_lora_model(model)
    model.config.use_cache = False
    model = model.to(DEVICE)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        num_train_epochs=6,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=20,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=FP16,
        gradient_checkpointing=True,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("validation metrics:", metrics)

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


model.eval()

# Optional: inspect a few holdout examples before test submission.
preview = valid_split.head(5).copy()
preview["prediction"] = preview["audio_path"].apply(lambda p: transcribe_file(p, model))
print(preview[["text", "prediction"]].to_string(index=False))


# =========================
# Submission
# =========================

submission = pd.read_csv(SAMPLE_SUB)
submission["text"] = test_df["audio_path"].apply(lambda p: transcribe_file(p, model))
submission.to_csv(SUBMISSION_PATH, index=False)

print("Saved:", SUBMISSION_PATH)
print(submission.head())
