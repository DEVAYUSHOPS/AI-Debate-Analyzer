import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModel
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load Dataset
# =========================

test_dataset = load_from_disk("./notebooks/data/test")

test_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","label","task_id"]
)

test_loader = DataLoader(test_dataset, batch_size=16)


# =========================
# Model Definition
# =========================

class MultiTaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained("roberta-base")

        hidden = self.encoder.config.hidden_size

        self.quality_head = nn.Linear(hidden, 1)
        self.component_head = nn.Linear(hidden, 3)
        self.stance_head = nn.Linear(hidden, 3)


    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0]

        quality_logits = self.quality_head(cls)
        component_logits = self.component_head(cls)
        stance_logits = self.stance_head(cls)

        return quality_logits, component_logits, stance_logits


model = MultiTaskModel().to(device)

model_path = os.path.join(current_dir, "debate_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()


# =========================
# Evaluation Containers
# =========================

quality_preds = []
quality_labels = []

component_preds = []
component_labels = []

stance_preds = []
stance_labels = []


# =========================
# Evaluation Loop
# =========================

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        task_ids = batch["task_id"].to(device)

        quality_logits, component_logits, stance_logits = model(
            input_ids,
            attention_mask
        )

        for i in range(len(task_ids)):

            task = task_ids[i]

            if task == 0:

                pred = quality_logits[i].item()

                quality_preds.append(pred)
                quality_labels.append(labels[i].item())


            elif task == 1:

                pred = torch.argmax(component_logits[i]).item()

                component_preds.append(pred)
                component_labels.append(int(labels[i].item()))


            elif task == 2:

                pred = torch.argmax(stance_logits[i]).item()

                stance_preds.append(pred)
                stance_labels.append(int(labels[i].item()))


# =========================
# Compute Metrics
# =========================

print("\nEvaluation Results\n")

if len(quality_preds) > 0:

    corr, _ = pearsonr(quality_labels, quality_preds)

    print(f"Argument Quality Pearson Correlation: {corr:.4f}")


if len(component_preds) > 0:

    f1 = f1_score(component_labels, component_preds, average="macro")

    print(f"Argument Component F1: {f1:.4f}")


if len(stance_preds) > 0:

    f1 = f1_score(stance_labels, stance_preds, average="macro")

    print(f"Stance Detection F1: {f1:.4f}")