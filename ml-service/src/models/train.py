import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_from_disk
from transformers import AutoModel, get_scheduler
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import os

# PEFT
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 0. Hugging Face Login
# =========================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("Logging into Hugging Face Hub...")
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found.")

# =========================
# 1. Load Dataset
# =========================
train_dataset = load_from_disk("./notebooks/data/train")
val_dataset = load_from_disk("./notebooks/data/val")
test_dataset = load_from_disk("./notebooks/data/test")

columns = ["input_ids", "attention_mask", "label", "task_id"]

train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# =========================
# 2. Model
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
        self.stance_head = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling (better than CLS for DeBERTa)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1)

        pooled = pooled.to(torch.float32)

        quality = self.quality_head(pooled).squeeze(-1)
        component = self.component_head(pooled)
        stance = self.stance_head(pooled)

        return quality, component, stance


model = MultiTaskModel().to(device)
model.encoder.print_trainable_parameters()

# =========================
# 3. Loss Functions
# =========================
ce_loss = nn.CrossEntropyLoss()

# 🔥 Adjust these weights after checking distribution
stance_class_weights = torch.tensor([1.2, 1.0, 1.3]).to(device)
stance_loss_fn = nn.CrossEntropyLoss(weight=stance_class_weights)

def weighted_mse_loss(pred, labels):
    weights = torch.ones_like(labels)
    weights[labels < 0.4] = 2.0
    weights[labels < 0.2] = 3.0
    return (weights * (pred - labels) ** 2).mean()

# =========================
# 4. Optimizer
# =========================
optimizer = AdamW([
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": model.quality_head.parameters(), "lr": 1e-4},
    {"params": model.component_head.parameters(), "lr": 1e-4},
    {"params": model.stance_head.parameters(), "lr": 1e-4},
])

num_epochs = 6

num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

# =========================
# 5. Training
# =========================
def train_epoch():
    model.train()
    total_loss = 0

    progress = tqdm(train_loader)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        task_ids = batch["task_id"].to(device)

        optimizer.zero_grad()

        quality_logits, component_logits, stance_logits = model(
            input_ids, attention_mask
        )

        # Masks
        quality_mask = task_ids == 0
        component_mask = task_ids == 1
        stance_mask = task_ids == 2

        loss = 0

        # Quality
        if quality_mask.sum() > 0:
            q_pred = quality_logits[quality_mask]
            q_label = labels[quality_mask].float()
            q_loss = weighted_mse_loss(q_pred, q_label)
        else:
            q_loss = 0

        # Component
        if component_mask.sum() > 0:
            c_pred = component_logits[component_mask]
            c_label = labels[component_mask].long()
            c_loss = ce_loss(c_pred, c_label)
        else:
            c_loss = 0

        # Stance (BOOSTED)
        if stance_mask.sum() > 0:
            s_pred = stance_logits[stance_mask]
            s_label = labels[stance_mask].long()
            s_loss = stance_loss_fn(s_pred, s_label)
        else:
            s_loss = 0

        # 🔥 Task weighting
        loss = (1.0 * q_loss) + (0.7 * c_loss) + (1.5 * s_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress.set_description(f"loss {loss.item():.4f}")

    return total_loss / len(train_loader)

# =========================
# 6. Evaluation
# =========================
def evaluate(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            task_ids = batch["task_id"].to(device)

            quality_logits, component_logits, stance_logits = model(
                input_ids, attention_mask
            )

            quality_mask = task_ids == 0
            component_mask = task_ids == 1
            stance_mask = task_ids == 2

            loss = 0

            if quality_mask.sum() > 0:
                loss += weighted_mse_loss(
                    quality_logits[quality_mask],
                    labels[quality_mask].float()
                )

            if component_mask.sum() > 0:
                loss += ce_loss(
                    component_logits[component_mask],
                    labels[component_mask].long()
                )

            if stance_mask.sum() > 0:
                loss += stance_loss_fn(
                    stance_logits[stance_mask],
                    labels[stance_mask].long()
                )

            total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# 7. Training Loop
# =========================
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = evaluate(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

# =========================
# 8. Save Model
# =========================
torch.save(model.state_dict(), "debate_model.pt")
print("Model saved.")