import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModel, AutoTokenizer, get_scheduler
from tqdm import tqdm
import os

# PEFT
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 6
batch_size = 16

# =========================
# 1. Tokenizer (NEW)
# =========================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def add_task_prefix(example):
    text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

    if example["task_id"] == 0:
        text = "[QUALITY] " + text
    elif example["task_id"] == 1:
        text = "[COMPONENT] " + text
    else:
        text = "[STANCE] " + text

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    example["input_ids"] = enc["input_ids"]
    example["attention_mask"] = enc["attention_mask"]
    return example


# # =========================
# # 2. Load Dataset
# # =========================
# train_dataset = load_from_disk("./notebooks/data/train").map(add_task_prefix)
# val_dataset = load_from_disk("./notebooks/data/val").map(add_task_prefix)

# columns = ["input_ids", "attention_mask", "label", "task_id"]

# train_dataset.set_format(type="torch", columns=columns)
# val_dataset.set_format(type="torch", columns=columns)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# =========================
# 2. Load Dataset
# =========================
print("📥 Loading base training data...")
train_dataset = load_from_disk("./notebooks/data/train").map(add_task_prefix)
val_dataset = load_from_disk("./notebooks/data/val").map(add_task_prefix)

# 🔥 NEW: The RLAIF Data Flywheel Merge
rlaif_path = "./notebooks/data/rlaif_finetune"
if os.path.exists(rlaif_path):
    print("🚀 RLAIF Dataset found! Injecting Gemini corrections into training pipeline...")
    # Load and format the new RLAIF data just like the base data
    rlaif_dataset = load_from_disk(rlaif_path).map(add_task_prefix)
    
    # Merge them together into one massive dataset
    train_dataset = concatenate_datasets([train_dataset, rlaif_dataset])
    print(f"📈 Total training examples after merge: {len(train_dataset)}")
else:
    print("🌱 No RLAIF data found yet. Training on base dataset only.")

columns = ["input_ids", "attention_mask", "label", "task_id"]

train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# =========================
# 3. Model
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

        self.dropout = nn.Dropout(0.2)

        self.quality_head = nn.Linear(hidden, 1)
        self.component_head = nn.Linear(hidden, 3)
        self.stance_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1)

        pooled = self.dropout(pooled).to(torch.float32)

        quality = self.quality_head(pooled).squeeze(-1)
        component = self.component_head(pooled)
        stance = self.stance_head(pooled)

        return quality, component, stance


model = MultiTaskModel().to(device)


# =========================
# 4. Loss Functions
# =========================

# 🔥 Label smoothing (better than focal here)
component_loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor([10.1, 2.33, 1.0]).to(device),
    label_smoothing=0.1
)

stance_loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor([1.57, 1.0]).to(device)
)

def weighted_mse_loss(pred, labels):
    weights = torch.ones_like(labels)
    weights[labels < 0.4] = 2.0
    weights[labels < 0.2] = 3.0
    return (weights * (pred - labels) ** 2).mean()


# =========================
# 5. Optimizer
# =========================
optimizer = AdamW([
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": model.quality_head.parameters(), "lr": 1e-4},
    {"params": model.component_head.parameters(), "lr": 1e-4},
    {"params": model.stance_head.parameters(), "lr": 1e-4},
])

num_training_steps = num_epochs * len(train_loader)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

scaler = torch.amp.GradScaler()


# =========================
# 6. Training
# =========================
def train_epoch():
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        task_ids = batch["task_id"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):

            q_logits, c_logits, s_logits = model(input_ids, attention_mask)

            q_mask = task_ids == 0
            c_mask = task_ids == 1
            s_mask = task_ids == 2

            q_loss = c_loss = s_loss = torch.tensor(0.0, device=device)

            if q_mask.sum() > 0:
                q_loss = weighted_mse_loss(q_logits[q_mask], labels[q_mask].float())

            if c_mask.sum() > 0:
                c_loss = component_loss_fn(c_logits[c_mask], labels[c_mask].long())

            if s_mask.sum() > 0:
                s_loss = stance_loss_fn(s_logits[s_mask], labels[s_mask].long())

            # 🔥 Better balance
            loss = (1.0 * q_loss) + (1.2 * c_loss) + (1.3 * s_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# 7. Evaluation
# =========================
def evaluate(loader):
    model.eval()

    q_preds, q_labels = [], []
    c_preds, c_labels = [], []
    s_preds, s_labels = [], []

    with torch.no_grad():
        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            task_ids = batch["task_id"].to(device)

            q_logits, c_logits, s_logits = model(input_ids, attention_mask)

            q_mask = task_ids == 0
            c_mask = task_ids == 1
            s_mask = task_ids == 2

            if q_mask.sum() > 0:
                q_preds.extend(q_logits[q_mask].cpu().numpy())
                q_labels.extend(labels[q_mask].cpu().numpy())

            if c_mask.sum() > 0:
                preds = torch.argmax(c_logits[c_mask], dim=1)
                c_preds.extend(preds.cpu().numpy())
                c_labels.extend(labels[c_mask].cpu().numpy())

            if s_mask.sum() > 0:
                preds = torch.argmax(s_logits[s_mask], dim=1)
                s_preds.extend(preds.cpu().numpy())
                s_labels.extend(labels[s_mask].cpu().numpy())

    pearson = pearsonr(q_preds, q_labels)[0]
    comp_f1 = f1_score(c_labels, c_preds, average="macro")
    stance_f1 = f1_score(s_labels, s_preds, average="macro")

    return pearson, comp_f1, stance_f1


# =========================
# 8. Training Loop
# =========================
best_score = 0

for epoch in range(num_epochs):

    print(f"\n🚀 Epoch {epoch+1}")

    train_loss = train_epoch()
    pearson, comp_f1, stance_f1 = evaluate(val_loader)

    score = (pearson + comp_f1 + stance_f1) / 3

    print(f"Loss: {train_loss:.4f}")
    print(f"Pearson: {pearson:.4f}")
    print(f"Component F1: {comp_f1:.4f}")
    print(f"Stance F1: {stance_f1:.4f}")

    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), "debate_model.pt")
        print("🌟 Model improved and saved!")


print("✅ Training Complete")