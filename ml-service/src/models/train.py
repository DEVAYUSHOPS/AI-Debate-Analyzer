import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from transformers import get_scheduler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1. Load Dataset
# =========================

train_dataset = load_from_disk("./notebooks/data/train")
val_dataset = load_from_disk("./notebooks/data/val")
test_dataset = load_from_disk("./notebooks/data/test")


train_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label","task_id"])
val_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label","task_id"])
test_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label","task_id"])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


# =========================
# 2. Multi-Task Model
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


# =========================
# 3. Loss Functions
# =========================

mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()


# =========================
# 4. Optimizer
# =========================

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# =========================
# 5. Training Function
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
            input_ids,
            attention_mask
        )

        loss = 0

        for i in range(len(task_ids)):

            task = task_ids[i]

            if task == 0:
                pred = quality_logits[i]
                label = labels[i].float()
                loss += mse_loss(pred.squeeze(), label)

            elif task == 1:
                pred = component_logits[i].unsqueeze(0)
                label = labels[i].long().unsqueeze(0)
                loss += ce_loss(pred, label)

            elif task == 2:
                pred = stance_logits[i].unsqueeze(0)
                label = labels[i].long().unsqueeze(0)
                loss += ce_loss(pred, label)

        loss = loss / len(task_ids)

        loss.backward()

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
                input_ids,
                attention_mask
            )

            loss = 0

            for i in range(len(task_ids)):

                task = task_ids[i]

                if task == 0:
                    pred = quality_logits[i]
                    label = labels[i].float()
                    loss += mse_loss(pred.squeeze(), label)

                elif task == 1:
                    pred = component_logits[i].unsqueeze(0)
                    label = labels[i].long().unsqueeze(0)
                    loss += ce_loss(pred, label)

                elif task == 2:
                    pred = stance_logits[i].unsqueeze(0)
                    label = labels[i].long().unsqueeze(0)
                    loss += ce_loss(pred, label)

            loss = loss / len(task_ids)

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
