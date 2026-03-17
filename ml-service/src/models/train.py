import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from utils import TaskBalancedBatchSampler
from datasets import load_from_disk
from transformers import AutoModel, get_scheduler
from tqdm import tqdm
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import os
import wandb

# PEFT
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 6

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

# Weights & Biases
wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    print("Warning: WANDB_API_KEY not found in .env. Run might fail to log.")

wandb.init(
    project="AI-Debate-Analyzer", 
    name="deberta-multitask-balanced",
    config={
        "epochs": num_epochs,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "architecture": "DeBERTa-v3-base + LoRA"
    }
)

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

train_loader = DataLoader(
    train_dataset, 
    batch_sampler=TaskBalancedBatchSampler(train_dataset, batch_size=16)
)
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
        self.stance_head = nn.Linear(hidden, 2)

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

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        # 'weight' is our [10.1, 2.33, 1.0] penalty tensor
        self.weight = weight 
        # 'gamma' controls how hard we ignore easy examples. 2.0 is the industry standard.
        self.gamma = gamma 

    def forward(self, inputs, targets):
        # 1. Calculate standard Cross Entropy (but don't average it yet)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # 2. Extract the probability the model assigned to the CORRECT class (pt)
        # Fun PyTorch trick: ce_loss is -log(pt), so exp(-ce_loss) gives us pt back!
        pt = torch.exp(-ce_loss)
        
        # 3. Apply the focal scaling factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Return the mean loss for the batch
        return focal_loss.mean()

# =========================
# 3. Loss Functions
# =========================

# 🧩 Component Class Weights + Focal Loss (The Ultimate Fix)
component_class_weights = torch.tensor([10.1, 2.33, 1.0]).to(device)

# 🔥 Replaced nn.CrossEntropyLoss with our new FocalLoss!
ce_loss = FocalLoss(weight=component_class_weights, gamma=2.0) 

# 🎯 Stance Class Weights
stance_class_weights = torch.tensor([1.57, 1.0]).to(device)
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

scaler = torch.cuda.amp.GradScaler()

def train_epoch():
    model.train()
    total_loss = 0

    progress = tqdm(train_loader, desc="Training", leave=False, position=0, dynamic_ncols=True)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        task_ids = batch["task_id"].to(device)

        optimizer.zero_grad()

        # 🔥 1. Wrap the forward pass and loss computation in autocast
        with torch.cuda.amp.autocast():
            quality_logits, component_logits, stance_logits = model(
                input_ids, attention_mask
            )

            quality_mask = task_ids == 0
            component_mask = task_ids == 1
            stance_mask = task_ids == 2

            # Losses
            q_loss = c_loss = s_loss = torch.tensor(0.0, device=device)

            if quality_mask.sum() > 0:
                q_loss = weighted_mse_loss(
                    quality_logits[quality_mask],
                    labels[quality_mask].float()
                )

            if component_mask.sum() > 0:
                c_loss = ce_loss(
                    component_logits[component_mask],
                    labels[component_mask].long()
                )

            if stance_mask.sum() > 0:
                s_loss = stance_loss_fn(
                    stance_logits[stance_mask],
                    labels[stance_mask].long()
                )

            loss = (1.0 * q_loss) + (0.7 * c_loss) + (1.5 * s_loss)

        # 🔥 2. Scale the loss and call backward
        scaler.scale(loss).backward()
        
        # 🔥 3. Unscale the gradients BEFORE clipping them
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 🔥 4. Step the optimizer using the scaler, then update the scaler
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()

        total_loss += loss.item()

        # ✅ Clean postfix instead of new lines
        progress.set_postfix(loss=f"{loss.item():.4f}")

        wandb.log({"batch_train_loss": loss.item()})

    return total_loss / len(train_loader)

# =========================
# 6. Evaluation
# =========================

def evaluate_with_metrics(loader):
    model.eval()
    total_loss = 0

    all_q_preds, all_q_labels = [], []
    all_c_preds, all_c_labels = [], [] # 🔥 Added Component Tracking
    all_s_preds, all_s_labels = [], []

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
            component_mask = task_ids == 1 # 🔥 Added Component Mask
            stance_mask = task_ids == 2

            # 🔥 CRITICAL FIX: Initialize as tensor so .item() never crashes
            loss = torch.tensor(0.0, device=device)

            # 1. Quality
            if quality_mask.sum() > 0:
                loss += weighted_mse_loss(
                    quality_logits[quality_mask], labels[quality_mask].float()
                )
                all_q_preds.extend(quality_logits[quality_mask].cpu().numpy())
                all_q_labels.extend(labels[quality_mask].cpu().numpy())

            # 2. Components (Added)
            if component_mask.sum() > 0:
                loss += ce_loss(
                    component_logits[component_mask], labels[component_mask].long()
                )
                preds = torch.argmax(component_logits[component_mask], dim=1)
                all_c_preds.extend(preds.cpu().numpy())
                all_c_labels.extend(labels[component_mask].cpu().numpy())

            # 3. Stance
            if stance_mask.sum() > 0:
                loss += stance_loss_fn(
                    stance_logits[stance_mask], labels[stance_mask].long()
                )
                preds = torch.argmax(stance_logits[stance_mask], dim=1)
                all_s_preds.extend(preds.cpu().numpy())
                all_s_labels.extend(labels[stance_mask].cpu().numpy())

            total_loss += loss.item()

    # --- Compute Metrics ---
    pearson = pearsonr(all_q_preds, all_q_labels)[0] if len(all_q_preds) > 1 else 0
    
    # 🔥 Added zero_division=0 to prevent console spam
    comp_f1 = f1_score(all_c_labels, all_c_preds, average="macro", zero_division=0) if len(all_c_preds) > 0 else 0
    stance_f1 = f1_score(all_s_labels, all_s_preds, average="macro", zero_division=0) if len(all_s_preds) > 0 else 0

    return total_loss / len(loader), pearson, comp_f1, stance_f1

# 🔥 PASTE THE TEST CODE RIGHT HERE 🔥
# ==========================================
print("\n--- Testing Batch Sampler ---")
first_batch = next(iter(train_loader))
task_ids = first_batch["task_id"].numpy()

print(f"Total samples in batch: {len(task_ids)}")
print(f"Task 0 (Quality) count: {(task_ids == 0).sum()}")
print(f"Task 1 (Component) count: {(task_ids == 1).sum()}")
print(f"Task 2 (Stance) count: {(task_ids == 2).sum()}")
print("-----------------------------\n")
# ==========================================

# =========================
# 7. Training Loop
# =========================
# best_val_loss = float('inf')
best_composite_score = 0.0

for epoch in range(num_epochs):

    print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")

    train_loss = train_epoch()

    val_loss, pearson, comp_f1, stance_f1 = evaluate_with_metrics(val_loader) # Update unpacking

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"📊 Pearson (Quality): {pearson:.4f}")
    print(f"🧩 Component F1: {comp_f1:.4f}") # Print the new metric
    print(f"🎯 Stance F1: {stance_f1:.4f}")

    # if val_loss < best_val_loss:
    #     print(f"🌟 Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving weights!")
    #     best_val_loss = val_loss
    #     # Save the model IMMEDIATELY when it hits a new peak
    #     torch.save(model.state_dict(), "debate_model.pt")
    # else:
    #     print(f"⚠️ Validation loss did not improve.")

    # 🔥 The New Checkpointing Logic: Average the three metrics
    composite_score = (pearson + comp_f1 + stance_f1) / 3.0

    if composite_score > best_composite_score:
        print(f"🌟 Composite score improved to {composite_score:.4f}. Saving weights!")
        best_composite_score = composite_score
        torch.save(model.state_dict(), "debate_model.pt")
    else:
        print(f"⚠️ Score did not improve.")

    wandb.log({
        "epoch": epoch + 1,
        "epoch_train_loss": train_loss,
        "val_loss": val_loss,
        "pearson_quality": pearson,
        "component_f1": comp_f1, # Log it to W&B
        "stance_f1": stance_f1
    })

    print("Training complete. The best weights are saved in 'debate_model.pt'.")

# =========================
# 8. Save Model
# =========================
wandb.finish()
torch.save(model.state_dict(), "debate_model.pt")
print("Model saved.")