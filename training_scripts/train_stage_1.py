import yaml
import torch
import os
import json
import gc
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MistralForCausalLM,
    CLIPVisionModel,
)
from models import VisionLanguageConnector
from data import COCO_Loader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- IMPROVEMENT: Seed setting for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# --- 1. Load Configuration ---
print("--- Initializing Stage 1: Vision Connector Training ---")
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
train_params = config["training_stage1"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW: Add Gradient Accumulation Steps from config ---
accumulation_steps = train_params.get("gradient_accumulation_steps", 1)


print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: CUDA not available, using CPU!")
    exit(1)

# --- 2. Load Foundational Models ---
print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
llm = MistralForCausalLM.from_pretrained(
    paths["mistral_local_path"],
    load_in_8bit=True,
    torch_dtype=torch.bfloat16 # Use bfloat16 for consistency
)
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

for param in vision_encoder.parameters():
    param.requires_grad = False
for param in llm.parameters():
    param.requires_grad = False
print("✅ Models loaded and frozen.")

# --- 3. Create Datasets and DataLoaders ---
print("Creating datasets and dataloaders...")
train_dataset = COCO_Loader(
    image_dir=paths["image_dir"],
    annotations_file=paths["annotations_file"],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"],
    split="train",
)
val_dataset = COCO_Loader(
    image_dir=paths["image_dir"],
    annotations_file=paths["annotations_file"],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"],
    split="val",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_params["batch_size"],
    shuffle=True,
    num_workers=loader_params["num_workers_s1"],
    pin_memory=True,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=train_params["batch_size"],
    shuffle=False,
    num_workers=loader_params["num_workers_s1"],
    pin_memory=True, # IMPROVEMENT: Faster data transfer to GPU
    persistent_workers=True
)

# --- 4. Setup Model, Optimizer, and Checkpointing ---
vision_connector = VisionLanguageConnector().to(DEVICE)

optimizer = optim.AdamW(
    vision_connector.parameters(),
    lr=train_params["learning_rate"],
    weight_decay=train_params.get("weight_decay", 0.01)
)

scheduler = CosineAnnealingLR(optimizer, T_max=(len(train_loader) // accumulation_steps) * NUM_EPOCHS)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scaler = GradScaler()
metrics_history = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate": []}
best_val_loss = float('inf')

os.makedirs(OUTPUT_DIR, exist_ok=True)
best_model_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth")
latest_model_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1_latest.pth")


if os.path.exists(latest_model_path):
    print(f"💾 Loading saved weights from {latest_model_path}")
    vision_connector.load_state_dict(torch.load(latest_model_path, map_location=DEVICE))

vision_connector = torch.compile(vision_connector)

# --- 5. The Training and Validation Loop ---
print("🚀 Starting training...")
for epoch in range(NUM_EPOCHS):
    vision_connector.train()
    total_train_loss = 0
    
    # Reset gradients at the start of the epoch
    optimizer.zero_grad()
    
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
        )
        
        with torch.no_grad():
            patch_embeddings = vision_encoder(images).last_hidden_state
            text_embeddings = llm.model.embed_tokens(input_ids)
        
        with autocast(dtype=torch.bfloat16):
            visual_soft_tokens = vision_connector(patch_embeddings)
            num_visual_tokens = visual_soft_tokens.shape[1]

            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            combined_attention_mask = torch.cat([
                torch.ones(visual_soft_tokens.shape[:2], device=DEVICE),
                attention_mask
            ], dim=1)

            outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
            logits = outputs.logits

            text_logits = logits[:, num_visual_tokens - 1: -1, :].contiguous()
            text_labels = input_ids.contiguous()
            loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))
            
            # --- CHANGE: Scale loss for gradient accumulation ---
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        
        # --- CHANGE: Optimizer step only after accumulating gradients ---
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vision_connector.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad() # Zero gradients after optimizer step

        total_train_loss += loss.item() * accumulation_steps # Un-scale for logging

        if (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}]")

        del images, input_ids, attention_mask, patch_embeddings, text_embeddings
        del visual_soft_tokens, combined_embeddings, outputs, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

    # -- Validation Phase --
    vision_connector.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
            )
            with autocast(dtype=torch.bfloat16):
                patch_embeddings = vision_encoder(images).last_hidden_state
                text_embeddings = llm.model.embed_tokens(input_ids)
                visual_soft_tokens = vision_connector(patch_embeddings)
                num_visual_tokens = visual_soft_tokens.shape[1]

                combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
                combined_attention_mask = torch.cat([
                    torch.ones(visual_soft_tokens.shape[:2], device=DEVICE),
                    attention_mask
                ], dim=1)

                outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
                logits = outputs.logits
                text_logits = logits[:, num_visual_tokens - 1: -1, :].contiguous()
                text_labels = input_ids.contiguous()
                loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # --- Early Stopping and Checkpointing Logic ---
    # Access the original model with ._orig_mod to get a compatible state dict
    state_to_save = vision_connector._orig_mod.state_dict()

    # Save the clean state for the latest checkpoint
    torch.save(state_to_save, latest_model_path)
    print(f"💾 Latest model checkpoint saved to {latest_model_path}")
        
    # If it's the best model, save the clean state for the best checkpoint too
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(state_to_save, best_model_path)
        print(f"🏆 New best validation loss! Model saved to {best_model_path}")

    # --- Metrics Logging ---
    metrics_path = os.path.join(OUTPUT_DIR, "loss_history_stage1.json")
    
    metrics_history["epoch"].append(epoch + 1)
    metrics_history["train_loss"].append(avg_train_loss)
    metrics_history["val_loss"].append(avg_val_loss)
    metrics_history["learning_rate"].append(optimizer.param_groups[0]['lr'])
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=4)
    print(f"✅ Metrics saved to {metrics_path}")

print("✅ Training complete.")

