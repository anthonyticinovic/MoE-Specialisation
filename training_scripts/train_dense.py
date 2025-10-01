import time
import json
import yaml
import torch
import os
import gc
import sys
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    CLIPVisionModel,
    MistralForCausalLM,
    MistralConfig,
)
from models import VisionLanguageConnector
from data import COCO_Loader
from torch.amp import GradScaler, autocast
from torch.distributed.fsdp import CPUOffload
from torch.optim.lr_scheduler import CosineAnnealingLR

# Para GPU imports
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

from transformers.models.mistral.modeling_mistral import MistralMLP

from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer


# ====================================================================================
# 2. SETUP AND CONFIGURATION
# ====================================================================================
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
# CHANGED: Use dense_control parameters from config
train_params = config["dense_control"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
# CHANGED: New checkpoint directory for the dense model
DENSE_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "dense_checkpoints")

# --- Initialize the distributed environment ---
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = local_rank

if local_rank == 0:
    # CHANGED: Print statement for Dense Control model
    print("--- Initializing Dense Control Model Training ---")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
if local_rank == 0:
    print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

# CHANGED: Load the standard dense Mistral-7B model
dense_model_path = paths["mistral_local_path"] 
llm = AutoModelForCausalLM.from_pretrained(
    dense_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

# ====================================================================================
# 4. TRAINING SETUP (PART 1 - Parameter Freezing)
# ====================================================================================
if local_rank == 0:
    print("Preparing dense model: Freezing vision, VLC, and embeddings.")

# Freeze all parameters by default
for param in llm.parameters():
    param.requires_grad = False

# NEW: Unfreeze only the attention and MLP layers
for layer in llm.model.layers:
    layer.self_attn.requires_grad_(True)
    layer.mlp.requires_grad_(True)

vision_connector = VisionLanguageConnector().to(DEVICE)
# NEW: Freeze the vision connector
for param in vision_connector.parameters():
    param.requires_grad = False

# Ensure the vision encoder remains frozen
for param in vision_encoder.parameters():
    param.requires_grad = False

# ====================================================================================
# 5. FSDP WRAPPING & CHECKPOINTING
# ====================================================================================
my_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        MistralMLP,
    },
)

llm = FSDP(
    llm,
    device_id=torch.cuda.current_device(),
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    use_orig_params=True,
)

# --- Load Stage 1 Vision Connector weights ---
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth")
if os.path.exists(stage1_weights_path):
    if local_rank == 0:
        print(f"💾 Loading Stage 1 Vision Connector weights from {stage1_weights_path}")
    vision_connector.load_state_dict(torch.load(stage1_weights_path, map_location=DEVICE))

dist.barrier()

# ====================================================================================
# 6. DATA & OPTIMIZER
# ====================================================================================
if local_rank == 0:
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

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(
    train_dataset, batch_size=train_params["batch_size"], sampler=train_sampler,
    num_workers=loader_params["num_workers"], pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=train_params["batch_size"], sampler=val_sampler,
    num_workers=loader_params["num_workers"], pin_memory=True,
)

accumulation_steps = train_params.get("gradient_accumulation_steps", 1)

# Get only the parameters that are unfrozen
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=True)
scaler = GradScaler()
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# --- Resumption logic for Dense Model ---
start_epoch = 0
best_val_loss = float('inf')
latest_checkpoint_path = os.path.join(DENSE_CHECKPOINT_DIR, "dense_latest.pth")

should_resume = 1.0 if local_rank == 0 and os.path.exists(latest_checkpoint_path) else 0.0
resume_tensor = torch.tensor([should_resume], dtype=torch.float32).to(DEVICE)
dist.broadcast(resume_tensor, src=0)

if resume_tensor.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Resuming dense training from latest checkpoint: {latest_checkpoint_path}")
    
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
    
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    del checkpoint
    gc.collect()
    
    dist.barrier()
    if local_rank == 0:
        print(f"✅ Resumed successfully. Starting from epoch {start_epoch}.")
else:
    if local_rank == 0:
        print("🏁 No 'latest' checkpoint found. Starting training from scratch.")

if local_rank == 0:
    print(f"Optimizing {sum(p.numel() for p in trainable_params)} trainable parameters.")

metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_dense.json")
metrics_history = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate":[]}
if local_rank == 0 and start_epoch > 0 and os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)

# ====================================================================================
# 8. TRAINING LOOP
# ====================================================================================
if local_rank == 0:
    print(f"🚀 Starting dense model training from epoch {start_epoch}...")
start_time = time.time()
for epoch in range(start_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    llm.train()
    total_train_loss = 0
    optimizer.zero_grad()
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
        )

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            # The VLC and embedding layers are frozen, so run them in no_grad
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)
            
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            combined_attention_mask = torch.cat(
                [torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask],
                dim=1,
            )
            
            outputs = llm(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
            )
            logits = outputs.logits
            
            num_visual_tokens = visual_soft_tokens.shape[1]
            text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
            text_labels = input_ids[..., 1:].contiguous()
            
            loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        
        if loss.item() > 0:
            total_train_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    llm.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
            )
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)
                combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
                combined_attention_mask = torch.cat(
                    [torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask],
                    dim=1,
                )
                outputs = llm(
                    inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask,
                )
                logits = outputs.logits
                num_visual_tokens = visual_soft_tokens.shape[1]
                text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
                text_labels = input_ids[..., 1:].contiguous()
                
                loss = loss_fn(
                    text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1)
                )
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    val_loss_tensor = torch.tensor(avg_val_loss).to(DEVICE)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    avg_val_loss = val_loss_tensor.item()
    
    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # --- Metrics and Checkpoint Saving ---
    if local_rank == 0:
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"✅ Metrics saved to {metrics_path}")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, save_policy):
        llm_state_dict = llm.state_dict()
    
    if local_rank == 0:
        consolidated_checkpoint = {
            'model_state_dict': llm_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_loss,
        }

        os.makedirs(DENSE_CHECKPOINT_DIR, exist_ok=True)
        
        latest_checkpoint_path = os.path.join(DENSE_CHECKPOINT_DIR, "dense_latest.pth")
        torch.save(consolidated_checkpoint, latest_checkpoint_path)
        print(f"💾 Saved latest checkpoint to {latest_checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            consolidated_checkpoint['best_val_loss'] = best_val_loss 
            
            best_checkpoint_path = os.path.join(DENSE_CHECKPOINT_DIR, "dense_best.pth")
            torch.save(consolidated_checkpoint, best_checkpoint_path)
            print(f"🏆 New best model! Val loss: {avg_val_loss:.4f}. Saved to {best_checkpoint_path}")

    dist.barrier()

if local_rank == 0:
    end_time = time.time()
    duration_seconds = end_time - start_time
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    print(f"--- Total Training Time: {hours}h {minutes}m {seconds}s ---")

dist.destroy_process_group()
print("Job finished.")