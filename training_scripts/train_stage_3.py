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
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# Import your custom MoE classes
from models.custom_mistral import (
    MistralMoEConfig,
    MistralMoEForCausalLM,
    MistralMoEDecoderLayer,
)

from transformers.models.mistral.modeling_mistral import MistralMLP

# --- 1. Register Your Custom Architecture ---
AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

# ====================================================================================
# 2. SETUP AND CONFIGURATION
# ====================================================================================
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
# CHANGED: Use training_stage3 parameters from config
train_params = config["training_stage3"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
# CHANGED: Define all necessary checkpoint directories
STAGE2_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_checkpoints")
STAGE2_5_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_5_checkpoints")
STAGE3_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage3_checkpoints")
# ADDED: Load balancing coefficient for MoE training
LOAD_BALANCING_COEFF = train_params.get("load_balancing_coeff", 0.01)

# --- Initialize the distributed environment ---
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = local_rank

if local_rank == 0:
    # CHANGED: Print statement for Stage 3
    print("--- Initializing Stage 3 Training (End-to-End) ---")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
if local_rank == 0:
    print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

moe_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"

llm = AutoModelForCausalLM.from_pretrained(
    moe_model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

llm.gradient_checkpointing_enable()

# Explicitly set all MoE layers to use soft routing for training
if local_rank == 0:
    print("Setting MoE layers to 'soft' routing mode for Stage 3.")
for layer in llm.model.layers:
    if hasattr(layer.mlp, "routing_mode"):
        layer.mlp.routing_mode = 'soft'
# ====================================================================================
# 4. TRAINING SETUP (PART 1 - Parameter Freezing)
# ====================================================================================
if local_rank == 0:
    print("Preparing model for Stage 3: Selective unfreezing (self-attn, router, MLP only).")

# Freeze all LLM parameters first
for param in llm.parameters():
    param.requires_grad = False

# Selectively unfreeze: self-attention, router (gate), and MLP layers
for name, param in llm.named_parameters():
    if any(x in name for x in ['self_attn', 'mlp.gate', 'mlp.experts']):
        param.requires_grad = True
        if local_rank == 0 and 'layers.0' in name:  # Print first layer as example
            print(f"  Unfrozen: {name}")

vision_connector = VisionLanguageConnector().to(DEVICE)
#  Unfreeze vision connector for end-to-end training
for param in vision_connector.parameters():
    param.requires_grad = True

# Ensure the vision encoder remains frozen
for param in vision_encoder.parameters():
    param.requires_grad = False

if local_rank == 0:
    trainable_count = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in llm.parameters())
    print(f"LLM: {trainable_count:,} / {total_count:,} parameters trainable ({100*trainable_count/total_count:.1f}%)")

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
    cpu_offload=CPUOffload(offload_params=False),
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    use_orig_params=True,
)

load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)

# Load Stage 2 weights first (expert weights)
stage2_checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, "llm_stage2_best.pth")
if os.path.exists(stage2_checkpoint_path):
    if local_rank == 0:
        print(f"💾 Loading Stage 2 (Expert) weights from: {stage2_checkpoint_path}")
    
    checkpoint = torch.load(stage2_checkpoint_path, map_location="cpu")
    s2_state_dict = checkpoint['model_state_dict']  # Extract nested dict
    
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        missing_keys, unexpected_keys = llm.load_state_dict(s2_state_dict, strict=False)
        if local_rank == 0:
            print(f"  Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    del checkpoint, s2_state_dict
    gc.collect()
    if local_rank == 0:
        print("✅ Stage 2 expert weights loaded.")
else:
    if local_rank == 0:
        print("⚠️ WARNING: Stage 2 checkpoint not found. Using base MoE weights.")

dist.barrier()

# Load Stage 2.5 router weights separately (safer!)
stage2_5_checkpoint_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, "llm_stage2_5_best.pth")
if os.path.exists(stage2_5_checkpoint_path):
    if local_rank == 0:
        print(f"💾 Loading Stage 2.5 (Router-only) weights from: {stage2_5_checkpoint_path}")
    
    checkpoint = torch.load(stage2_5_checkpoint_path, map_location="cpu")
    router_state_dict = checkpoint['model_state_dict']
    
    # Load router weights directly into the model
    loaded_count = 0
    with torch.no_grad():
        for name, param in llm.named_parameters():
            if name in router_state_dict and 'mlp.gate' in name:
                param.data.copy_(router_state_dict[name].to(param.device))
                loaded_count += 1
    
    if local_rank == 0:
        print(f"  Loaded {loaded_count} router parameters")
    
    del checkpoint, router_state_dict
    gc.collect()
    if local_rank == 0:
        print("✅ Stage 2.5 router weights loaded.")
else:
    if local_rank == 0:
        print("⚠️ WARNING: Stage 2.5 checkpoint not found. Using random router initialization.")

dist.barrier()

# 3. Load Stage 1 Vision Connector weights
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth")
if os.path.exists(stage1_weights_path):
    if local_rank == 0:
        print(f"💾 Loading Stage 1 Vision Connector weights from {stage1_weights_path}")
    map_loc = f"cuda:{DEVICE}"
    vision_connector.load_state_dict(torch.load(stage1_weights_path, map_location=map_loc))

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

# NEW: Combine all trainable parameters (LLM + Vision Connector) for the optimizer
trainable_params = list(llm.parameters()) + list(vision_connector.parameters())
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=True)
scaler = GradScaler()
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# --- Resumption logic for Stage 3 ---
start_epoch = 0
best_val_loss = float('inf')
latest_checkpoint_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_latest.pth")

should_resume = 1.0 if local_rank == 0 and os.path.exists(latest_checkpoint_path) else 0.0
resume_tensor = torch.tensor([should_resume], dtype=torch.float32).to(DEVICE)
dist.broadcast(resume_tensor, src=0)

if resume_tensor.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Resuming Stage 3 training from latest checkpoint: {latest_checkpoint_path}")

    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
    
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(checkpoint['model_state_dict'])
    vision_connector.load_state_dict(checkpoint['connector_state_dict'])
    
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

metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_stage3.json")
metrics_history = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate":[]}
if local_rank == 0 and start_epoch > 0 and os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)

# After loading all checkpoints and before the training loop, add this debugging:
if local_rank == 0:
    print("=== DEBUGGING EMBEDDING LAYER ===")
    embed_layer = llm.module.model.embed_tokens
    print(f"Embedding weight shape: {embed_layer.weight.shape}")
    print(f"Embedding weight device: {embed_layer.weight.device}")
    print(f"Embedding weight dtype: {embed_layer.weight.dtype}")
    print("=================================")

dist.barrier()
# ====================================================================================
# 8. TRAINING LOOP
# ====================================================================================
if local_rank == 0:
    print(f"🚀 Starting Stage 3 training from epoch {start_epoch}...")
start_time = time.time()
for epoch in range(start_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    llm.train()
    vision_connector.train()
    total_train_loss = 0
    optimizer.zero_grad()
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
        )

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
            
            visual_soft_tokens = vision_connector(patch_embeddings)
            text_embeddings = llm.module.model.embed_tokens(input_ids)
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
            
            # Initialize with correct device AND dtype to match the model
            total_load_balancing_loss = torch.tensor(0.0, device=DEVICE, dtype=torch.bfloat16)
            
            for layer in llm.module.model.layers:
                # Check if the attribute exists and is a tensor
                if hasattr(layer.mlp, "load_balancing_loss") and isinstance(layer.mlp.load_balancing_loss, torch.Tensor):
                    # This .to(DEVICE) is a safeguard that forces the tensor to the GPU
                    total_load_balancing_loss += layer.mlp.load_balancing_loss.to(DEVICE)

            num_visual_tokens = visual_soft_tokens.shape[1]
            text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
            text_labels = input_ids[..., 1:].contiguous()
            
            ce_loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))
            
            if i == 0: # Only print for the first batch to avoid spamming logs
                print("\n--- DEBUG INFO ---")
                print(f"  - text_logits device: {text_logits.device}")
                print(f"  - text_labels device: {text_labels.device}")
                print(f"  - ce_loss device: {ce_loss.device}")
                print(f"  - total_load_balancing_loss device: {total_load_balancing_loss.device}")
                print("--------------------\n")

            loss = (ce_loss + LOAD_BALANCING_COEFF * total_load_balancing_loss) / accumulation_steps

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
    vision_connector.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
            )
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.module.model.embed_tokens(input_ids)
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
    
    connector_state_dict = vision_connector.state_dict()

    if local_rank == 0:
        consolidated_checkpoint = {
            'model_state_dict': llm_state_dict,
            'connector_state_dict': connector_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_loss,
        }

        os.makedirs(STAGE3_CHECKPOINT_DIR, exist_ok=True)
        
        latest_checkpoint_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_latest.pth")
        torch.save(consolidated_checkpoint, latest_checkpoint_path)
        print(f"💾 Saved latest checkpoint to {latest_checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            consolidated_checkpoint['best_val_loss'] = best_val_loss 
            
            best_checkpoint_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_best.pth")
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
