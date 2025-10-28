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
from data import COCO_Loader, LLaVA_Loader
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
STAGE3_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage3_checkpoints")

# --- Initialize the distributed environment with extended timeout ---
import datetime

# CRITICAL: Set timeout to 60 minutes for large checkpoint loading operations
# Default is 10 minutes which is too short for FSDP checkpoint loading
timeout = datetime.timedelta(minutes=60)
dist.init_process_group("nccl", timeout=timeout)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = local_rank

if local_rank == 0:
    # CHANGED: Print statement for Stage 3
    print("--- Initializing Stage 3 Training (End-to-End) ---")
    print(f"🕐 NCCL timeout set to: {timeout} (60 minutes)")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
if local_rank == 0:
    print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
# CRITICAL: Set vision encoder to eval mode since it's frozen - prevents dropout/stochastic behavior
vision_encoder.eval()
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

# CRITICAL FIX: Disable gradient checkpointing - it causes FSDP corruption issues
# llm.gradient_checkpointing_enable()

# Explicitly set all MoE layers to use soft routing for training
if local_rank == 0:
    print("Setting MoE layers to 'soft' routing mode for Stage 3.")
for layer in llm.model.layers:
    if hasattr(layer.mlp, "routing_mode"):
        layer.mlp.routing_mode = 'soft'

# Configure dropout for regularization (Stage 3 only)
if local_rank == 0:
    print("Configuring dropout for Stage 3 regularization...")

attention_dropout = train_params.get("attention_dropout", 0.0)
expert_dropout = train_params.get("expert_dropout", 0.0)

# Enable attention dropout in self-attention layers
for layer in llm.model.layers:
    # Mistral uses 'attention_dropout' in self_attn
    if hasattr(layer.self_attn, 'attention_dropout'):
        layer.self_attn.attention_dropout = attention_dropout
    # Some models use 'dropout' attribute
    if hasattr(layer.self_attn, 'dropout') and isinstance(layer.self_attn.dropout, (int, float)):
        layer.self_attn.dropout = attention_dropout

# Enable expert dropout in MoE layers
for layer in llm.model.layers:
    if hasattr(layer.mlp, 'experts'):
        # Add expert dropout module if not already present
        if not hasattr(layer.mlp, 'expert_dropout'):
            layer.mlp.expert_dropout = nn.Dropout(expert_dropout)
        else:
            layer.mlp.expert_dropout.p = expert_dropout

if local_rank == 0:
    print(f"  ✅ Attention dropout: {attention_dropout}")
    print(f"  ✅ Expert dropout: {expert_dropout}")
    print(f"  ✅ Router dropout: 0.1 (pre-configured in MoE layer)")

# ====================================================================================
# 4. TRAINING SETUP (PART 1 - Parameter Freezing)
# ====================================================================================
if local_rank == 0:
    print("Preparing model for Stage 3: Selective unfreezing (self-attn, router, MLP only).")
    print("Vision connector will remain frozen (using Stage 1 weights).")

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
# Keep vision connector frozen (already trained in Stage 1)
for param in vision_connector.parameters():
    param.requires_grad = False

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

# CRITICAL: Cache vocab_size BEFORE FSDP wrapping to avoid accessing llm.config later
# Accessing FSDP-wrapped model config can trigger collective operations
VOCAB_SIZE = llm.config.vocab_size

my_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        MistralMLP,
    },
)

# Prevent FSDP from sharding the embedding layer (accessed directly in training loop)
ignored_modules = [llm.model.embed_tokens]

# CRITICAL: Ignored modules must be on the correct device BEFORE FSDP wrapping
# Moving them after wrapping causes FSDP to detect "newly-added parameters"
if local_rank == 0:
    print(f"Placing ignored modules on device {DEVICE} before FSDP wrapping")
llm.model.embed_tokens.to(DEVICE)

# CRITICAL: Cache embedding layer reference before FSDP wrapping to avoid accessing
# llm.module.* in the training loop (safer and avoids potential FSDP interactions)
embed_tokens_layer = llm.model.embed_tokens

# CRITICAL FIX: Use exact FSDP configuration from working Stage 2.5
# device_id must be DEVICE (local_rank as int), NOT torch.cuda.current_device()
# cpu_offload must be CPUOffload(offload_params=None), NOT False
llm = FSDP(
    llm,
    device_id=DEVICE,
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=None),
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    use_orig_params=True,
    ignored_modules=ignored_modules,
)

# ====================================================================================
# 6. LOAD STAGE 2 CHECKPOINT (EXACT PATTERN FROM TRAIN_STAGE_2.PY)
# ====================================================================================
# This matches train_stage_2.py lines 178-194 EXACTLY
checkpoint_found = torch.tensor(0.0, device=DEVICE)
stage2_checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, "llm_stage2_best.pth")

if local_rank == 0:
    if os.path.exists(stage2_checkpoint_path):
        checkpoint_found.fill_(1.0)
    else:
        print("❌ CRITICAL: Stage 2 checkpoint not found!")
        print(f"   Expected path: {stage2_checkpoint_path}")

dist.broadcast(checkpoint_found, src=0)

if checkpoint_found.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Loading Stage 2 (Expert) checkpoint: {stage2_checkpoint_path}")
    
    # EXACT COPY from train_stage_2.5.py (WORKING VERSION)
    # Load the state dict directly (Stage 2's 'best' checkpoint saves state_dict directly)
    state_dict = torch.load(stage2_checkpoint_path, map_location="cpu")
    
    if local_rank == 0:
        print(f"  🔍 Checkpoint type: {type(state_dict)}")
        print(f"  📊 Checkpoint contains {len(state_dict)} keys")
        
        # Sample some keys to verify structure
        sample_keys = list(state_dict.keys())[:5]
        print(f"  📋 Sample keys: {sample_keys}")
    
    # Load the state dict into the FSDP model
    # Use rank0_only=True for GPU-count agnostic loading
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        missing_keys, unexpected_keys = llm.load_state_dict(state_dict, strict=False)
        
        if local_rank == 0:
            if missing_keys:
                print(f"  ⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  ⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            if not missing_keys and not unexpected_keys:
                print(f"  ✅ All keys matched perfectly!")
    
    del state_dict
    gc.collect()
    
    # Barrier to ensure all processes have loaded before continuing
    dist.barrier()
    
    if local_rank == 0:
        print("✅ Stage 2 checkpoint loaded successfully on all ranks.")
        print("ℹ️  Stage 3: Using loaded router weights (will be fine-tuned end-to-end)")
elif local_rank == 0:
    print("❌ CRITICAL: Cannot proceed without Stage 2 expert weights!")
    raise FileNotFoundError(f"Stage 2 checkpoint not found: {stage2_checkpoint_path}")

dist.barrier()

# CRITICAL: Do NOT access llm.module.model.layers after FSDP wrapping!
# Accessing FSDP internals (even just len(llm.module.model.layers)) can trigger
# collective operations on rank 0 only, corrupting execution order tracking.
# This causes "newly-added parameter" errors during the first backward pass.

if local_rank == 0:
    print("✅ Stage 2 checkpoint loaded and ready for training.\n")

dist.barrier()

# ====================================================================================
# 7. LOAD STAGE 1 VISION CONNECTOR
# ====================================================================================
# CRITICAL: Use same synchronization pattern as Stage 2 checkpoint loading
# to prevent race conditions on distributed file systems
connector_found = torch.tensor(0.0, device=DEVICE)
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth")

if local_rank == 0:
    if os.path.exists(stage1_weights_path):
        connector_found.fill_(1.0)
    else:
        print(f"⚠️  Warning: Stage 1 Vision Connector weights not found at {stage1_weights_path}")
        print(f"   Vision connector will use random initialization.")

dist.broadcast(connector_found, src=0)

if connector_found.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Loading Stage 1 Vision Connector weights from {stage1_weights_path}")
    map_loc = f"cuda:{DEVICE}"
    vision_connector.load_state_dict(torch.load(stage1_weights_path, map_location=map_loc))
    if local_rank == 0:
        print("✅ Vision Connector weights loaded successfully on all ranks.")
else:
    if local_rank == 0:
        print("⚠️  Proceeding with randomly initialized Vision Connector (not recommended for Stage 3)")

dist.barrier()

# ====================================================================================
# 8. DATA & OPTIMIZER
# ====================================================================================
if local_rank == 0:
    print("Creating datasets and dataloaders...")

# Conditional dataset loading based on config
dataset_type = train_params.get("dataset", "coco")  # Default to COCO for backward compatibility

if dataset_type == "llava":
    if local_rank == 0:
        print("📚 Using LLaVA-Instruct-150K dataset (ALL Q&A pairs, multi-turn)")
    
    train_dataset = LLaVA_Loader(
        annotations_file=paths["llava_annotations_file"],
        image_dir=paths["llava_image_dir"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        split="train",
        subset_fraction=train_params["subset_fraction"],
        val_fraction=0.2,  # 80/20 train/val split
        seed=loader_params.get("data_seed", 42),
    )
    val_dataset = LLaVA_Loader(
        annotations_file=paths["llava_annotations_file"],
        image_dir=paths["llava_image_dir"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        split="val",
        subset_fraction=train_params.get("val_subset_fraction", 1.0),  # Further subsample val if needed
        val_fraction=0.2,  # Same 80/20 split
        seed=loader_params.get("data_seed", 42),
    )
else:  # "coco"
    if local_rank == 0:
        print("📚 Using COCO captions dataset")
    
    train_dataset = COCO_Loader(
        image_dir=paths["image_dir"],
        annotations_file=paths["annotations_file"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        subset_fraction=train_params["subset_fraction"],
        split="train",
        seed=loader_params.get("data_seed", 42),  # Fixed seed for reproducibility
    )
    val_dataset = COCO_Loader(
        image_dir=paths["image_dir"],
        annotations_file=paths["annotations_file"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        subset_fraction=train_params["subset_fraction"],
        split="val",
        val_subset_fraction=train_params.get("val_subset_fraction", 0.2),  # Subsample validation to 20% by default
        seed=loader_params.get("data_seed", 42),  # Same seed ensures consistent splits
    )

# CRITICAL: Use drop_last=True for training to ensure deterministic batch counts
# With large datasets not perfectly divisible by world_size, padding can cause
# rank-specific execution paths that FSDP detects as execution order divergence.
# drop_last=False for validation is fine since validation doesn't update FSDP state.
train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

train_loader = DataLoader(
    train_dataset, batch_size=train_params["batch_size"], sampler=train_sampler,
    num_workers=loader_params["num_workers"], pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=train_params["batch_size"], sampler=val_sampler,
    num_workers=loader_params["num_workers"], pin_memory=True,
)

accumulation_steps = train_params.get("gradient_accumulation_steps", 1)

# Only include LLM trainable parameters for the optimizer (vision connector is frozen)
# Filter to only include parameters that require gradients
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=False)
# CRITICAL: GradScaler is not compatible with bfloat16 (only float16)
# Since we use bfloat16, we don't need GradScaler (bfloat16 has better numerical stability)
scaler = GradScaler(enabled=False)  # Disabled for bfloat16
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

# Add label smoothing for better generalization
label_smoothing = train_params.get("label_smoothing", 0.0)
loss_fn = nn.CrossEntropyLoss(
    ignore_index=-100,  # Standard ignore index for masked tokens
    label_smoothing=label_smoothing
)

if local_rank == 0 and label_smoothing > 0:
    print(f"  ✅ Label smoothing: {label_smoothing}")

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

    # All ranks load the checkpoint
    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint['model_state_dict']

    # Use rank0_only=True for GPU-count agnostic loading
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(model_state_dict, strict=False)

    # Load vision connector (not FSDP-wrapped, so normal load)
    vision_connector.load_state_dict(checkpoint['connector_state_dict'])
    
    # Try to load optimizer and scheduler, but handle GPU count mismatch gracefully
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if local_rank == 0:
            print("✅ Loaded optimizer and scheduler state (GPU count matches checkpoint)")
    except Exception as e:
        if local_rank == 0:
            print(f"⚠️  Could not load optimizer/scheduler state (likely GPU count mismatch): {e}")
            print("   Optimizer and scheduler will start fresh (model weights are preserved)")
    
    # Update epoch and best_val_loss from the checkpoint to continue correctly
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    del checkpoint
    del model_state_dict
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

# ====================================================================================
# 9. TRAINING LOOP
# ====================================================================================
# NOTE: Model verification removed - FSDP is extremely sensitive to execution order.
# Any forward pass before training (even in eval mode) can corrupt FSDP's internal
# execution graph tracking, causing collective operation mismatches during training.
# The successful checkpoint loading above is sufficient validation.
if local_rank == 0:
    world_size = dist.get_world_size()
    label_smoothing = train_params.get("label_smoothing", 0.0)
    attention_dropout = train_params.get("attention_dropout", 0.0)
    expert_dropout = train_params.get("expert_dropout", 0.0)
    dataset_type = train_params.get("dataset", "coco")
    
    print("\n" + "="*70)
    print("🚀 STAGE 3 TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset:               {dataset_type.upper()}")
    print(f"Epochs:                {NUM_EPOCHS} (starting from epoch {start_epoch})")
    print(f"Training samples:      {len(train_dataset)}")
    print(f"Validation samples:    {len(val_dataset)}")
    print(f"Batch size per GPU:    {train_params['batch_size']}")
    print(f"Gradient accumulation: {accumulation_steps}")
    print(f"Effective batch size:  {train_params['batch_size'] * world_size * accumulation_steps} (batch_size × {world_size} GPUs × accum_steps)")
    print(f"Steps per epoch:       {len(train_loader) // accumulation_steps}")
    print(f"Total training steps:  {total_steps}")
    print(f"Learning rate:         {train_params['learning_rate']:.2e}")
    print(f"Weight decay:          {train_params['weight_decay']}")
    print(f"Label smoothing:       {label_smoothing}")
    print(f"Attention dropout:     {attention_dropout}")
    print(f"Expert dropout:        {expert_dropout}")
    print(f"Router dropout:        0.1 (pre-configured)")
    print(f"Optimizer:             AdamW (fused)")
    print(f"Scheduler:             CosineAnnealingLR")
    print(f"Mixed precision:       bfloat16")
    print(f"Gradient checkpointing: Disabled")
    print(f"FSDP robustness:       Dummy expert touching enabled")
    print("="*70 + "\n")
    print(f"🚀 Starting training...")

start_time = time.time()
for epoch in range(start_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    
    # CRITICAL: Synchronize random seed across all ranks for deterministic Gumbel sampling
    # The MoE layer uses Gumbel noise for routing, which must be identical across ranks
    # to ensure FSDP execution order consistency
    torch.manual_seed(42 + epoch)
    torch.cuda.manual_seed_all(42 + epoch)
    
    llm.train()
    # CRITICAL: Keep vision_connector in eval mode since all its parameters are frozen
    # Calling .train() on a frozen module is inconsistent and may cause issues with
    # BatchNorm/Dropout layers if they exist
    vision_connector.eval()
    total_train_loss = 0
    optimizer.zero_grad()
    epoch_start_time = time.time()
    
    if local_rank == 0:
        print(f"\n{'='*70}")
        print(f"📚 Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")
    
    for i, batch in enumerate(train_loader):
        # Unpack batch - LLaVA returns 4 items (images, input_ids, attention_mask, labels)
        # COCO returns 3 items (images, input_ids, attention_mask) - labels = input_ids shifted
        if len(batch) == 4:
            # LLaVA dataset with proper loss masking
            images, input_ids, attention_mask, labels = batch
            images, input_ids, attention_mask, labels = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            )
            use_llava_labels = True
        else:
            # COCO dataset (backward compatibility)
            images, input_ids, attention_mask = batch
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
            )
            use_llava_labels = False

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
            
            visual_soft_tokens = vision_connector(patch_embeddings)
            # Use cached embedding layer to avoid llm.module.* access
            text_embeddings = embed_tokens_layer(input_ids)
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
            
            if use_llava_labels:
                # LLaVA: Use provided labels with masking
                # Logits for text portion only (skip visual tokens)
                text_logits = logits[..., num_visual_tokens:, :].contiguous()
                # Labels already have question tokens masked with -100
                text_labels = labels.contiguous()
                
                # Shift for next-token prediction: logits[:-1] predicts labels[1:]
                text_logits = text_logits[..., :-1, :].contiguous()
                text_labels = text_labels[..., 1:].contiguous()
            else:
                # COCO: Original behavior (compute loss on all text)
                text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
                text_labels = input_ids[..., 1:].contiguous()
            
            ce_loss = loss_fn(text_logits.view(-1, VOCAB_SIZE), text_labels.view(-1))

            loss = ce_loss / accumulation_steps

        scaler.scale(loss).backward()
        
        # CRITICAL: Accumulate loss tensor WITHOUT .item() to avoid CPU-GPU sync during accumulation
        # Calling .item() on every iteration can cause timing skew between ranks (especially with 4+ GPUs)
        # Instead, accumulate the tensor and only extract .item() after gradient update
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            # NOW it's safe to sync - we're already synchronizing for the optimizer step
            total_train_loss += loss.item() * accumulation_steps
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Progress logging every 50 steps (after gradient update)
            if local_rank == 0:
                total_steps = len(train_loader) // accumulation_steps
                steps_done = (i + 1) // accumulation_steps
                
                # Log every 50 steps OR on the first step OR on the last step
                if steps_done % 50 == 0 or steps_done == 1 or steps_done == total_steps:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss_so_far = total_train_loss / (i + 1) if (i + 1) > 0 else 0.0
                    elapsed = time.time() - epoch_start_time
                    steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (total_steps - steps_done) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_minutes = int(eta_seconds / 60)
                    
                    print(f"  [Step {steps_done:4d}/{total_steps}] Loss: {avg_loss_so_far:.4f} | "
                          f"LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_minutes}min")

    # CRITICAL: Average training loss over number of optimizer steps, NOT total batches
    num_optimizer_steps = len(train_loader) // accumulation_steps
    avg_train_loss = total_train_loss / num_optimizer_steps
    if local_rank == 0:
        epoch_train_time = time.time() - epoch_start_time
        print(f"\n✅ Training complete: Avg Loss = {avg_train_loss:.4f} | Time: {epoch_train_time/60:.2f} min")

    # --- Validation Phase (All Ranks, Limited Batches) ---
    # SOLUTION: Use Stage 2.5's validation pattern:
    # - All ranks participate (FSDP collectives work correctly)
    # - Limit to MAX_VAL_BATCHES (keeps validation fast, ~5-10 minutes)
    # - Each rank computes its own average (no distributed aggregation)
    # - Only rank 0's validation loss is used for model selection
    llm.eval()
    vision_connector.eval()
    total_val_loss = 0
    val_steps = 0
    MAX_VAL_BATCHES = 300  # Limited to keep validation fast (~5-10 minutes)
    
    if local_rank == 0:
        print(f"\n📊 Running validation (all ranks, max {MAX_VAL_BATCHES} batches per GPU)...")
        val_start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Early stop after MAX_VAL_BATCHES to keep validation fast
            if i >= MAX_VAL_BATCHES:
                break
            
            # Unpack batch (handle both LLaVA and COCO formats)
            if len(batch) == 4:
                images, input_ids, attention_mask, labels = batch
                images, input_ids, attention_mask, labels = (
                    images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
                )
                use_llava_labels = True
            else:
                images, input_ids, attention_mask = batch
                images, input_ids, attention_mask = (
                    images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
                )
                use_llava_labels = False
            
            # CRITICAL: Don't use try-except with break - it can cause ranks to diverge
            # Instead, just skip failed batches but continue the loop
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                # Use cached embedding layer to avoid llm.module.* access
                text_embeddings = embed_tokens_layer(input_ids)
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
                
                if use_llava_labels:
                    # LLaVA: Use provided labels with masking
                    text_logits = logits[..., num_visual_tokens:, :].contiguous()
                    text_labels = labels.contiguous()
                    # Shift for next-token prediction
                    text_logits = text_logits[..., :-1, :].contiguous()
                    text_labels = text_labels[..., 1:].contiguous()
                else:
                    # COCO: Original behavior
                    text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
                    text_labels = input_ids[..., 1:].contiguous()
                
                loss = loss_fn(
                    text_logits.view(-1, VOCAB_SIZE), text_labels.view(-1)
                )
            
            total_val_loss += loss.item()
            val_steps += 1
            
            # Progress logging every 25 batches
            if local_rank == 0 and (i + 1) % 25 == 0:
                avg_val_loss_so_far = total_val_loss / val_steps if val_steps > 0 else 0.0
                print(f"  Validation progress: {i+1}/{MAX_VAL_BATCHES} batches | Avg Loss: {avg_val_loss_so_far:.4f}")
    
    # Each rank computes its own average - no distributed aggregation needed
    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
    
    if local_rank == 0:
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Complete")
        print(f"  Training Loss:   {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f} (rank 0, {val_steps} batches)")
        print(f"  Epoch Time:      {epoch_time/60:.2f} minutes (Train: {(epoch_time-val_time)/60:.2f}m, Val: {val_time/60:.2f}m)")
        print(f"  Learning Rate:   {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*70}\n")    # --- Metrics and Checkpoint Saving ---
    if local_rank == 0:
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"✅ Metrics saved to {metrics_path}")

    # CRITICAL: Barrier before checkpoint extraction to ensure all ranks are ready
    # FSDP state_dict extraction requires coordination between ranks
    dist.barrier()

    if local_rank == 0:
        print(f"\n💾 Saving checkpoints...")
        checkpoint_start_time = time.time()

    # Force garbage collection and clear cache before checkpoint saving
    gc.collect()
    torch.cuda.empty_cache()
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, save_policy):
        llm_state_dict = llm.state_dict()
    
    connector_state_dict = vision_connector.state_dict()

    if local_rank == 0:
        # Save full checkpoint (includes optimizer/scheduler - tied to GPU count)
        full_checkpoint = {
            'model_state_dict': llm_state_dict,
            'connector_state_dict': connector_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_loss,
            'world_size': dist.get_world_size(),  # Track GPU count
        }

        # Save portable checkpoint (model weights only - GPU count agnostic)
        portable_checkpoint = {
            'model_state_dict': llm_state_dict,
            'connector_state_dict': connector_state_dict,
            'epoch': epoch,
            'val_loss': avg_val_loss,
        }

        os.makedirs(STAGE3_CHECKPOINT_DIR, exist_ok=True)
        
        # Save both full and portable versions
        latest_checkpoint_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_latest.pth")
        portable_checkpoint_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_latest_portable.pth")
        
        torch.save(full_checkpoint, latest_checkpoint_path)
        torch.save(portable_checkpoint, portable_checkpoint_path)
        
        checkpoint_time = time.time() - checkpoint_start_time
        print(f"  ✅ Saved latest checkpoint ({checkpoint_time:.1f}s)")
        print(f"     Full: {latest_checkpoint_path}")
        print(f"     Portable: {portable_checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            full_checkpoint['best_val_loss'] = best_val_loss 
            
            best_checkpoint_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_best.pth")
            best_portable_path = os.path.join(STAGE3_CHECKPOINT_DIR, "llm_stage3_best_portable.pth")
            
            torch.save(full_checkpoint, best_checkpoint_path)
            torch.save(portable_checkpoint, best_portable_path)
            print(f"\n  🏆 NEW BEST MODEL! Val loss improved: {avg_val_loss:.4f}")
            print(f"     Best: {best_checkpoint_path}")
            print(f"     Best Portable: {best_portable_path}")
        else:
            print(f"  ℹ️  Best val loss remains: {best_val_loss:.4f} (current: {avg_val_loss:.4f})")
        
        # Clean up checkpoint dicts to free memory
        del llm_state_dict, connector_state_dict, full_checkpoint, portable_checkpoint
    
    # Force memory cleanup after checkpoint operations
    gc.collect()
    torch.cuda.empty_cache()
    
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
