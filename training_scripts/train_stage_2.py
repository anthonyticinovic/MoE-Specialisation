import time
import json
import yaml
import torch
import os
import gc
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

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from transformers.models.mistral.modeling_mistral import MistralMLP
from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM, MistralMoEDecoderLayer

AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

# ====================================================================================
# 2. SETUP AND CONFIGURATION
# ====================================================================================
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
train_params = config["training_stage2"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
STAGE2_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_checkpoints")

# --- Initialize the distributed environment ---
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = local_rank

if local_rank == 0:
    print("--- Initializing Stage 2 Training ---")
    print(f"PyTorch: {torch.__version__}")
print(f"--- Rank {local_rank} --- Using device: cuda:{DEVICE}")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
if local_rank == 0:
    print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

moe_model_path = paths["moe_model_path"]

if local_rank == 0:
    print(f"Loading custom MoE model from {moe_model_path}...")

llm = AutoModelForCausalLM.from_pretrained(
    moe_model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

if local_rank == 0:
    print("✅ Custom MoE model loaded on CPU.")

# ====================================================================================
# 4. TRAINING SETUP (PART 1 - Parameter Freezing)
# ====================================================================================
if local_rank == 0:
    print("Preparing model for Stage 2 training...")

for param in llm.parameters():
    param.requires_grad = False

for layer in llm.model.layers:
    if hasattr(layer.mlp, "experts"):
        for expert in layer.mlp.experts:
            for param in expert.parameters():
                param.requires_grad = True
        if local_rank == 0:
            print(f"✅ Unfroze {len(layer.mlp.experts)} experts in layer")

# ====================================================================================
# 5. FSDP WRAPPING
# ====================================================================================
print(f"--- Rank {local_rank} --- Wrapping model with FSDP...")

ignored_modules = [llm.model.embed_tokens]

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
    ignored_modules=ignored_modules,
)
print(f"--- Rank {local_rank} --- ✅ Model wrapped with FSDP.")

# --- MODIFIED: Robust 'best' and 'latest' checkpoint loading ---
latest_epoch = 0
best_val_loss = float('inf')
checkpoint_found = torch.tensor(0.0, device=DEVICE)

latest_checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, "llm_stage2_latest.pth")

if local_rank == 0:
    if os.path.exists(latest_checkpoint_path):
        checkpoint_found.fill_(1.0)
    else:
        print("�� No checkpoint found. Starting training from scratch.")

dist.broadcast(checkpoint_found, src=0)

if checkpoint_found.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Found latest checkpoint. Resuming training...")
    
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        if local_rank == 0:
            checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
            state_dict_to_load = checkpoint['model_state_dict']
            latest_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"✅ Resumed from epoch {latest_epoch}. Previous best validation loss: {best_val_loss:.4f}")
            
            del checkpoint
            gc.collect()
        else:
            state_dict_to_load = {}
        
        # Use strict=False for FSDP rank0_only loading
        llm.load_state_dict(state_dict_to_load, strict=False)
        
        if local_rank == 0:
            print(f"✅ Checkpoint loaded successfully!")

# Broadcast epoch and best_val_loss to all ranks
state_data = [latest_epoch, best_val_loss]
dist.broadcast_object_list(state_data, src=0)
latest_epoch, best_val_loss = int(state_data[0]), state_data[1]

dist.barrier()
if checkpoint_found.item() == 1.0:
    print(f"--- Rank {local_rank} --- ✅ Model weights and state synchronized.")

# Manually move the ignored module to the correct GPU device.
if local_rank == 0:
    print(f"Manually moving ignored modules to device {DEVICE}")
llm.model.embed_tokens.to(DEVICE)

llm.gradient_checkpointing_enable()

vision_connector = VisionLanguageConnector().to(DEVICE)
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth")
if os.path.exists(stage1_weights_path):
    if local_rank == 0:
        print(f"💾 Loading Stage 1 Vision Connector weights from {stage1_weights_path}")
    map_loc = f"cuda:{DEVICE}"
    state_dict = torch.load(stage1_weights_path, map_location=map_loc)
    vision_connector.load_state_dict(state_dict)
else:
    if local_rank == 0:
        print("🚨 WARNING: Stage 1 weights not found.")

for param in vision_encoder.parameters():
    param.requires_grad = False
for param in vision_connector.parameters():
    param.requires_grad = False

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
    seed=loader_params.get("data_seed", 42),  # Fixed seed for reproducibility
)
val_dataset = COCO_Loader(
    image_dir=paths["image_dir"],
    annotations_file=paths["annotations_file"],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"],
    split="val",
    seed=loader_params.get("data_seed", 42),  # Same seed ensures consistent splits
)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_params["batch_size"],
    sampler=train_sampler,
    num_workers=loader_params["num_workers"],
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=train_params["batch_size"],
    sampler=val_sampler,
    num_workers=loader_params["num_workers"],
    pin_memory=True,
)

accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
if local_rank == 0:
    print(f"Using gradient accumulation with {accumulation_steps} steps.")
    print(f"Effective batch size: {train_params['batch_size'] * accumulation_steps * dist.get_world_size()}")

trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=True)
scaler = GradScaler()
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

if local_rank == 0:
    print(f"Optimizing {sum(p.numel() for p in llm.parameters() if p.requires_grad)} trainable parameters.")

# --- Load optimizer & scheduler state if resuming ---
if checkpoint_found.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Loading optimizer and scheduler states from checkpoint...")
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✅ Loaded optimizer state (learning_rate: {optimizer.param_groups[0]['lr']:.2e})")
        else:
            print(f"  ⚠️ No optimizer state found in checkpoint (old format)")
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✅ Loaded scheduler state (last_epoch: {scheduler.last_epoch})")
        else:
            print(f"  ⚠️ No scheduler state found in checkpoint (old format)")
        
        del checkpoint
        gc.collect()
    
    dist.barrier()

metrics_history = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate":[]}
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_stage2.json")
if local_rank == 0 and latest_epoch > 0 and os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)

# ====================================================================================
# 8. TRAINING LOOP
# ====================================================================================
if local_rank == 0:
    print(f"🚀 Starting Stage 2 training from epoch {latest_epoch} for {NUM_EPOCHS} total epochs...")
start_time = time.time()
for epoch in range(latest_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    llm.train()
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

            text_embeddings = llm.model.embed_tokens(input_ids)
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            combined_embeddings.requires_grad_(True)

            routing_mask = torch.cat(
                [
                    torch.zeros(visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE),
                    torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=DEVICE),
                ],
                dim=1,
            )
            for layer in llm.model.layers:
                layer.mlp.routing_mask = routing_mask

            combined_attention_mask = torch.cat(
                [
                    torch.ones(visual_soft_tokens.shape[:2], device=DEVICE),
                    attention_mask,
                ],
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

            if text_logits.shape[1] == text_labels.shape[1]:
                loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))
                loss = loss / accumulation_steps
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

        scaler.scale(loss).backward()
        if loss.item() > 0:
            total_train_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if local_rank == 0 and (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}]")

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

                routing_mask = torch.cat(
                    [
                        torch.zeros(visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE),
                        torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=DEVICE),
                    ],
                    dim=1,
                )
                for layer in llm.model.layers:
                    layer.mlp.routing_mask = routing_mask

                combined_attention_mask = torch.cat(
                    [
                        torch.ones(visual_soft_tokens.shape[:2], device=DEVICE),
                        attention_mask,
                    ],
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

                if text_logits.shape[1] == text_labels.shape[1]:
                    loss = loss_fn(
                        text_logits.view(-1, llm.config.vocab_size),
                        text_labels.view(-1),
                    )
                else:
                    loss = torch.tensor(0.0, device=DEVICE)

                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    # --- MODIFIED: Synchronize avg_val_loss across all GPUs ---
    val_loss_tensor = torch.tensor(avg_val_loss).to(DEVICE)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    avg_val_loss = val_loss_tensor.item()
    
    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(current_lr)
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"✅ Metrics saved to {metrics_path}")

    # --- MODIFIED: 'best' and 'latest' checkpoint saving ---
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = llm.state_dict()

    if local_rank == 0:
        os.makedirs(STAGE2_CHECKPOINT_DIR, exist_ok=True)
        latest_model_path = os.path.join(STAGE2_CHECKPOINT_DIR, "llm_stage2_latest.pth")
        
        # Save the 'latest' model for resuming (with optimizer & scheduler)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': cpu_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_loss,
        }, latest_model_path)
        print(f"💾 Saved latest model checkpoint to {latest_model_path}")

        # Save the 'best' model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(STAGE2_CHECKPOINT_DIR, "llm_stage2_best.pth")
            # Save best checkpoint with full state for proper resumption
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': cpu_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'current_val_loss': avg_val_loss,
            }, best_model_path)
            print(f"🏆 New best model found! Validation loss: {avg_val_loss:.4f}. Saved to {best_model_path}")

    dist.barrier()

# --- 3. Calculate and print the total training time ---
if local_rank == 0:
    end_time = time.time()
    duration_seconds = end_time - start_time
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    print(f"--- Total Training Time: {hours}h {minutes}m {seconds}s ---")

dist.barrier()
dist.destroy_process_group()

print("Job finished.")
