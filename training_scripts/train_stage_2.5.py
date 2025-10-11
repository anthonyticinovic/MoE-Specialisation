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
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.custom_mistral import (
    MistralMoEConfig,
    MistralMoEForCausalLM,
    MistralMoEDecoderLayer,
)

from transformers.models.mistral.modeling_mistral import MistralMLP

# --- 1. Register Custom Architecture ---
AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

# ====================================================================================
# 2. SETUP AND CONFIGURATION
# ====================================================================================
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
train_params = config["training_stage2.5"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
STAGE2_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_checkpoints")
STAGE2_5_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_5_checkpoints")
LOAD_BALANCING_COEFF = train_params.get("load_balancing_coeff", 0.01)

# --- Initialize the distributed environment ---
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = f"cuda:{local_rank}"

if local_rank == 0:
    print("--- Initializing Stage 2.5 Training (Training the Router) ---")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
if local_rank == 0:
    print("Loading foundational models with low_cpu_mem_usage to prevent OOM...")
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

# ====================================================================================
# 4. TRAINING SETUP
# ====================================================================================
if local_rank == 0:
    print("Setting MoE layers to 'soft' routing mode for Stage 2.5.")
for layer in llm.model.layers:
    if hasattr(layer.mlp, "routing_mode"):
        layer.mlp.routing_mode = 'soft'

if local_rank == 0:
    print("Preparing model for Stage 2.5: Freezing experts, unfreezing routers.")
for name, param in llm.named_parameters():
    if "mlp.gate" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# ====================================================================================
# 5. FSDP WRAPPING & CHECKPOINTING
# ====================================================================================
my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={MistralMLP})
ignored_modules = [llm.model.embed_tokens]

llm = FSDP(
    llm,
    device_id=DEVICE,
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
    ),
    use_orig_params=True,
    ignored_modules=ignored_modules,
)

# --- 5.1. Load Stage 2 Checkpoint (Base weights for experts) ---
# Define the path to the best checkpoint directly
checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, "llm_stage2_best.pth")

# Have rank 0 check if the file exists
file_exists = 1.0 if local_rank == 0 and os.path.exists(checkpoint_path) else 0.0
file_exists_tensor = torch.tensor([file_exists], dtype=torch.float32).to(DEVICE)

# Broadcast the existence status to all other ranks
dist.broadcast(file_exists_tensor, src=0)
should_load = file_exists_tensor.item() == 1.0

# All ranks will attempt to load if the file was found on rank 0
if should_load:
    if local_rank == 0:
        print(f"💾 Loading Stage 2 BEST expert weights from: {checkpoint_path}")

    # Load the state dict. Your 'best' checkpoint saves the state_dict directly.
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load the state dict into the FSDP model
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(state_dict, strict=False)
    
    del state_dict
    gc.collect()
    
    # Barrier to ensure all processes have loaded before continuing
    dist.barrier()

    if local_rank == 0:
        print("✅ Stage 2 'best' state loaded successfully.")
else:
    if local_rank == 0:
        print(f"🚨 WARNING: Stage 2 best checkpoint not found at '{checkpoint_path}'. Starting from base weights.")

#ROBUST FIX: Manually re-initialize gates AFTER all loading is done
if local_rank == 0:
    print("--- Forcing re-initialization of router gates to fix corruption ---")

for layer in llm.module.model.layers:
    if hasattr(layer.mlp, "gate"):
        new_gate = nn.Linear(layer.mlp.d_model, layer.mlp.num_experts, bias=False)
        nn.init.normal_(new_gate.weight, std=0.01)
        new_gate = new_gate.to(DEVICE)
        layer.mlp.gate = new_gate
        layer.mlp.gate.weight.requires_grad = True

# ====================================================================================
# 6. DATA & OPTIMIZER
# ====================================================================================
if local_rank == 0:
    print("Creating datasets and dataloaders...")
train_dataset = COCO_Loader(
    image_dir=paths["image_dir"], annotations_file=paths["annotations_file"],
    clip_processor=clip_processor, tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"], split="train",
)
val_dataset = COCO_Loader(
    image_dir=paths["image_dir"], annotations_file=paths["annotations_file"],
    clip_processor=clip_processor, tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"], split="val",
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

# Create optimizer AFTER gates are fixed 
if local_rank == 0:
    print("Creating optimizer and scheduler with refreshed trainable parameters...")
accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(
    trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=True,
)
scaler = GradScaler()
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# --- 5.2. Resume from Stage 2.5 Checkpoint ---
start_epoch = 0
best_val_loss = float('inf')
latest_checkpoint_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, "llm_stage2_5_latest.pth")

# Have rank 0 check if the 'latest' checkpoint exists
should_resume = 1.0 if local_rank == 0 and os.path.exists(latest_checkpoint_path) else 0.0
resume_tensor = torch.tensor([should_resume], dtype=torch.float32).to(DEVICE)
dist.broadcast(resume_tensor, src=0)

if resume_tensor.item() == 1.0:
    if local_rank == 0:
        print(f"💾 Resuming training from latest checkpoint: {latest_checkpoint_path}")

    # All ranks load the checkpoint
    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint['model_state_dict']

    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(model_state_dict, strict=False)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
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

# --- 5.3. Finalize Model Setup ---
llm.model.embed_tokens.to(DEVICE)
#llm.gradient_checkpointing_enable()
vision_connector = VisionLanguageConnector().to(DEVICE)
vision_connector.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, "archive/vision_connector_stage1_best.pth"), map_location=DEVICE,)
)
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in vision_connector.parameters():
    param.requires_grad = False

if local_rank == 0:
    print(f"Optimizing {sum(p.numel() for p in trainable_params)} trainable router parameters.")

metrics_history = {
    "epoch": [], "train_loss": [], "train_ce_loss": [], "train_lb_loss": [],
    "val_loss": [], "learning_rate": [],
}
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_stage2.5.json")
if local_rank == 0 and start_epoch > 0 and os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)

# Optional: Verification block to confirm gate shapes are correct before training
if local_rank == 0:
    print("--- Verifying Gate Parameter Shapes Before Training ---")
    for i, layer in enumerate(llm.module.model.layers):
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "weight"):
            print(f"  Layer {i} gate shape: {tuple(layer.mlp.gate.weight.shape)}")
    print("---------------------------------------------------------")

# ====================================================================================
# 8. TRAINING LOOP
# ====================================================================================
if local_rank == 0:
    print(f"🚀 Starting Stage 2.5 training from epoch {start_epoch}...")
for epoch in range(start_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    llm.train()

    total_train_loss, total_ce_loss, total_lb_loss = 0, 0, 0
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
            combined_attention_mask = torch.cat(
                [torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask,],
                dim=1,
            )

            outputs = llm(
                inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask,
            )
            logits = outputs.logits

            num_visual_tokens = visual_soft_tokens.shape[1]
            text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
            text_labels = input_ids[..., 1:].contiguous()
            ce_loss = loss_fn(
                text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1)
            )

            total_load_balancing_loss = 0
            for layer in llm.module.model.layers:
                if hasattr(layer.mlp, "load_balancing_loss"):
                    total_load_balancing_loss += layer.mlp.load_balancing_loss

            loss = (
                ce_loss + LOAD_BALANCING_COEFF * total_load_balancing_loss
            ) / accumulation_steps

        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            # Step 1: Unscale gradients
            scaler.unscale_(optimizer)

            # Step 2: Collect all router parameters and compute their norm BEFORE clipping
            router_params = []
            for name, param in llm.named_parameters():
                if 'mlp.gate' in name and param.requires_grad:  # ← Remove grad check
                    router_params.append(param)

            if router_params:
                # Compute the ACTUAL gradient norm before clipping
                total_norm = 0.0
                for p in router_params:
                    if p.grad is not None:  # ← Add this check
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                # Now clip the gradients
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    router_params, 
                    max_norm=1.0
                )
                
                # Debug logging every 100 batches
                if local_rank == 0 and (i + 1) % 100 == 0:
                    print(f"--- Gradient Check ---")
                    print(f"  Router grad norm BEFORE clip: {total_norm:.2f}")
                    print(f"  Router grad norm AFTER clip: {clipped_norm:.2f}")
                    max_norm = 1.0
                    was_clipped = total_norm > max_norm
                    actual_norm_after_clip = min(total_norm, max_norm)
                    print(f"  Clipping applied: {'YES' if was_clipped else 'NO'}")
                    print(f"  Actual norm after clip: {actual_norm_after_clip:.2f}")
                    
                    # Sample a few layers to check individual norms
                    sample_layers = [0, 1, 15, 31]  # First, second, middle, last
                    for name, param in llm.named_parameters():
                        if 'mlp.gate' in name and param.grad is not None:
                            # Extract layer number from name (e.g., "model.layers.15.mlp.gate.weight")
                            match = re.search(r'layers\.(\d+)\.mlp\.gate', name)
                            if match:
                                layer_idx = int(match.group(1))
                                if layer_idx in sample_layers:
                                    layer_norm = param.grad.detach().data.norm(2).item()
                                    print(f"    Layer {layer_idx} gate: {layer_norm:.2e}")
                    print("----------------------")

            # Step 3: Update optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if loss.item() > 0:
            total_train_loss += loss.item() * accumulation_steps
            total_ce_loss += ce_loss.item()
            if isinstance(total_load_balancing_loss, torch.Tensor):
                total_lb_loss += total_load_balancing_loss.item()

        if local_rank == 0 and (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}]")

    avg_train_loss = total_train_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_lb_loss = total_lb_loss / len(train_loader)

    if local_rank == 0:
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f} | CE Loss: {avg_ce_loss:.4f} | LB Loss: {avg_lb_loss:.4f}"
        )

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
                combined_embeddings = torch.cat(
                    [visual_soft_tokens, text_embeddings], dim=1
                )
                combined_attention_mask = torch.cat(
                    [torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask,],
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

    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # --- Metrics and Checkpoint Saving ---
    if local_rank == 0:
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["train_ce_loss"].append(avg_ce_loss)
        metrics_history["train_lb_loss"].append(avg_lb_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"✅ Metrics saved to {metrics_path}")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, save_policy):
        full_state_dict = llm.state_dict()

    if local_rank == 0:
        # Get only the trainable router weights for the checkpoint
        router_weights = {}
        for name, weight in full_state_dict.items():
            param = llm.get_parameter(name)
            if param.requires_grad:
                router_weights[name] = weight
        
        # Create a single consolidated checkpoint for this epoch
        consolidated_checkpoint = {
            'model_state_dict': router_weights,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_loss,
        }

        os.makedirs(STAGE2_5_CHECKPOINT_DIR, exist_ok=True)
        
        # 1. Save this as the 'latest' checkpoint, overwriting the previous one
        latest_checkpoint_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, "llm_stage2_5_latest.pth")
        torch.save(consolidated_checkpoint, latest_checkpoint_path)
        print(f"💾 Saved latest checkpoint to {latest_checkpoint_path}")

        # 2. Check if this is the 'best' model and save it if so
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Update the best_val_loss in the checkpoint before saving as 'best'
            consolidated_checkpoint['best_val_loss'] = best_val_loss 
            
            best_checkpoint_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, "llm_stage2_5_best.pth")
            torch.save(consolidated_checkpoint, best_checkpoint_path)
            print(f"🏆 New best model! Val loss: {avg_val_loss:.4f}. Saved to {best_checkpoint_path}")

    dist.barrier()

dist.destroy_process_group()
print("Job finished.")
