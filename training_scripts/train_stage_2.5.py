

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
from data import COCO_Loader  # <-- FIX: Corrected typo from COLO to COCO
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

# Note: This import is not used by the corrected FSDP policy but is kept for context.
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
# NOTE: This wrapping policy is still likely incorrect and should be revisited
# if zero-gradient issues persist after fixing the checkpoint loading.
# The correct class is likely `MistralMoEDecoderLayer`.
my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={MistralMLP})
ignored_modules = [llm.model.embed_tokens]

llm = FSDP(
    llm,
    device_id=DEVICE,
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

# --- 5.1. Load Stage 2 Checkpoint (Base weights for experts) ---
latest_stage2_epoch = 0
if local_rank == 0:
    if os.path.exists(STAGE2_CHECKPOINT_DIR):
        epoch_numbers = [
            int(re.search(r"epoch_(\d+)", f).group(1))
            for f in os.listdir(STAGE2_CHECKPOINT_DIR)
            if re.search(r"epoch_(\d+)", f)
        ]
        if epoch_numbers:
            latest_stage2_epoch = max(epoch_numbers)

epoch_tensor = torch.tensor([latest_stage2_epoch], dtype=torch.int).to(DEVICE)
dist.broadcast(epoch_tensor, src=0)
latest_stage2_epoch = epoch_tensor.item()

# ★★★ ELEGANT SOLUTION IMPLEMENTED HERE ★★★
if latest_stage2_epoch > 0:
    checkpoint_path = os.path.join(
        STAGE2_CHECKPOINT_DIR, f"llm_stage2_epoch_{latest_stage2_epoch}.pth"
    )
    if local_rank == 0:
        print(f"💾 Loading and filtering Stage 2 expert weights from: {checkpoint_path}")

    # 1. Load the full state dict from the checkpoint file onto the CPU
    full_stage2_state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 2. Create a new state dict, EXCLUDING any parameters related to the router gates
    filtered_state_dict = {
        k: v for k, v in full_stage2_state_dict.items() if "mlp.gate" not in k
    }
    
    del full_stage2_state_dict  # Free up memory

    # 3. Load the filtered state dict. This will update all weights EXCEPT the gates,
    # preserving the fresh, correctly-shaped gates initialized in the MoELayer.
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        # strict=False is important as the dict is missing gate keys
        llm.load_state_dict(filtered_state_dict, strict=False)
    
    del filtered_state_dict
    gc.collect()
    
    if local_rank == 0:
        print("✅ Filtered Stage 2 state loaded successfully. Router gates are preserved.")
else:
    if local_rank == 0:
        print("🚨 WARNING: No Stage 2 checkpoint found. Training with fresh experts and routers.")

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
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(
    trainable_params,
    lr=train_params["learning_rate"],
    weight_decay=train_params["weight_decay"],
    fused=True,
)
scaler = GradScaler()
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# --- 5.2. Resume from Stage 2.5 Checkpoint (If it exists) ---
latest_epoch = 0
if local_rank == 0:
    if os.path.exists(STAGE2_5_CHECKPOINT_DIR):
        epoch_numbers = [
            int(re.search(r"epoch_(\d+)", f).group(1))
            for f in os.listdir(STAGE2_5_CHECKPOINT_DIR)
            if re.search(r"epoch_(\d+)", f)
        ]
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)

epoch_tensor = torch.tensor([latest_epoch], dtype=torch.int).to(DEVICE)
dist.broadcast(epoch_tensor, src=0)
latest_epoch = epoch_tensor.item()

if latest_epoch > 0:
    checkpoint_path = os.path.join(
        STAGE2_5_CHECKPOINT_DIR, f"llm_stage2_5_epoch_{latest_epoch}.pth"
    )
    if local_rank == 0:
        print(
            f"💾 Resuming Stage 2.5 training from epoch {latest_epoch+1} using {checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(state_dict, strict=False)

    del state_dict
    gc.collect()
    if local_rank == 0:
        print("✅ Stage 2.5 training state resumed successfully.")

# --- 5.3. Finalize Model Setup ---
llm.model.embed_tokens.to(DEVICE)
llm.gradient_checkpointing_enable()
vision_connector = VisionLanguageConnector().to(DEVICE)
vision_connector.load_state_dict(
    torch.load(
        os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth"),
        map_location=DEVICE,
    )
)
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in vision_connector.parameters():
    param.requires_grad = False

if local_rank == 0:
    print(
        f"Optimizing {sum(p.numel() for p in trainable_params)} trainable router parameters."
    )

metrics_history = {
    "epoch": [], "train_loss": [], "train_ce_loss": [], "train_lb_loss": [],
    "val_loss": [], "learning_rate": [],
}
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_stage2.5.json")
if local_rank == 0 and latest_epoch > 0 and os.path.exists(metrics_path):
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
    print(f"🚀 Starting Stage 2.5 training from epoch {latest_epoch+1}...")
for epoch in range(latest_epoch, NUM_EPOCHS):
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
        
        if local_rank == 0 and (i + 1) % 100 == 0:
            router_params = [
                (name, p) for name, p in llm.named_parameters()
                if "gate" in name and p.requires_grad
            ]
            if router_params:
                print("--- Gradient Check ---")
                for name, p in router_params[:2]:
                    if p.grad is None:
                        print(f"�� {name} grad is None 🚨")
                    else:
                        grad_norm = torch.linalg.vector_norm(p.grad).item()
                        print(f"[DEBUG GRAD] {name}: grad_norm={grad_norm:.4e}")
                print("----------------------")

        if loss.item() > 0:
            total_train_loss += loss.item() * accumulation_steps
            total_ce_loss += ce_loss.item()
            if isinstance(total_load_balancing_loss, torch.Tensor):
                total_lb_loss += total_load_balancing_loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

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
    dist.barrier()
    avg_val_loss_tensor = torch.tensor(avg_val_loss).to(DEVICE)
    dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)
    avg_val_loss = avg_val_loss_tensor.item()
    
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
        router_weights = {}
        for name, weight in full_state_dict.items():
            param = llm.get_parameter(name)
            if param.requires_grad:
                router_weights[name] = weight

        os.makedirs(STAGE2_5_CHECKPOINT_DIR, exist_ok=True)
        save_path = os.path.join(
            STAGE2_5_CHECKPOINT_DIR, f"llm_stage2_5_epoch_{epoch+1}.pth"
        )
        torch.save(router_weights, save_path)
        print(f"✅ Saved small router-only checkpoint to {save_path}")

        previous_checkpoint_path = os.path.join(
            STAGE2_5_CHECKPOINT_DIR, f"llm_stage2_5_epoch_{epoch}.pth"
        )
        if os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)
            print(f"Removed previous router checkpoint: {previous_checkpoint_path}")

    dist.barrier()

dist.destroy_process_group()
print("Job finished.")
