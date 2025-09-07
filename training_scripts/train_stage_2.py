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

from transformers.models.mistral.modeling_mistral import MistralMLP

# Import your custom MoE classes
from models.custom_mistral import (
    MistralMoEConfig,
    MistralMoEForCausalLM,
    MistralMoEDecoderLayer,
)

from models.moe_layer import MoELayer

# --- 1. Register Your Custom Architecture ---
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

moe_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"

if local_rank == 0:
    print(f"Loading custom MoE model from {moe_model_path} onto CPU...")

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

# The embedding layer is `llm.model.embed_tokens`.
ignored_modules = [llm.model.embed_tokens]


# --- NEW: Define the auto-wrap policy for your custom decoder layer ---
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

# --- MODIFIED: More robust checkpoint loading ---
latest_epoch = 0
# Let only rank 0 scan the filesystem to find the latest checkpoint
if local_rank == 0:
    if os.path.exists(STAGE2_CHECKPOINT_DIR):
        epoch_numbers = []
        # Use regex to find all checkpoint files and extract their epoch number
        for filename in os.listdir(STAGE2_CHECKPOINT_DIR):
            match = re.match(r"llm_stage2_epoch_(\d+)\.pth", filename)
            if match:
                epoch_numbers.append(int(match.group(1)))
        
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)

# Broadcast the found latest_epoch from rank 0 to all other ranks
# This ensures all processes agree on which epoch to start from.
epoch_tensor = torch.tensor([latest_epoch], dtype=torch.int).to(DEVICE)
dist.broadcast(epoch_tensor, src=0)
latest_epoch = epoch_tensor.item()

# Handle case where training is already complete
if latest_epoch >= NUM_EPOCHS:
    if local_rank == 0:
        print(f"Latest checkpoint (epoch {latest_epoch}) is already >= NUM_EPOCHS ({NUM_EPOCHS}). Nothing to do.")
    dist.destroy_process_group()
    sys.exit(0) # Exit gracefully

# If a checkpoint is found, load it
if latest_epoch > 0:
    checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, f"llm_stage2_epoch_{latest_epoch}.pth")
    if local_rank == 0:
        print(f"💾 Found existing checkpoint. Resuming training from epoch {latest_epoch+1} using {checkpoint_path}")

    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        llm.load_state_dict(state_dict)
    
    del state_dict
    gc.collect()

    if local_rank == 0:
        print("✅ Resumed model weights loaded.")

# Manually move the ignored module to the correct GPU device.
if local_rank == 0:
    print(f"Manually moving ignored modules to device {DEVICE}")
llm.model.embed_tokens.to(DEVICE)

llm.gradient_checkpointing_enable()

vision_connector = VisionLanguageConnector().to(DEVICE)
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1.pth")
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

#if local_rank == 0:
#    print("Compiling the model with torch.compile()...")
#llm = torch.compile(llm)
#if local_rank == 0:
#    print("✅ Model compiled.")

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

# --- NEW: Add Gradient Accumulation Steps from config ---
accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
if local_rank == 0:
    print(f"Using gradient accumulation with {accumulation_steps} steps.")
    print(
        f"Effective batch size: {train_params['batch_size'] * accumulation_steps * dist.get_world_size()}"
    )

trainable_params = [p for p in llm.parameters() if p.requires_grad]

optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=True)
scaler = GradScaler()

# T_max is the total number of optimizer steps in the entire training run.
total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

if local_rank == 0:
    print(
        f"Optimizing {sum(p.numel() for p in llm.parameters() if p.requires_grad)} trainable parameters."
    )
# --- NEW: Initialize metrics history ---
metrics_history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "learning_rate":[]
}
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_stage2.json")

# If resuming, load previous metrics
if local_rank == 0 and latest_epoch > 0 and os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)


# ====================================================================================
# 8. TRAINING LOOP
# ====================================================================================
if local_rank == 0:
    print(f"🚀 Starting Stage 2 training for {NUM_EPOCHS} epochs...")
start_time = time.time()
for epoch in range(latest_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    llm.train()
    total_train_loss = 0
    optimizer.zero_grad()  # Zero gradients at the start of each epoch
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE),
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
        )

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)

            text_embeddings = llm.model.embed_tokens(input_ids)
            combined_embeddings = torch.cat(
                [visual_soft_tokens, text_embeddings], dim=1
            )

            combined_embeddings.requires_grad_(True)

            routing_mask = torch.cat(
                [
                    torch.zeros(
                        visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE
                    ),
                    torch.ones(
                        text_embeddings.shape[:2], dtype=torch.long, device=DEVICE
                    ),
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
                    text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1)
                )
                # --- MODIFIED: Scale loss for accumulation ---
                loss = loss / accumulation_steps
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

        scaler.scale(loss).backward()

        # --- MODIFIED: Rescale for logging before optimizer step ---
        if loss.item() > 0:
            total_train_loss += loss.item() * accumulation_steps

        # --- MODIFIED: Step optimizer only after accumulation_steps ---
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # --- Add periodic print statement for progress ---
        if local_rank == 0 and (i + 1) % 100 == 0:
            # This will print an update roughly every 100 batches
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
                images.to(DEVICE),
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
            )
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)

                text_embeddings = llm.model.embed_tokens(input_ids)
                combined_embeddings = torch.cat(
                    [visual_soft_tokens, text_embeddings], dim=1
                )

                routing_mask = torch.cat(
                    [
                        torch.zeros(
                            visual_soft_tokens.shape[:2],
                            dtype=torch.long,
                            device=DEVICE,
                        ),
                        torch.ones(
                            text_embeddings.shape[:2], dtype=torch.long, device=DEVICE
                        ),
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
    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # Record metrics for the epoch ---
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(current_lr)

        # Save metrics to a file at the end of each epoch
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"✅ Metrics saved to {metrics_path}")

    # Save checkpoint at the end of each epoch ---
    if local_rank == 0:
        print(f"Saving model checkpoint at the end of epoch {epoch+1}...")

    # Configure the policy for saving the full state dict
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = llm.state_dict()

    if local_rank == 0:

        os.makedirs(STAGE2_CHECKPOINT_DIR, exist_ok=True)
        file_path = os.path.join(STAGE2_CHECKPOINT_DIR, f"llm_stage2_epoch_{epoch+1}.pth")

        # Save the model state dict
        torch.save(cpu_state_dict, file_path)
        print(f"Model checkpoint saved to {file_path}")

        # Remove the previous checkpoint to save space ---
        # The checkpoint for the previous epoch is identified by the loop variable `epoch`.
        previous_checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, f"llm_stage2_epoch_{epoch}.pth")
        if os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)
            print(f"Removed previous checkpoint: {previous_checkpoint_path}")

    dist.barrier()

# --- 3. Calculate and print the total training time ---
if local_rank == 0:
    end_time = time.time()
    duration_seconds = end_time - start_time
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    print(f"--- Total Training Time: {hours}h {minutes}m {seconds}s ---")

dist.barrier()  # Wait for all processes to finish before exiting
dist.destroy_process_group()

print("Job finished.")
