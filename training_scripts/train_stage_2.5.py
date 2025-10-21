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
    cpu_offload=CPUOffload(offload_params=None),
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
        # Increased std to create stronger initial preferences, helping routers explore specialization
        nn.init.normal_(new_gate.weight, std=0.1)  # Changed from 0.02 to 0.1
        new_gate = new_gate.to(DEVICE)
        layer.mlp.gate = new_gate
        layer.mlp.gate.weight.requires_grad = True

# ====================================================================================
# SOLUTION 4: VERIFY EXPERT WEIGHTS ARE FROZEN
# ====================================================================================
if local_rank == 0:
    print("\n=== Expert Weight Verification ===")
    sample_layer = llm.module.model.layers[0]
    expert_0_requires_grad = any(p.requires_grad for p in sample_layer.mlp.experts[0].parameters())
    expert_1_requires_grad = any(p.requires_grad for p in sample_layer.mlp.experts[1].parameters())
    gate_requires_grad = sample_layer.mlp.gate.weight.requires_grad
    
    print(f"Expert 0 trainable: {expert_0_requires_grad} (should be False)")
    print(f"Expert 1 trainable: {expert_1_requires_grad} (should be False)")
    print(f"Gate trainable: {gate_requires_grad} (should be True)")
    
    if expert_0_requires_grad or expert_1_requires_grad:
        raise RuntimeError("❌ Experts are not frozen! This will corrupt training.")
    if not gate_requires_grad:
        raise RuntimeError("❌ Router gate is frozen! Cannot train routers.")
    
    print("✅ All weight freeze states correct")
    print("==================================\n")

# ====================================================================================
# 6. DATA & OPTIMIZER
# ====================================================================================
if local_rank == 0:
    print("Creating datasets and dataloaders...")
train_dataset = COCO_Loader(
    image_dir=paths["image_dir"], annotations_file=paths["annotations_file"],
    clip_processor=clip_processor, tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"], split="train",
    seed=loader_params.get("data_seed", 42),  # Fixed seed for reproducibility
)
val_dataset = COCO_Loader(
    image_dir=paths["image_dir"], annotations_file=paths["annotations_file"],
    clip_processor=clip_processor, tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"], split="val",
    seed=loader_params.get("data_seed", 42),  # Same seed ensures consistent splits
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
    torch.load(os.path.join(OUTPUT_DIR, "vision_connector_stage1_best.pth"), map_location=DEVICE,)
)
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in vision_connector.parameters():
    param.requires_grad = False

if local_rank == 0:
    print(f"Optimizing {sum(p.numel() for p in trainable_params)} trainable router parameters.")

metrics_history = {
    "epoch": [], "train_loss": [], "train_ce_loss": [], "train_lb_loss": [],
    "val_loss": [], "learning_rate": [], "entropy": [], "temperature": [],
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

    # Temperature annealing: Start high (exploration) -> decay to 1.0 (exploitation)
    # High temperature = softer probabilities = more exploration
    temperature = max(1.0, 2.0 * (0.9 ** epoch))  # Starts at 2.0, decays to 1.0
    
    # Set temperature for all MoE layers
    for layer in llm.module.model.layers:
        if hasattr(layer.mlp, "temperature"):
            layer.mlp.temperature = temperature
    
    if local_rank == 0 and epoch == start_epoch:
        print(f"  Router temperature for epoch {epoch+1}: {temperature:.3f}")

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

            # Pass temperature to MoE layers through forward hooks
            # Note: LLM doesn't directly accept temperature, so we set it as layer attribute
            for layer in llm.module.model.layers:
                if hasattr(layer.mlp, "routing_mode") and layer.mlp.routing_mode == 'soft':
                    # Store temperature for this forward pass
                    layer.mlp._forward_temperature = temperature
            
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
            total_entropy_bonus = 0
            for layer in llm.module.model.layers:
                if hasattr(layer.mlp, "load_balancing_loss"):
                    total_load_balancing_loss += layer.mlp.load_balancing_loss
                
                # Add entropy bonus to encourage exploration
                if hasattr(layer.mlp, "gate"):
                    # Get routing logits for current batch
                    gate_logits = layer.mlp.gate(combined_embeddings.view(-1, 4096))
                    gate_probs = torch.softmax(gate_logits, dim=-1)
                    # Entropy = -sum(p * log(p)). Higher entropy = more exploration
                    entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=-1).mean()
                    total_entropy_bonus += entropy

            # Entropy coefficient: negative because we want to MAXIMIZE entropy (minimize negative entropy)
            # Start with small coefficient and decay over time to allow specialization later
            entropy_coeff = 0.001 * (0.95 ** epoch)  # Decays from 0.001 to ~0.0003 over 20 epochs
            
            loss = (
                ce_loss 
                + LOAD_BALANCING_COEFF * total_load_balancing_loss
                - entropy_coeff * total_entropy_bonus  # Negative = maximize entropy
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
                
                # Now clip the gradients - use MUCH higher threshold
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    router_params, 
                    max_norm=10.0  # Increased from 1.0 to 10.0
                )
                
                    # Debug logging every 100 batches
                if local_rank == 0 and (i + 1) % 100 == 0:
                    print(f"--- Gradient Check ---")
                    print(f"  Router grad norm BEFORE clip: {total_norm:.2f}")
                    print(f"  Router grad norm AFTER clip: {clipped_norm:.2f}")
                    max_norm = 10.0  # Updated to match new clipping threshold
                    was_clipped = total_norm > max_norm
                    actual_norm_after_clip = min(total_norm, max_norm)
                    print(f"  Clipping applied: {'YES' if was_clipped else 'NO'}")
                    print(f"  Actual norm after clip: {actual_norm_after_clip:.2f}")                    # Sample a few layers to check individual norms
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
            if isinstance(total_entropy_bonus, torch.Tensor):
                # Track average entropy (will be added to metrics later)
                if not hasattr(torch.cuda, '_total_entropy'):
                    torch.cuda._total_entropy = 0
                    torch.cuda._entropy_count = 0
                torch.cuda._total_entropy += total_entropy_bonus.item()
                torch.cuda._entropy_count += 1

        # ====================================================================================
        # SOLUTION 3: ROUTER MONITORING (Enhanced with modality-specific routing)
        # ====================================================================================
        if local_rank == 0 and (i + 1) % 500 == 0:
            # Monitor routing decisions for first layer
            with torch.no_grad():
                num_visual = visual_soft_tokens.shape[1]
                
                # Sample visual token (middle of visual sequence)
                visual_idx = num_visual // 2
                visual_hidden = combined_embeddings[0:1, visual_idx:visual_idx+1, :]
                
                # Sample text token (middle of text sequence)
                text_idx = num_visual + (input_ids.shape[1] // 2)
                text_hidden = combined_embeddings[0:1, text_idx:text_idx+1, :]
                
                print(f"\n{'='*60}")
                print(f"Router Analysis - Batch {i+1} | Epoch {epoch+1}")
                print(f"{'='*60}")
                
                for layer_idx in [0, 15, 31]:  # First, middle, last
                    check_layer = llm.module.model.layers[layer_idx]
                    if hasattr(check_layer.mlp, 'gate'):
                        # Convert to float32 to match gate weights dtype
                        visual_hidden_fp32 = visual_hidden.float()
                        text_hidden_fp32 = text_hidden.float()
                        
                        # Visual token routing
                        visual_logits = check_layer.mlp.gate(visual_hidden_fp32.view(-1, 4096))
                        visual_probs = torch.softmax(visual_logits, dim=-1)
                        
                        # Text token routing
                        text_logits = check_layer.mlp.gate(text_hidden_fp32.view(-1, 4096))
                        text_probs = torch.softmax(text_logits, dim=-1)
                        
                        print(f"\nLayer {layer_idx:2d}:")
                        print(f"  Visual Token -> E0: {visual_probs[0,0]:.3f}, E1: {visual_probs[0,1]:.3f} ({'✓E0' if visual_probs[0,0] > 0.6 else '✓E1' if visual_probs[0,1] > 0.6 else 'MIXED'})")
                        print(f"  Text   Token -> E0: {text_probs[0,0]:.3f}, E1: {text_probs[0,1]:.3f} ({'✓E0' if text_probs[0,0] > 0.6 else '✓E1' if text_probs[0,1] > 0.6 else 'MIXED'})")
                        
                        # Calculate specialization score (how different are the routings?)
                        specialization = abs(visual_probs[0,0] - text_probs[0,0]).item()
                        print(f"  Specialization score: {specialization:.3f} ({'GOOD' if specialization > 0.3 else 'WEAK'})")
                
                print(f"{'='*60}\n")

        if local_rank == 0 and (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}]")

    avg_train_loss = total_train_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_lb_loss = total_lb_loss / len(train_loader)
    
    # Calculate average entropy
    avg_entropy = 0
    if hasattr(torch.cuda, '_total_entropy') and torch.cuda._entropy_count > 0:
        avg_entropy = torch.cuda._total_entropy / torch.cuda._entropy_count
        # Reset for next epoch
        torch.cuda._total_entropy = 0
        torch.cuda._entropy_count = 0

    if local_rank == 0:
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f} | CE Loss: {avg_ce_loss:.4f} | "
            f"LB Loss: {avg_lb_loss:.4f} | Entropy: {avg_entropy:.4f} | Temp: {temperature:.3f}"
        )

    # --- Validation Phase ---
    llm.eval()
    total_val_loss = 0
    val_steps = 0
    MAX_VAL_BATCHES = 75  # DRASTICALLY reduced to prevent timeout
    
    if local_rank == 0:
        print(f"  Starting validation (max {MAX_VAL_BATCHES} batches per GPU)...")
    
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            # Early stop after MAX_VAL_BATCHES
            if i >= MAX_VAL_BATCHES:
                break
                
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE),
            )
            
            try:
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
                val_steps += 1
            except Exception as e:
                if local_rank == 0:
                    print(f"⚠️  Validation batch {i} failed: {e}")
                break

    # Each rank computed its own average - just use local average (no distributed aggregation)
    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')

    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f} (rank 0, {val_steps} batches)")

    # --- Metrics and Checkpoint Saving ---
    if local_rank == 0:
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["train_ce_loss"].append(avg_ce_loss)
        metrics_history["train_lb_loss"].append(avg_lb_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        metrics_history["entropy"].append(avg_entropy)
        metrics_history["temperature"].append(temperature)
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

    # Removed barrier - not needed since only rank 0 saves checkpoints

dist.destroy_process_group()
print("Job finished.")
