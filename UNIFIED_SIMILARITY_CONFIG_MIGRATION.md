# Unified Similarity Matrix Configuration Migration

## Summary
Updated the `cross_concept_similarity_matrix.py` analysis script to use a unified configuration architecture that supports both Stage 2 and Stage 3 experiments through CLI arguments.

## Changes Made

### 1. Unified Config File
**File:** `configs/similarity_matrix.json` (renamed from `similarity_matrix_stage3.json`)

**Removed fields** (now CLI-only):
- `mode` - specify via `--mode stage2` or `--mode stage3`
- `stage2_checkpoint` - specify via `--stage2-checkpoint <path>`
- `stage3_checkpoint` - specify via `--stage3-checkpoint <path>`
- `temperature` - specify via `--temperature <value>`
- `output_dir` - specify via `--output-dir <path>`

**Retained fields** (shared across both modes):
- `concepts`: List of concept keywords for COCO sampling
- `samples_per_concept`: Number of samples to extract per concept
- `annotations_file`: Path to COCO annotations
- `image_dir`: Path to COCO images
- `layers`: Layer indices to analyze (default: [0, 16, 31])
- `pooling`: Pooling strategy (default: "mean")
- `seed`: Random seed for reproducibility (default: 42)

### 2. Python Script Updates
**File:** `analysis_scripts/cross_concept_similarity_matrix.py`

#### New CLI Arguments:
```bash
--mode {stage2,stage3}              # Analysis mode (required via CLI)
--stage2-checkpoint <path>          # Optional, defaults to llm_stage2_best.pth
--stage3-checkpoint <path>          # Required for stage3 mode
--temperature <float>               # Stage 3 routing temperature (default: 0.01)
--output-dir <path>                 # Output directory (mode-specific)
```

#### New Methods:
- `_load_stage2_models()`: Handles Stage 2 checkpoint loading with optional custom path
- Updated `__init__()`: Accepts `stage2_checkpoint` parameter
- Updated config validation: Mode no longer required in config file

#### Key Behavior:
- **Stage 2 Mode**: Loads Stage 1 vision connector + Stage 2 LLM checkpoint, uses hard routing
- **Stage 3 Mode**: Loads Stage 3 checkpoint (overrides Stage 2 weights), uses soft routing
- Both modes share same COCO concept sampling and layer analysis logic

### 3. Stage 2 SBATCH Script
**File:** `hpc/analysis_scripts/cross_concept_stage2.sbatch`

**Updates:**
- Uses unified config: `configs/similarity_matrix.json`
- Explicitly passes `--mode stage2` via CLI
- Specifies Stage 2 checkpoint path: `--stage2-checkpoint /data/gpfs/projects/COMP90055/aticinovic/outputs/stage2_checkpoints/llm_stage2_best.pth`
- Sets output directory: `--output-dir results/stage2_alignment/similarity_matrix/`
- Updated job name and output paths to reflect Stage 2

### 4. Stage 3 SBATCH Script
**File:** `hpc/analysis_scripts/cross_concept_stage3.sbatch`

**Updates:**
- Uses unified config: `configs/similarity_matrix.json` (previously `similarity_matrix_stage3.json`)
- Retains existing CLI arguments: `--mode stage3`, `--stage3-checkpoint`, `--temperature 0.01`
- Maintains output directory: `results/stage3_alignment/similarity_matrix/`

## Usage Examples

### Running Stage 2 Analysis:
```bash
sbatch hpc/analysis_scripts/cross_concept_stage2.sbatch
```

Or manually:
```bash
python analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/similarity_matrix.json \
    --mode stage2 \
    --stage2-checkpoint /path/to/llm_stage2_best.pth \
    --output-dir results/stage2_alignment/similarity_matrix/ \
    --training-config configs/training_config.yaml \
    --device cuda
```

### Running Stage 3 Analysis:
```bash
sbatch hpc/analysis_scripts/cross_concept_stage3.sbatch
```

Or manually:
```bash
python analysis_scripts/cross_concept_similarity_matrix.py \
    --config-file configs/similarity_matrix.json \
    --mode stage3 \
    --stage3-checkpoint /path/to/llm_stage3_best.pth \
    --temperature 0.01 \
    --output-dir results/stage3_alignment/similarity_matrix/ \
    --training-config configs/training_config.yaml \
    --device cuda
```

## Architecture Benefits

### ✅ Unified Configuration
- Single config file for shared parameters (concepts, layers, COCO paths)
- Reduces duplication and maintenance burden
- Consistent concept selection across both modes

### ✅ CLI-Driven Mode Selection
- Mode-specific parameters passed via command line
- Clear separation between shared and mode-specific configuration
- Easy to compare Stage 2 vs Stage 3 results with identical settings

### ✅ Flexible Checkpoint Loading
- Optional checkpoint path arguments for both modes
- Defaults to standard training output paths
- Easy to test different checkpoint versions

### ✅ Separate Output Directories
- Stage 2: `results/stage2_alignment/similarity_matrix/`
- Stage 3: `results/stage3_alignment/similarity_matrix/`
- Prevents result overwriting, enables easy comparison

## Technical Details

### Stage 2 Loading Process:
1. Load base MoE architecture
2. Load Stage 1 vision connector (`vision_connector_stage1_best.pth`)
3. Load Stage 2 expert weights (`llm_stage2_best.pth`)
4. Enable hard routing mode (vision→expert0, text→expert1)

### Stage 3 Loading Process:
1. Load base MoE architecture + Stage 2 weights (for initialization)
2. Load Stage 1 vision connector
3. Override with Stage 3 checkpoint (learned routers + optionally updated connector)
4. Enable soft routing mode with temperature=0.01 (near-deterministic)

### Common Analysis:
- Extract COCO concept samples (8 concepts × 20 samples each)
- Analyze layers 0, 16, 31
- Compute 2N×2N similarity matrices (image and text representations)
- Generate heatmap visualizations

## Verification

All changes verified:
- ✅ No syntax errors in Python script
- ✅ Config file properly formatted (JSON valid)
- ✅ Both SBATCH scripts use correct config path
- ✅ Mode-specific parameters properly handled via CLI
- ✅ Output directories correctly separated

## Migration Notes

**No breaking changes for Stage 3**: Existing Stage 3 workflow unchanged except for config filename.

**New capability for Stage 2**: Can now run the same COCO-based analysis that Stage 3 uses, enabling direct comparison of representation alignment between forced routing (Stage 2) and learned routing (Stage 3).

## Next Steps

1. Test Stage 2 analysis with unified config
2. Compare Stage 2 vs Stage 3 similarity matrices
3. Analyze differences in cross-modal alignment between hard and soft routing
4. Consider extending to Stage 2.5 if needed
