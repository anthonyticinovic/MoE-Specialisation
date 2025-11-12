#!/usr/bin/env python3
"""
Fix caption IDs: Convert from COCO IDs to Karpathy sequential IDs.
"""

import json
from pathlib import Path
import os

# Change to script directory
os.chdir(Path(__file__).parent)

# Load the ID mapping
print("Loading ID mapping...")
with open('results/karpathy_evaluation/karpathy_test_images.json', 'r') as f:
    mapping_data = json.load(f)

# Create coco_id -> karpathy_id mapping
coco_to_karpathy = {item['coco_id']: item['image_id'] for item in mapping_data}
print(f"Loaded {len(coco_to_karpathy)} image ID mappings")

# Fix Stage 2 captions
print("\nFixing Stage 2 captions...")
stage2_path = 'results/karpathy_evaluation/captioning/stage2_captions.json'
with open(stage2_path, 'r') as f:
    stage2_captions = json.load(f)

stage2_fixed = []
missing_ids = []
for item in stage2_captions:
    coco_id = item['image_id']
    if coco_id in coco_to_karpathy:
        stage2_fixed.append({
            'image_id': coco_to_karpathy[coco_id],
            'caption': item['caption']
        })
    else:
        missing_ids.append(coco_id)

print(f"  Original: {len(stage2_captions)} captions")
print(f"  Fixed: {len(stage2_fixed)} captions")
if missing_ids:
    print(f"  ⚠️  Missing mappings for {len(missing_ids)} IDs: {missing_ids[:5]}...")

with open(stage2_path, 'w') as f:
    json.dump(stage2_fixed, f, indent=2)
print(f"  ✅ Saved fixed captions to {stage2_path}")

# Fix Stage 3 captions
print("\nFixing Stage 3 captions...")
stage3_path = 'results/karpathy_evaluation/captioning/stage3_captions.json'
with open(stage3_path, 'r') as f:
    stage3_captions = json.load(f)

stage3_fixed = []
missing_ids = []
for item in stage3_captions:
    coco_id = item['image_id']
    if coco_id in coco_to_karpathy:
        stage3_fixed.append({
            'image_id': coco_to_karpathy[coco_id],
            'caption': item['caption']
        })
    else:
        missing_ids.append(coco_id)

print(f"  Original: {len(stage3_captions)} captions")
print(f"  Fixed: {len(stage3_fixed)} captions")
if missing_ids:
    print(f"  ⚠️  Missing mappings for {len(missing_ids)} IDs: {missing_ids[:5]}...")

with open(stage3_path, 'w') as f:
    json.dump(stage3_fixed, f, indent=2)
print(f"  ✅ Saved fixed captions to {stage3_path}")

print("\n✅ Done! Caption IDs have been converted from COCO IDs to Karpathy IDs.")
