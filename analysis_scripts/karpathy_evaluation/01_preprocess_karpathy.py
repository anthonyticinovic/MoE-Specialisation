#!/usr/bin/env python3
"""
Step 1: Preprocess Karpathy COCO split for evaluation.
Extracts test set and prepares data structures for both retrieval and captioning.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def parse_karpathy_split(dataset_json_path: str) -> Dict:
    """
    Parse the Karpathy dataset JSON and extract relevant information.
    
    Args:
        dataset_json_path: Path to dataset_coco.json
        
    Returns:
        Dictionary with parsed data
    """
    print(f"\n📖 Loading Karpathy split from: {dataset_json_path}")
    
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    print(f"   Total images in dataset: {len(images)}")
    
    # Split by dataset split
    split_counts = defaultdict(int)
    for img in images:
        split_counts[img['split']] += 1
    
    print(f"\n   Split distribution:")
    for split_name, count in sorted(split_counts.items()):
        print(f"      {split_name}: {count:,} images")
    
    return data


def extract_test_set_retrieval(data: Dict, output_path: str):
    """
    Extract test set for retrieval evaluation.
    
    Creates JSON with:
    - images: List of test image metadata
    - captions: List of all captions (5 per image = 25,000 total)
    """
    print(f"\n🔍 Extracting test set for retrieval evaluation...")
    
    test_images = [img for img in data['images'] if img['split'] == 'test']
    
    print(f"   Test images: {len(test_images)}")
    
    # Prepare retrieval data structure
    retrieval_data = {
        'images': [],
        'captions': []
    }
    
    caption_id = 0
    for img_idx, img in enumerate(test_images):
        # Add image info
        image_entry = {
            'image_id': img_idx,
            'coco_id': img['cocoid'],
            'filename': img['filename'],
            'filepath': f"{img['filepath']}/{img['filename']}",  # e.g., 'val2014/COCO_val2014_000000391895.jpg'
        }
        retrieval_data['images'].append(image_entry)
        
        # Add all 5 captions for this image
        for sent in img['sentences']:
            caption_entry = {
                'caption_id': caption_id,
                'image_id': img_idx,
                'text': sent['raw'].strip()
            }
            retrieval_data['captions'].append(caption_entry)
            caption_id += 1
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(retrieval_data, f, indent=2)
    
    print(f"   ✅ Saved: {output_path}")
    print(f"      Images: {len(retrieval_data['images'])}")
    print(f"      Captions: {len(retrieval_data['captions'])}")


def extract_test_set_captioning(data: Dict, images_output: str, references_output: str):
    """
    Extract test set for captioning evaluation.
    
    Creates two files:
    1. images JSON: List of images to caption
    2. references JSON: Ground-truth captions in COCO format
    """
    print(f"\n📝 Extracting test set for captioning evaluation...")
    
    test_images = [img for img in data['images'] if img['split'] == 'test']
    
    # Prepare image list (what to caption)
    images_data = []
    for img_idx, img in enumerate(test_images):
        entry = {
            'image_id': img_idx,
            'coco_id': img['cocoid'],
            'filename': img['filename'],
            'filepath': f"{img['filepath']}/{img['filename']}",  # e.g., 'val2014/COCO_val2014_000000391895.jpg'
        }
        images_data.append(entry)
    
    # Prepare references (ground truth) in COCO evaluation format
    references_data = {
        'images': [],
        'annotations': []
    }
    
    ann_id = 0
    for img_idx, img in enumerate(test_images):
        # Add image entry
        references_data['images'].append({'id': img_idx})
        
        # Add all 5 reference captions
        for sent in img['sentences']:
            ann_entry = {
                'image_id': img_idx,
                'id': ann_id,
                'caption': sent['raw'].strip()
            }
            references_data['annotations'].append(ann_entry)
            ann_id += 1
    
    # Save images list
    Path(images_output).parent.mkdir(parents=True, exist_ok=True)
    with open(images_output, 'w') as f:
        json.dump(images_data, f, indent=2)
    print(f"   ✅ Saved images: {images_output}")
    print(f"      Images to caption: {len(images_data)}")
    
    # Save references
    with open(references_output, 'w') as f:
        json.dump(references_data, f, indent=2)
    print(f"   ✅ Saved references: {references_output}")
    print(f"      Reference captions: {len(references_data['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Karpathy COCO split')
    parser.add_argument(
        '--karpathy_json',
        type=str,
        default='YOUR_PATH_HERE/datasets/coco/karpathy_split/dataset_coco.json',
        help='Path to dataset_coco.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/karpathy_evaluation',
        help='Output directory for preprocessed files'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("KARPATHY COCO PREPROCESSING")
    print("="*80)
    
    # Parse Karpathy split
    data = parse_karpathy_split(args.karpathy_json)
    
    # Extract test set for retrieval
    retrieval_output = f"{args.output_dir}/karpathy_test_retrieval.json"
    extract_test_set_retrieval(data, retrieval_output)
    
    # Extract test set for captioning
    images_output = f"{args.output_dir}/karpathy_test_images.json"
    references_output = f"{args.output_dir}/karpathy_test_references.json"
    extract_test_set_captioning(data, images_output, references_output)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  1. {retrieval_output}")
    print(f"  2. {images_output}")
    print(f"  3. {references_output}")
    print()


if __name__ == "__main__":
    main()
