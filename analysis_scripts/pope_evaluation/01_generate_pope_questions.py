#!/usr/bin/env python3
"""
Step 1: Generate POPE (Polling-based Object Probing Evaluation) questions.
Creates yes/no questions from COCO val2017 annotations.

POPE evaluates object hallucination with 3 difficulty levels:
- Random: Objects randomly sampled from entire dataset
- Popular: Most frequent objects in COCO
- Adversarial: Objects that co-occur with image objects
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_coco_annotations(annotations_file: str) -> dict:
    """Load COCO annotations."""
    print(f"\n📂 Loading COCO annotations from: {annotations_file}")
    with open(annotations_file) as f:
        coco_data = json.load(f)
    print(f"   Found {len(coco_data['images'])} images")
    print(f"   Found {len(coco_data['annotations'])} object annotations")
    print(f"   Found {len(coco_data['categories'])} object categories")
    return coco_data


def build_category_mapping(coco_data: dict) -> dict[int, str]:
    """Build mapping from category ID to category name."""
    return {cat["id"]: cat["name"] for cat in coco_data["categories"]}


def build_image_to_objects(coco_data: dict, category_mapping: dict) -> dict[int, set[str]]:
    """
    Build mapping from image_id to set of object names present in the image.

    Returns:
        Dict[image_id, Set[object_name]]
    """
    image_objects = {}

    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        object_name = category_mapping[category_id]

        if image_id not in image_objects:
            image_objects[image_id] = set()

        image_objects[image_id].add(object_name)

    return image_objects


def get_object_frequency(coco_data: dict, category_mapping: dict) -> list[tuple]:
    """
    Get object frequency counts from entire COCO dataset.

    Returns:
        List of (object_name, count) tuples sorted by frequency (descending)
    """
    object_counter = Counter()

    for ann in coco_data["annotations"]:
        category_id = ann["category_id"]
        object_name = category_mapping[category_id]
        object_counter[object_name] += 1

    return object_counter.most_common()


def build_cooccurrence_matrix(
    image_objects: dict[int, set[str]], all_objects: set[str]
) -> dict[str, list[str]]:
    """
    Build co-occurrence matrix: for each object, list objects that co-occur with it.

    Returns:
        Dict[object_name, List[co-occurring objects sorted by frequency]]
    """
    cooccurrence = {obj: Counter() for obj in all_objects}

    for image_id, objects_in_image in image_objects.items():
        objects_list = list(objects_in_image)
        # For each pair of objects in the same image
        for i, obj1 in enumerate(objects_list):
            for obj2 in objects_list[i + 1 :]:
                cooccurrence[obj1][obj2] += 1
                cooccurrence[obj2][obj1] += 1

    # Convert to sorted lists
    cooccurrence_sorted = {}
    for obj, counter in cooccurrence.items():
        cooccurrence_sorted[obj] = [obj_name for obj_name, _ in counter.most_common()]

    return cooccurrence_sorted


def generate_pope_questions_for_image(
    image_id: int,
    objects_in_image: set[str],
    all_objects: list[str],
    popular_objects: list[str],
    cooccurrence: dict[str, list[str]],
    num_questions: int = 3,
    seed: int = None,
) -> dict[str, list[dict]]:
    """
    Generate POPE questions for a single image with 3 difficulty levels.

    Args:
        image_id: COCO image ID
        objects_in_image: Set of objects present in the image
        all_objects: List of all possible objects
        popular_objects: List of popular objects (sorted by frequency)
        cooccurrence: Co-occurrence matrix
        num_questions: Number of yes/no question pairs per difficulty level
        seed: Random seed for reproducibility

    Returns:
        Dict with 'random', 'popular', 'adversarial' keys, each containing list of questions
    """
    if seed is not None:
        random.seed(seed + image_id)  # Different seed per image but deterministic

    questions = {"random": [], "popular": [], "adversarial": []}

    objects_list = list(objects_in_image)

    # Need at least 1 object in image to generate negative examples
    if len(objects_list) == 0:
        return questions

    # ========== RANDOM DIFFICULTY ==========
    # Positive: Randomly sample from objects IN image
    # Negative: Randomly sample from objects NOT in image

    # Positive samples
    if len(objects_list) >= num_questions:
        positive_objects = random.sample(objects_list, num_questions)
    else:
        positive_objects = random.choices(objects_list, k=num_questions)

    for obj in positive_objects:
        questions["random"].append(
            {
                "image_id": image_id,
                "question": f"Is there a {obj} in the image?",
                "answer": "yes",
                "object": obj,
            }
        )

    # Negative samples (not in image)
    objects_not_in_image = [obj for obj in all_objects if obj not in objects_in_image]
    if len(objects_not_in_image) >= num_questions:
        negative_objects = random.sample(objects_not_in_image, num_questions)
    else:
        negative_objects = random.choices(objects_not_in_image, k=num_questions)

    for obj in negative_objects:
        questions["random"].append(
            {
                "image_id": image_id,
                "question": f"Is there a {obj} in the image?",
                "answer": "no",
                "object": obj,
            }
        )

    # ========== POPULAR DIFFICULTY ==========
    # Negative: Sample popular objects that are NOT in image (harder negatives)

    # Positive: Same as random
    for obj in positive_objects:
        questions["popular"].append(
            {
                "image_id": image_id,
                "question": f"Is there a {obj} in the image?",
                "answer": "yes",
                "object": obj,
            }
        )

    # Negative: Popular objects not in image
    popular_not_in_image = [obj for obj in popular_objects if obj not in objects_in_image]
    if len(popular_not_in_image) >= num_questions:
        negative_popular = popular_not_in_image[:num_questions]  # Take most popular
    else:
        negative_popular = random.choices(popular_not_in_image, k=num_questions)

    for obj in negative_popular:
        questions["popular"].append(
            {
                "image_id": image_id,
                "question": f"Is there a {obj} in the image?",
                "answer": "no",
                "object": obj,
            }
        )

    # ========== ADVERSARIAL DIFFICULTY ==========
    # Negative: Objects that co-occur with objects in the image (hardest negatives)

    # Positive: Same as random
    for obj in positive_objects:
        questions["adversarial"].append(
            {
                "image_id": image_id,
                "question": f"Is there a {obj} in the image?",
                "answer": "yes",
                "object": obj,
            }
        )

    # Negative: Co-occurring objects not in image
    cooccurring_candidates = set()
    for obj in objects_list:
        if obj in cooccurrence:
            cooccurring_candidates.update(cooccurrence[obj])

    # Remove objects that ARE in the image
    cooccurring_not_in_image = list(cooccurring_candidates - objects_in_image)

    if len(cooccurring_not_in_image) >= num_questions:
        negative_adversarial = random.sample(cooccurring_not_in_image, num_questions)
    elif len(cooccurring_not_in_image) > 0:
        negative_adversarial = random.choices(cooccurring_not_in_image, k=num_questions)
    else:
        # Fallback to random negatives if no co-occurring objects
        negative_adversarial = random.sample(
            objects_not_in_image, min(num_questions, len(objects_not_in_image))
        )

    for obj in negative_adversarial:
        questions["adversarial"].append(
            {
                "image_id": image_id,
                "question": f"Is there a {obj} in the image?",
                "answer": "no",
                "object": obj,
            }
        )

    return questions


def generate_pope_dataset(
    coco_data: dict, num_images: int = 500, questions_per_image: int = 3, seed: int = 42
) -> dict[str, list[dict]]:
    """
    Generate complete POPE dataset for evaluation.

    Args:
        coco_data: COCO annotations
        num_images: Number of images to sample
        questions_per_image: Number of positive/negative pairs per difficulty
        seed: Random seed

    Returns:
        Dict with 'random', 'popular', 'adversarial' keys
    """
    random.seed(seed)

    print("\n🔮 Generating POPE questions...")
    print(f"   Num images: {num_images}")
    print(
        f"   Questions per image: {questions_per_image} positive + {questions_per_image} negative = {questions_per_image * 2} per difficulty"
    )
    print(f"   Total questions: {num_images * questions_per_image * 2 * 3} across 3 difficulties")

    # Build mappings
    category_mapping = build_category_mapping(coco_data)
    image_objects = build_image_to_objects(coco_data, category_mapping)

    # Get object frequency
    object_frequency = get_object_frequency(coco_data, category_mapping)
    popular_objects = [obj for obj, _ in object_frequency]  # Sorted by frequency
    all_objects = list(category_mapping.values())

    print(f"   Built mappings: {len(all_objects)} unique objects")
    print(f"   Top 5 popular objects: {popular_objects[:5]}")

    # Build co-occurrence matrix
    print("   Building co-occurrence matrix...")
    cooccurrence = build_cooccurrence_matrix(image_objects, set(all_objects))

    # Sample images
    valid_image_ids = [img_id for img_id in image_objects.keys() if len(image_objects[img_id]) > 0]
    sampled_image_ids = random.sample(valid_image_ids, min(num_images, len(valid_image_ids)))

    print(f"   Sampled {len(sampled_image_ids)} images with objects")

    # Generate questions for each image
    all_questions = {"random": [], "popular": [], "adversarial": []}

    for image_id in sampled_image_ids:
        objects_in_image = image_objects[image_id]

        image_questions = generate_pope_questions_for_image(
            image_id=image_id,
            objects_in_image=objects_in_image,
            all_objects=all_objects,
            popular_objects=popular_objects,
            cooccurrence=cooccurrence,
            num_questions=questions_per_image,
            seed=seed,
        )

        for difficulty in ["random", "popular", "adversarial"]:
            all_questions[difficulty].extend(image_questions[difficulty])

    # Print statistics
    print("\n📊 Generated questions:")
    for difficulty in ["random", "popular", "adversarial"]:
        num_yes = sum(1 for q in all_questions[difficulty] if q["answer"] == "yes")
        num_no = sum(1 for q in all_questions[difficulty] if q["answer"] == "no")
        print(
            f"   {difficulty.capitalize():12s}: {len(all_questions[difficulty])} questions ({num_yes} yes, {num_no} no)"
        )

    return all_questions


def main():
    parser = argparse.ArgumentParser(description="Generate POPE evaluation questions")
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help="COCO instances annotations (default: <parent of paths.image_dir>/"
        "annotations/instances_val2017.json from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pope_evaluation",
        help="Output directory for POPE questions",
    )
    parser.add_argument(
        "--num_images", type=int, default=500, help="Number of images to sample for evaluation"
    )
    parser.add_argument(
        "--questions_per_image",
        type=int,
        default=3,
        help="Number of positive/negative question pairs per image per difficulty",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.annotations_file is None:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from analysis_scripts._lib import get_paths

        coco_root = Path(get_paths()["image_dir"]).parent
        args.annotations_file = str(coco_root / "annotations" / "instances_val2017.json")

    print("=" * 80)
    print("POPE QUESTION GENERATION".center(80))
    print("=" * 80)

    # Load COCO annotations
    coco_data = load_coco_annotations(args.annotations_file)

    # Generate POPE questions
    pope_questions = generate_pope_dataset(
        coco_data=coco_data,
        num_images=args.num_images,
        questions_per_image=args.questions_per_image,
        seed=args.seed,
    )

    # Save questions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for difficulty in ["random", "popular", "adversarial"]:
        output_file = output_dir / f"pope_{difficulty}.json"
        with open(output_file, "w") as f:
            json.dump(pope_questions[difficulty], f, indent=2)
        print(f"\n💾 Saved {len(pope_questions[difficulty])} questions to: {output_file}")

    print("\n" + "=" * 80)
    print("✅ POPE QUESTION GENERATION COMPLETE".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
