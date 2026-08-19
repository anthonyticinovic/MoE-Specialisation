"""Balanced concept sampling from COCO captions.

Every concept-level analysis starts the same way: read the captions, find the
ones that mention exactly one concept from a list, and take N of each. That was
written three times — once in `cross_concept_similarity_matrix`, once in
`cross_modality_purity`, and once as an override in `layer_clustering_analysis`
that was byte-identical to the base it overrode.

The version kept here is the most capable of the three: it also matches compound
concepts such as ``red_apple``, which the other two silently failed to match at
all (a caption never contains an underscore, so those analyses would have
returned zero samples for such a concept without saying why).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _matches(caption_words: set[str], concept: str) -> bool:
    """Whether one concept appears in a caption.

    A concept with an underscore is compound — ``red_apple`` matches a caption
    containing both ``red`` and ``apple``, in any order. Anything else is a
    single word and must appear verbatim.
    """
    concept = concept.lower()
    if "_" in concept:
        return all(part in caption_words for part in concept.split("_"))
    return concept in caption_words


def extract_concept_samples(
    annotations_file: str,
    concepts: list[str],
    samples_per_concept: int,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Take up to ``samples_per_concept`` unambiguous captions for each concept.

    A caption mentioning two of the requested concepts is skipped rather than
    assigned to one of them: the analyses that consume this compare concepts
    against each other, so an ambiguous sample would appear on both sides.

    Args:
        annotations_file: Path to a COCO captions JSON.
        concepts: Concept keywords, e.g. ``["cat", "dog", "red_apple"]``.
        samples_per_concept: Target count per concept; fewer is warned about.
        seed: Unused by the selection itself, which is deterministic in file
            order — kept in the signature because callers pass it and because
            reintroducing randomness here must not silently change results.

    Returns:
        concept → list of ``{image_id, caption, image_path, concept}``, where
        ``image_path`` is the COCO ``file_name``, relative to the image dir.
    """
    logger.info("Extracting concept samples from COCO annotations...")
    logger.info("   Concepts: %s", concepts)
    logger.info("   Target: %d samples per concept", samples_per_concept)

    with open(annotations_file) as f:
        coco_data = json.load(f)

    image_id_to_path = {img["id"]: img["file_name"] for img in coco_data["images"]}
    concept_samples: dict[str, list[dict[str, Any]]] = {concept: [] for concept in concepts}

    for annotation in coco_data["annotations"]:
        words = set(annotation["caption"].lower().split())
        matching = [concept for concept in concepts if _matches(words, concept)]
        if len(matching) != 1:
            continue  # zero matches, or ambiguous between two concepts

        concept = matching[0]
        if len(concept_samples[concept]) < samples_per_concept:
            concept_samples[concept].append(
                {
                    "image_id": annotation["image_id"],
                    "caption": annotation["caption"],
                    "image_path": image_id_to_path[annotation["image_id"]],
                    "concept": concept,
                }
            )

    logger.info("   Extracted samples:")
    for concept, samples in concept_samples.items():
        logger.info("      %s: %d samples", concept, len(samples))

    short = {c: len(s) for c, s in concept_samples.items() if len(s) < samples_per_concept}
    if short:
        logger.warning(
            "Under-sampled concepts (target %d): %s. The analyses weight every "
            "concept equally, so an unbalanced set skews the comparison.",
            samples_per_concept,
            short,
        )
    return concept_samples
