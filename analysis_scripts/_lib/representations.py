"""Representation maths shared across analysis scripts."""

from __future__ import annotations

from collections import Counter

import numpy as np


def compute_cosine_similarity_matrix(representations: list[np.ndarray]) -> np.ndarray:
    """Pairwise cosine-similarity matrix for a list of 1-D representations.

    The diagonal is set to exactly 1.0; off-diagonal entries use the
    ``dot / (||a|| ||b|| + 1e-8)`` form (epsilon guards zero vectors).
    """
    n = len(representations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                cos_sim = np.dot(representations[i], representations[j]) / (
                    np.linalg.norm(representations[i]) * np.linalg.norm(representations[j]) + 1e-8
                )
                matrix[i, j] = float(cos_sim)
    return matrix


def majority_vote_expert(expert_choices, confidence_threshold: float = 0.6) -> tuple[str, float]:
    """Reduce per-token argmax expert ids to a single label by majority vote.

    Args:
        expert_choices: 1-D array/sequence of per-token chosen expert ids.
        confidence_threshold: Minimum winning-vote fraction for a decisive label.

    Returns:
        ``(label, fraction)`` where label is ``"Expert {k}"``, ``"mixed"`` or
        ``"unknown"`` (empty input), and fraction is the winning-vote share.
    """
    if len(expert_choices) == 0:
        return "unknown", 0.0

    votes = Counter(expert_choices)
    total_votes = len(expert_choices)
    if total_votes == 0 or len(votes) == 0:
        return "unknown", 0.0

    winner_expert = votes.most_common(1)[0][0]
    winner_fraction = votes[winner_expert] / total_votes

    if winner_fraction >= confidence_threshold:
        return f"Expert {winner_expert}", winner_fraction
    return "mixed", winner_fraction
