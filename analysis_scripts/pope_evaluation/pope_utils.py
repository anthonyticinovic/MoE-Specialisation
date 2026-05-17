#!/usr/bin/env python3
"""Shared helpers for the POPE evaluation pipeline.

Holds the single home for the yes/no answer extractors (standard + priming
variant) and the canonical POPE metric computation, removing the copies that
were previously duplicated across the 02/02b/03 scripts.
"""

import re


def extract_yes_no_answer(text: str, question: str = None) -> str:
    """
    Extract yes/no answer from model output.
    Handles both concise answers (Stage 2) and elaborate answers (Stage 3).

    Args:
        text: Generated text from model
        question: Optional - the original question, used to check if queried object is mentioned

    Returns:
        'yes', 'no', or 'unclear'
    """
    text_lower = text.lower().strip()

    # Direct matches (most common for Stage 2)
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    # Check first few words
    words = text_lower.split()
    if len(words) > 0:
        first_word = words[0].strip(".,!?")
        if first_word == "yes":
            return "yes"
        if first_word == "no":
            return "no"

    # Strong negative indicators (explicit negation)
    strong_negative_phrases = [
        "there is no",
        "there are no",
        "there isn't",
        "there aren't",
        "not visible",
        "cannot see",
        "no visible",
        "absence of",
        "no sign of",
        "does not show",
        "does not feature",
    ]
    for phrase in strong_negative_phrases:
        if phrase in text_lower[:80]:
            return "no"

    # Strong affirmative indicators (explicit affirmation)
    strong_affirmative_phrases = [
        "yes,",
        "yes there",
        "yes it",
        "there is a",
        "there are",
        "shows a",
        "features a",
        "depicts a",
        "contains a",
        "includes a",
        "has a",
        "with a",
        "shows the",
        "features the",
    ]
    for phrase in strong_affirmative_phrases:
        if phrase in text_lower[:80]:
            return "yes"

    # For Stage 3: Check if the queried object is actually mentioned in the response
    # This prevents treating generic captions as "yes" answers
    if question:
        # Extract object from question: "Is there a/an X in the image?"
        match = re.search(r"is there (?:a |an )?(\w+)", question.lower())
        if match:
            queried_object = match.group(1)
            object_mentioned = queried_object in text_lower[:80]

            if object_mentioned:
                # Object is mentioned - check if it's in a descriptive context (likely yes)
                # vs. a negative context
                # If definite article precedes object, it's describing it (yes)
                if f"the {queried_object}" in text_lower[:80]:
                    return "yes"
                # If possessive or descriptive phrase
                if any(
                    p in text_lower[:80]
                    for p in [
                        f"{queried_object} is",
                        f"{queried_object} in",
                        f"{queried_object} on",
                        f"{queried_object} at",
                    ]
                ):
                    return "yes"
            else:
                # Object NOT mentioned but we have descriptive text
                # Could be describing something else in the image (maybe no, maybe unclear)
                # If it's a generic truncated description, mark as unclear
                if len(text_lower) < 20 or text_lower.endswith(
                    (",", "and", "or", "with", "in", "a")
                ):
                    # Truncated or incomplete - unclear
                    return "unclear"
                # If it describes other objects, tentatively say no
                # But this is weak evidence
                return "unclear"

    # Fallback: If no strong indicators and no question context, check generic patterns
    # Descriptive patterns WITHOUT object mention are unclear (could be anything)
    descriptive_patterns = [
        "the image features",
        "the image shows",
        "the image depicts",
        "the scene features",
        "the scene shows",
    ]
    for pattern in descriptive_patterns:
        if pattern in text_lower[:50]:
            # Generic description without clear answer - unclear
            return "unclear"

    # Definite article at start suggests describing something, but we don't know what
    if text_lower.startswith("the ") and len(words) > 2:
        # Could be describing the queried object or something else
        return "unclear"

    return "unclear"


def extract_yes_no_answer_primed(text: str, question: str = None) -> str:
    """
    Extract yes/no answer from generated text using multiple strategies.

    Tuned for the Stage-3 priming generation path (see ``--use-priming``).

    Args:
        text: Generated text from model
        question: Original question (optional, for context-aware extraction)

    Returns:
        'yes', 'no', or 'unclear'
    """
    text_lower = text.lower().strip()

    # Strategy 1: Direct yes/no at start (most reliable)
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    # Strategy 2: Pattern matching for common formats
    if re.match(r"^yes[,.\s]", text_lower):
        return "yes"
    if re.match(r"^no[,.\s]", text_lower):
        return "no"

    # Strategy 3: Check first few words
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in ["yes", "yeah", "yep", "yup"]:
        return "yes"
    if first_word in ["no", "nope", "nah"]:
        return "no"

    # Strategy 4: For Stage 3 - check if it's trying to describe (means it failed to answer)
    descriptive_starts = [
        "the image",
        "there is",
        "there are",
        "this image",
        "in the image",
        "the photo",
        "this photo",
        "a ",
        "an ",
        "it shows",
        "it depicts",
    ]
    for desc_start in descriptive_starts:
        if text_lower.startswith(desc_start):
            # It's generating a description instead of yes/no
            # Try to infer from content if object is mentioned
            if question:
                # Extract object from question: "Is there a dog" -> "dog"
                match = re.search(r"is there (?:a |an )?(\w+)", question.lower())
                if match:
                    queried_object = match.group(1)
                    # Check if object is mentioned in first 80 chars of response
                    if queried_object in text_lower[:80]:
                        # Object mentioned in description = implicit yes
                        # But only if it's clearly referring to THE object
                        # e.g., "the dog is" = yes, but "a dog" might be hallucination
                        if f"the {queried_object}" in text_lower[:80]:
                            return "yes"
            return "unclear"

    # Strategy 5: Contains yes/no somewhere in first sentence
    first_sentence = text_lower.split(".")[0] if "." in text_lower else text_lower
    if "yes" in first_sentence and "no" not in first_sentence:
        return "yes"
    if "no" in first_sentence and "yes" not in first_sentence:
        return "no"

    # Strategy 6: Affirmative/negative words
    affirmative_words = ["correct", "indeed", "absolutely", "certainly"]
    negative_words = ["not", "none", "never", "incorrect"]

    words_in_first_sentence = first_sentence.split()[:10]
    has_affirmative = any(word in affirmative_words for word in words_in_first_sentence)
    has_negative = any(word in negative_words for word in words_in_first_sentence)

    if has_affirmative and not has_negative:
        return "yes"
    if has_negative and not has_affirmative:
        return "no"

    # Unable to determine
    return "unclear"


def compute_metrics(answers: list[dict]) -> dict:
    """
    Compute POPE evaluation metrics.

    Metrics:
    - Accuracy: Overall correctness
    - Precision: Of predicted yes, how many are truly yes?
    - Recall: Of true yes, how many are predicted yes?
    - F1: Harmonic mean of precision and recall
    - Yes ratio: Proportion of yes answers (measures over-generation/hallucination)

    Args:
        answers: List of dicts with 'answer' (ground truth) and 'predicted_answer'

    Returns:
        Dict with metrics
    """
    # Count outcomes
    true_positive = 0  # Predicted yes, actually yes
    false_positive = 0  # Predicted yes, actually no (hallucination!)
    true_negative = 0  # Predicted no, actually no
    false_negative = 0  # Predicted no, actually yes
    unclear = 0

    for item in answers:
        gt = item["answer"].lower()
        pred = item["predicted_answer"].lower()

        if pred == "unclear":
            unclear += 1
            continue

        if gt == "yes" and pred == "yes":
            true_positive += 1
        elif gt == "no" and pred == "yes":
            false_positive += 1
        elif gt == "no" and pred == "no":
            true_negative += 1
        elif gt == "yes" and pred == "no":
            false_negative += 1

    # Compute metrics
    total = true_positive + false_positive + true_negative + false_negative

    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "yes_ratio": 0.0,
            "num_samples": len(answers),
            "num_unclear": unclear,
        }

    accuracy = (true_positive + true_negative) / total

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    yes_ratio = (true_positive + false_positive) / total

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "num_samples": len(answers),
        "num_unclear": unclear,
    }
