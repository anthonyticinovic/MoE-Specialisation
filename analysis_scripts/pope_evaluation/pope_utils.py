#!/usr/bin/env python3
"""Shared helpers for the POPE evaluation pipeline.

Holds the single home for the yes/no answer extractors (standard + priming
variant) and the canonical POPE metric computation, removing the copies that
were previously duplicated across the 02/02b/03 scripts.
"""

import re
from dataclasses import dataclass

# The phrase ladders below are scanned in the order they appear in
# `extract_yes_no_answer`. That order is load-bearing and pinned by
# `tests/test_analysis_lib.py::TestExtractYesNoAnswer` — see in particular the
# test recording that "the image features a ..." is scored "yes" by
# STRONG_AFFIRMATIVE before DESCRIPTIVE can call it unclear.
_SCAN_WINDOW = 80  # characters of the answer the phrase scans look at

STRONG_NEGATIVE = (
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
)

STRONG_AFFIRMATIVE = (
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
)

DESCRIPTIVE = (
    "the image features",
    "the image shows",
    "the image depicts",
    "the scene features",
    "the scene shows",
)


def _leading_answer(text: str) -> str | None:
    """A bare "yes"/"no" at the very start, the common Stage 2 shape."""
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    words = text.split()
    if words:
        first = words[0].strip(".,!?")
        if first in ("yes", "no"):
            return first
    return None


def _phrase_answer(text: str) -> str | None:
    """An explicit negation or affirmation in the opening of the answer."""
    window = text[:_SCAN_WINDOW]
    if any(phrase in window for phrase in STRONG_NEGATIVE):
        return "no"
    if any(phrase in window for phrase in STRONG_AFFIRMATIVE):
        return "yes"
    return None


def _object_answer(text: str, question: str) -> str | None:
    """Whether the object the question asked about is described in the answer.

    Stage 3 answers in prose, so a caption that never names the queried object
    must not read as "yes". Returns "unclear" rather than "no" when the object
    is absent: not mentioning something is weak evidence it is not there.
    """
    match = re.search(r"is there (?:a |an )?(\w+)", question.lower())
    if not match:
        return None

    obj = match.group(1)
    window = text[:_SCAN_WINDOW]

    if obj not in window:
        return "unclear"
    if f"the {obj}" in window:
        return "yes"
    if any(f"{obj} {preposition}" in window for preposition in ("is", "in", "on", "at")):
        return "yes"
    return None


def extract_yes_no_answer(text: str, question: str = None) -> str:
    """Reduce a model's free-form output to 'yes', 'no' or 'unclear'.

    Handles both the concise answers Stage 2 produces and the prose Stage 3
    produces. The checks are tried in a fixed order — a leading yes/no, then
    explicit phrases, then (given the question) whether the queried object is
    described — and anything reaching the end is 'unclear' rather than guessed.

    Args:
        text: Generated text from the model.
        question: The original question. Without it the object check is skipped
            and generic descriptions fall through to 'unclear'.

    Returns:
        'yes', 'no', or 'unclear'
    """
    text_lower = text.lower().strip()

    answer = _leading_answer(text_lower) or _phrase_answer(text_lower)
    if answer:
        return answer

    if question:
        answer = _object_answer(text_lower, question)
        if answer:
            return answer

    # No strong indicator: a generic description tells us nothing either way.
    if any(pattern in text_lower[:50] for pattern in DESCRIPTIVE):
        return "unclear"
    if text_lower.startswith("the ") and len(text_lower.split()) > 2:
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


@dataclass
class ConfusionCounts:
    """The four outcomes of a yes/no benchmark, plus the answers we could not read.

    Counting these is the only part of POPE scoring that can be got wrong in a
    way the numbers hide, so it happens once. Both metric presentations —
    `compute_metrics` here and `compute_priming_metrics` in
    `compare_priming_strategies` — derive from this rather than recounting.
    """

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    unclear: int = 0

    @property
    def answerable(self) -> int:
        """Everything but the unreadable answers."""
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def total(self) -> int:
        return self.answerable + self.unclear

    @property
    def correct(self) -> int:
        return self.true_positive + self.true_negative

    @property
    def predicted_yes(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def predicted_no(self) -> int:
        return self.true_negative + self.false_negative


def confusion_counts(answers: list[dict]) -> ConfusionCounts:
    """Tally predictions against ground truth.

    Both fields are lower-cased before comparison. One of the two copies this
    replaces did not, so a model answering "Yes" scored zero against a "yes"
    ground truth without anything looking wrong.
    """
    counts = ConfusionCounts()
    for item in answers:
        truth = str(item["answer"]).strip().lower()
        predicted = str(item["predicted_answer"]).strip().lower()

        if predicted not in ("yes", "no"):
            counts.unclear += 1
        elif truth == "yes":
            if predicted == "yes":
                counts.true_positive += 1
            else:
                counts.false_negative += 1
        else:
            if predicted == "yes":
                counts.false_positive += 1
            else:
                counts.true_negative += 1
    return counts


def compute_metrics(answers: list[dict]) -> dict:
    """POPE metrics as fractions in [0, 1].

    - Accuracy: overall correctness
    - Precision: of predicted yes, how many are truly yes?
    - Recall: of true yes, how many are predicted yes?
    - F1: harmonic mean of precision and recall
    - Yes ratio: proportion of yes answers (measures over-generation)
    """
    counts = confusion_counts(answers)

    if counts.answerable == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "yes_ratio": 0.0,
            "num_samples": len(answers),
            "num_unclear": counts.unclear,
        }

    accuracy = counts.correct / counts.answerable
    precision = _ratio(counts.true_positive, counts.predicted_yes)
    recall = _ratio(counts.true_positive, counts.true_positive + counts.false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": counts.predicted_yes / counts.answerable,
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "true_negative": counts.true_negative,
        "false_negative": counts.false_negative,
        "num_samples": len(answers),
        "num_unclear": counts.unclear,
    }


def _ratio(numerator: float, denominator: float) -> float:
    """Guarded division — every POPE metric is undefined on an empty class."""
    return numerator / denominator if denominator > 0 else 0.0
