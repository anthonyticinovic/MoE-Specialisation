"""
Compare POPE results across different priming strategies.

Shows whether priming Stage 3 with "fake previous answers" improves performance
by exploiting its learned multi-turn conversation behavior.
"""

import json
from pathlib import Path


def compute_metrics(results):
    """Compute POPE metrics from results"""
    correct = 0
    total = 0
    yes_count = 0
    no_count = 0
    unclear_count = 0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for item in results:
        pred = item["predicted_answer"]
        true_answer = item["answer"]

        total += 1

        if pred == "unclear":
            unclear_count += 1
            continue

        if pred == "yes":
            yes_count += 1
        elif pred == "no":
            no_count += 1

        if pred == true_answer:
            correct += 1
            if pred == "yes":
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if pred == "yes":
                false_positives += 1
            else:
                false_negatives += 1

    # Calculate metrics
    answerable = total - unclear_count
    accuracy = (correct / answerable * 100) if answerable > 0 else 0
    unclear_pct = (unclear_count / total * 100) if total > 0 else 0

    precision = (
        (true_positives / (true_positives + false_positives) * 100)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        (true_positives / (true_positives + false_negatives) * 100)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    specificity = (
        (true_negatives / (true_negatives + false_positives) * 100)
        if (true_negatives + false_positives) > 0
        else 0
    )

    yes_pct = (yes_count / answerable * 100) if answerable > 0 else 0
    no_pct = (no_count / answerable * 100) if answerable > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "yes_pct": yes_pct,
        "no_pct": no_pct,
        "unclear_pct": unclear_pct,
        "answerable": answerable,
        "total": total,
    }


def main():
    results_dir = Path("YOUR_PATH_HERE/results/pope_evaluation")
    answers_dir = results_dir / "answers_primed"

    if not answers_dir.exists():
        print(f"❌ Primed answers directory not found: {answers_dir}")
        return

    strategies = ["simple", "conversational", "none"]
    difficulties = ["random", "popular", "adversarial"]

    print("=" * 80)
    print("POPE EVALUATION - PRIMING STRATEGY COMPARISON")
    print("=" * 80)
    print()
    print("Testing whether priming Stage 3 with 'fake previous answers' improves")
    print("performance by exploiting learned multi-turn conversation behavior.")
    print()
    print("Strategies tested:")
    print("  - simple: Prime with 'This image has been analyzed.'")
    print("  - conversational: Prime with full Q&A pair")
    print("  - none: No priming (baseline)")
    print()
    print("=" * 80)
    print()

    # Collect results for each strategy
    all_results = {}

    for strategy in strategies:
        all_results[strategy] = {}

        for difficulty in difficulties:
            answer_file = answers_dir / f"stage3_{difficulty}_{strategy}.json"

            if not answer_file.exists():
                print(f"⚠️  Missing: {answer_file.name}")
                continue

            with open(answer_file) as f:
                results = json.load(f)

            metrics = compute_metrics(results)
            all_results[strategy][difficulty] = metrics

    # Print comparison table
    print()
    print("RESULTS BY PRIMING STRATEGY")
    print("=" * 80)
    print()

    for difficulty in difficulties:
        print(f"\n{'=' * 80}")
        print(f"DIFFICULTY: {difficulty.upper()}")
        print(f"{'=' * 80}")
        print()
        print(
            f"{'Strategy':<20} {'Accuracy':<10} {'Yes%':<10} {'No%':<10} {'Unclear%':<10} {'F1':<10}"
        )
        print("-" * 80)

        for strategy in strategies:
            if difficulty not in all_results[strategy]:
                print(f"{strategy:<20} {'N/A':<10}")
                continue

            m = all_results[strategy][difficulty]
            print(
                f"{strategy:<20} {m['accuracy']:>9.1f}% {m['yes_pct']:>9.1f}% {m['no_pct']:>9.1f}% {m['unclear_pct']:>9.1f}% {m['f1']:>9.1f}"
            )

        print()

    # Compare best vs worst
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Average accuracy across difficulties for each strategy
    avg_accuracies = {}
    avg_unclear = {}
    avg_no_pct = {}

    for strategy in strategies:
        accuracies = [m["accuracy"] for m in all_results[strategy].values() if m]
        unclear_pcts = [m["unclear_pct"] for m in all_results[strategy].values() if m]
        no_pcts = [m["no_pct"] for m in all_results[strategy].values() if m]

        if accuracies:
            avg_accuracies[strategy] = sum(accuracies) / len(accuracies)
            avg_unclear[strategy] = sum(unclear_pcts) / len(unclear_pcts)
            avg_no_pct[strategy] = sum(no_pcts) / len(no_pcts)

    print("Average Performance Across All Difficulties:")
    print()
    print(f"{'Strategy':<20} {'Avg Accuracy':<15} {'Avg Unclear%':<15} {'Avg No%':<10}")
    print("-" * 60)

    for strategy in strategies:
        if strategy in avg_accuracies:
            print(
                f"{strategy:<20} {avg_accuracies[strategy]:>14.1f}% {avg_unclear[strategy]:>14.1f}% {avg_no_pct[strategy]:>9.1f}%"
            )

    print()

    # Find best strategy
    if avg_accuracies:
        best_strategy = max(avg_accuracies, key=avg_accuracies.get)
        best_accuracy = avg_accuracies[best_strategy]

        # Compare to baseline (none)
        if "none" in avg_accuracies:
            baseline_accuracy = avg_accuracies["none"]
            improvement = best_accuracy - baseline_accuracy

            print(f"🏆 Best Strategy: {best_strategy}")
            print(f"   Average Accuracy: {best_accuracy:.1f}%")
            print(f"   Baseline (none): {baseline_accuracy:.1f}%")
            print(f"   Improvement: {improvement:+.1f}%")
            print()

            if improvement > 5:
                print("✅ PRIMING WORKS! Significant improvement detected.")
                print("   Stage 3's learned multi-turn behavior can be exploited.")
            elif improvement > 1:
                print("⚠️  MINOR IMPROVEMENT: Priming helps a bit, but not dramatically.")
            else:
                print("❌ NO IMPROVEMENT: Priming doesn't help (or makes it worse).")
                print("   Stage 3's issues may be too fundamental for priming to fix.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
