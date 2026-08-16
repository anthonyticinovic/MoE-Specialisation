"""Turn a demo run's artifacts into two plots and a one-page markdown report.

The point is human inspection: after an agentic refactor you want to look at
something for thirty seconds and know whether the pipeline still behaves, rather
than read a large diff. The plots show the two things most likely to break
silently — how load is split across experts, and whether visual and text tokens
route differently.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from demo import checks  # noqa: E402

logger = logging.getLogger(__name__)

STAGE_METRIC_FILES = {
    "Stage 1": "loss_history_stage1.json",
    "Stage 2": "training_metrics_stage2.json",
    "Stage 2.5": "training_metrics_stage2.5.json",
    "Stage 3": "training_metrics_stage3.json",
    "Dense": "training_metrics_dense.json",
}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _latest_expert_metrics(runs_dir: Path) -> dict | None:
    """Return the most recent per-epoch expert metrics dump, if Stage 3 ran."""
    files = sorted((runs_dir / "expert_metrics").glob("expert_metrics_epoch_*.json"))
    return _load_json(files[-1]) if files else None


def _plot_expert_load(metrics: dict, out_path: Path) -> None:
    """Per-layer expert load split — a flat 50/50 or a hard 100/0 both matter."""
    layers = [entry["layer"] for entry in metrics["per_layer"]]
    expert_0 = [
        entry["expert_load_distribution"].get("expert_0", 0) for entry in metrics["per_layer"]
    ]
    expert_1 = [
        entry["expert_load_distribution"].get("expert_1", 0) for entry in metrics["per_layer"]
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(layers, expert_0, label="Expert 0 (vision)", color="#4C72B0")
    ax.bar(layers, expert_1, bottom=expert_0, label="Expert 1 (text)", color="#DD8452")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, label="Even split")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of routed load (%)")
    ax.set_title("Expert load distribution per layer")
    ax.set_ylim(0, 100)
    ax.set_xticks(layers)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_modality_routing(metrics: dict, out_path: Path) -> None:
    """Visual vs text routing — the effect the whole project is about."""
    aggregate = metrics["aggregate"]
    visual = aggregate.get("visual_routing", {})
    text = aggregate.get("text_routing", {})

    experts = ["expert_0", "expert_1"]
    positions = range(len(experts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(
        [p - width / 2 for p in positions],
        [visual.get(e, 0) for e in experts],
        width,
        label="Visual tokens",
        color="#4C72B0",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [text.get(e, 0) for e in experts],
        width,
        label="Text tokens",
        color="#DD8452",
    )
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(["Expert 0", "Expert 1"])
    ax.set_ylabel("Share of that modality's load (%)")
    ax.set_title("Routing by modality (aggregate)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _losses_table(runs_dir: Path) -> list[str]:
    rows = ["| Stage | Epochs | First → last train | Val | Direction |", "|---|--:|--:|--:|---|"]
    for stage, filename in STAGE_METRIC_FILES.items():
        data = _load_json(runs_dir / filename)
        train_series = (data or {}).get("train_loss") or []
        if not train_series:
            rows.append(f"| {stage} | – | – | – | – |")
            continue
        val_series = data.get("val_loss") or []
        val = f"{val_series[-1]:.4f}" if val_series else "–"
        first, last = train_series[0], train_series[-1]
        if len(train_series) < 2:
            direction = "single epoch"
        elif last < first:
            direction = f"↓ {first - last:.4f}"
        else:
            direction = f"↑ {last - first:.4f}"
        rows.append(
            f"| {stage} | {len(train_series)} | {first:.4f} → {last:.4f} | {val} | {direction} |"
        )
    return rows


def _plot_training_curves(runs_dir: Path, out_path: Path) -> bool:
    """One panel per stage: train and validation loss against epoch."""
    available = [
        (stage, data)
        for stage, filename in STAGE_METRIC_FILES.items()
        if (data := _load_json(runs_dir / filename)) and data.get("train_loss")
    ]
    if not available:
        return False

    fig, axes = plt.subplots(1, len(available), figsize=(3.2 * len(available), 3.0), squeeze=False)
    for ax, (stage, data) in zip(axes[0], available, strict=True):
        epochs = range(1, len(data["train_loss"]) + 1)
        ax.plot(
            epochs, data["train_loss"], marker="o", markersize=3, label="train", color="#4C72B0"
        )
        if data.get("val_loss"):
            val_epochs = range(1, len(data["val_loss"]) + 1)
            ax.plot(
                val_epochs, data["val_loss"], marker="s", markersize=3, label="val", color="#DD8452"
            )
        ax.set_title(stage, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_routing_entropy(metrics: dict, out_path: Path, num_experts: int = 2) -> None:
    """Per-layer entropy against the ln(N) ceiling — collapse shows as a drop."""
    layers = [entry["layer"] for entry in metrics["per_layer"]]
    entropies = [entry["avg_routing_entropy"] for entry in metrics["per_layer"]]
    ceiling = math.log(num_experts)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(layers, entropies, marker="o", markersize=4, color="#4C72B0")
    ax.axhline(
        ceiling, color="grey", linestyle="--", linewidth=0.8, label=f"ln {num_experts} (max)"
    )
    ax.axhline(0, color="#C44E52", linestyle=":", linewidth=0.8, label="0 (fully collapsed)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean routing entropy")
    ax.set_title("Routing entropy per layer")
    ax.set_ylim(-0.05, ceiling * 1.15)
    ax.set_xticks(layers)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _checks_section(results: list[checks.CheckResult]) -> list[str]:
    """Lead the report with the invariant table — the part worth reading first."""
    failures = [r for r in results if not r.passed]
    skipped = [r for r in results if r.skipped]
    passed = len(results) - len(failures) - len(skipped)

    if failures:
        headline = (
            f"**{len(failures)} invariant(s) FAILED** — {passed} passed, {len(skipped)} skipped"
        )
    else:
        headline = f"**All {passed} invariants passed** ({len(skipped)} skipped)"

    lines = ["## Invariants", "", headline, "", "| | Check | Detail |", "|---|---|---|"]
    # Failures first: the thing you need to see is at the top.
    for result in sorted(results, key=lambda r: (r.passed, r.skipped)):
        lines.append(f"| {result.symbol} | {result.name} | {result.detail} |")
    return lines


def write_report(
    output_root: Path, results: list[tuple[str, bool, float]]
) -> tuple[Path, list[checks.CheckResult]]:
    """Write figures plus demo_report.md.

    Returns the report path and the invariant results, so the caller can fail
    the run when an invariant breaks even though every stage exited zero.
    """
    runs_dir = output_root / "runs"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    check_results = checks.run_all(output_root, STAGE_METRIC_FILES)

    lines = [
        "# CPU demo report",
        "",
        "Generated by `python demo/run_demo.py`. Every number here comes from the",
        "real training scripts run against tiny synthetic fixtures on CPU.",
        "",
    ]
    lines += _checks_section(check_results)

    lines += ["", "## Stages", "", "| Stage | Result | Time |", "|---|---|--:|"]
    for label, ok, elapsed in results:
        lines.append(f"| {label} | {'✅ passed' if ok else '❌ failed'} | {elapsed:.1f}s |")

    lines += ["", "## Losses", ""] + _losses_table(runs_dir)
    if _plot_training_curves(runs_dir, figures_dir / "training_curves.png"):
        lines += ["", "![Training curves](figures/training_curves.png)"]

    metrics = _latest_expert_metrics(runs_dir)
    if metrics:
        _plot_expert_load(metrics, figures_dir / "expert_load.png")
        _plot_modality_routing(metrics, figures_dir / "modality_routing.png")
        _plot_routing_entropy(metrics, figures_dir / "routing_entropy.png")

        aggregate = metrics["aggregate"]
        load = aggregate.get("expert_load_distribution", {})
        visual = aggregate.get("visual_routing", {})
        text = aggregate.get("text_routing", {})
        # How differently the two modalities route. Near 0 means the router is
        # ignoring modality; the paper's Stage 3 finding is that this collapses.
        separation = abs(visual.get("expert_0", 0) - text.get("expert_0", 0))

        lines += [
            "",
            "## Routing (Stage 3, final epoch)",
            "",
            f"- Expert load: **{load.get('expert_0', 0):.1f}% / {load.get('expert_1', 0):.1f}%**",
            f"- Mean routing entropy: **{aggregate.get('avg_routing_entropy', 0):.4f}** "
            f"(ln 2 ≈ {math.log(2):.3f} is maximum uncertainty for two experts)",
            f"- High-confidence tokens: **{aggregate.get('high_confidence_fraction', 0):.1%}**",
            f"- Modality separation: **{separation:.1f} points** "
            "(difference in how visual vs text tokens use Expert 0)",
            "",
            "![Routing entropy](figures/routing_entropy.png)",
            "",
            "![Expert load](figures/expert_load.png)",
            "",
            "![Routing by modality](figures/modality_routing.png)",
            "",
            "> On an untrained tiny model these sit near an even split — that is the",
            "> expected result, not a bug. The invariants above are what actually",
            "> gate correctness; these figures are context for reading a failure.",
        ]
    else:
        lines += [
            "",
            "## Routing",
            "",
            "_Stage 3 did not run, so no routing metrics were produced._",
        ]

    report_path = output_root / "demo_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)
    return report_path, check_results
