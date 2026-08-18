"""Run the whole training pipeline end-to-end on CPU against synthetic fixtures.

This drives the *real* training scripts — the same files used on the H100
cluster — by pointing them at a miniature config through ``MOE_CONFIG``. Nothing
is stubbed or reimplemented: Stage 0 builds a real MoE checkpoint, Stage 2
trains real experts under a hard routing mask, Stage 2.5 trains a real learned
gate, and Stage 3 runs the real end-to-end loop and emits real routing metrics.

What it is *not*: a demonstration that a 2-layer randomly-initialised model can
caption images. It cannot. The value is that the pipeline, the checkpoint
formats and the routing instrumentation all work, verifiably, in a few minutes
on a laptop.

    python demo/run_demo.py            # full pipeline + report
    python demo/run_demo.py --stages 0 1 2
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo import build_fixtures, report  # noqa: E402

logger = logging.getLogger(__name__)

# Stage id → (human label, command relative to the repo root).
STAGES: dict[str, tuple[str, list[str]]] = {
    "0": ("Stage 0 — build MoE model", []),  # filled in at runtime (needs paths)
    "1": ("Stage 1 — vision connector", [sys.executable, "training_scripts/train_stage_1.py"]),
    "2": ("Stage 2 — experts, hard routing", [sys.executable, "training_scripts/train_stage_2.py"]),
    "2.5": ("Stage 2.5 — learned router", [sys.executable, "training_scripts/train_stage_2.5.py"]),
    "3": (
        "Stage 3 — end-to-end, soft routing",
        [sys.executable, "training_scripts/train_stage_3.py"],
    ),
    "dense": (
        "Dense — control baseline",
        [sys.executable, "training_scripts/train_dense.py"],
    ),
    "analysis": ("Analysis — routing ablation", []),  # filled in at runtime (needs paths)
}
DEFAULT_STAGES = ["0", "1", "2", "2.5", "3", "dense", "analysis"]


def _reset_run_artifacts(output_root: Path) -> None:
    """Delete everything produced by a previous run, keeping the directory itself.

    Only the fixtures are rebuilt by ``build_fixtures``; checkpoints, metrics,
    figures and logs are outputs and must not survive into a run against
    freshly-built fixtures.
    """
    for name in ("runs", "figures", "logs", "analysis"):
        target = output_root / name
        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared stale %s/", name)


def _run(label: str, command: list[str], env: dict[str, str], log_path: Path) -> tuple[bool, float]:
    """Run one stage, tee its output to a log file, and time it."""
    logger.info("→ %s", label)
    started = time.time()
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started

    if result.returncode == 0:
        logger.info("  ✓ %.1fs", elapsed)
    else:
        logger.error("  ✗ failed (exit %d) — see %s", result.returncode, log_path)
        tail = log_path.read_text().strip().splitlines()[-15:]
        for line in tail:
            logger.error("    %s", line)
    return result.returncode == 0, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo_output", help="Where to write everything")
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=list(STAGES))
    parser.add_argument(
        "--keep-fixtures",
        action="store_true",
        help="Reuse existing fixtures instead of rebuilding them",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_root = (REPO_ROOT / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = output_root / "demo_config.yaml"
    if args.keep_fixtures and config_path.exists():
        logger.info("Reusing fixtures in %s", output_root)
    else:
        logger.info("Building synthetic fixtures...")
        # Checkpoints belong to the fixtures they were trained against. The
        # training scripts resume from `*_latest.pth` when one exists, so a
        # stale run directory would either mix weights from two different
        # models or — because the epoch loop is range(start_epoch, NUM_EPOCHS)
        # — skip training entirely while still reporting success.
        _reset_run_artifacts(output_root)
        config_path = build_fixtures.build(output_root, args.num_images)

    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    fixtures = output_root / "fixtures"
    STAGES["0"] = (
        "Stage 0 — build MoE model",
        [
            sys.executable,
            "-m",
            "models.utils.create_moe_model",
            "--base-model",
            str(fixtures / "base_llm"),
            "--output",
            str(fixtures / "moe_model"),
        ],
    )

    # The analysis half of the repo used to be unreachable from here: its
    # config loader took an explicit default path, which bypassed MOE_CONFIG.
    # Running one analysis script against the checkpoints the earlier stages
    # just produced is what keeps that seam honest.
    STAGES["analysis"] = (
        "Analysis — routing ablation",
        [
            sys.executable,
            "analysis_scripts/routing_ablation_experiment.py",
            "--image-dir",
            str(fixtures / "data" / "images"),
            "--annotations",
            str(fixtures / "data" / "coco_captions.json"),
            "--num-samples",
            str(args.num_images),
            "--output-dir",
            str(output_root / "analysis" / "routing_ablation"),
        ],
    )

    # The scripts read paths from MOE_CONFIG; everything else mirrors a normal
    # local run. HF_HUB_OFFLINE guarantees the demo never reaches the network.
    env = {
        **os.environ,
        "MOE_CONFIG": str(config_path),
        "PYTHONPATH": str(REPO_ROOT),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    results: list[tuple[str, bool, float]] = []
    for stage in args.stages:
        label, command = STAGES[stage]
        ok, elapsed = _run(label, command, env, logs_dir / f"stage_{stage}.log")
        results.append((label, ok, elapsed))
        if not ok:
            logger.error("\nPipeline stopped at %s.", label)
            break

    report_path, check_results = report.write_report(output_root, results, args.stages)
    total = sum(elapsed for _, _, elapsed in results)

    stages_ok = all(ok for _, ok, _ in results) and len(results) == len(args.stages)
    failed_checks = [check for check in check_results if not check.passed]

    logger.info("\n%s", "─" * 58)
    if stages_ok:
        logger.info("All %d stages passed in %.1fs", len(results), total)
    else:
        logger.error("Pipeline failed after %.1fs", total)

    # A stage exiting zero is not the same as the pipeline being correct: the
    # invariants are what catch a refactor that runs cleanly but behaves wrongly.
    if failed_checks:
        logger.error("\n%d invariant(s) FAILED:", len(failed_checks))
        for check in failed_checks:
            logger.error("  ✗ %s", check.name)
            logger.error("      %s", check.detail)
    else:
        passed = sum(1 for check in check_results if check.passed and not check.skipped)
        logger.info("All %d invariants passed", passed)

    logger.info("Report: %s", report_path)
    return 0 if stages_ok and not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
