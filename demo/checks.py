"""Executable invariants for the CPU demo.

A plot only helps if a human looks at it *and* knows what wrong looks like.
These are the properties that must hold for the pipeline to be correct,
expressed as named assertions so a regression fails loudly and says what broke
rather than quietly producing a slightly different picture.

Each check is deliberately tied to a design decision documented elsewhere in the
repo — the two experts starting as identical copies, the hard routing mask being
exact, Stage 2.5 training only the gate — so that if someone refactors that
decision away, the demo says so.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

EXPERT_KEY = ".mlp.experts."
GATE_KEY = ".mlp.gate."


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False

    @property
    def symbol(self) -> str:
        if self.skipped:
            return "—"
        return "✅" if self.passed else "❌"


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name, True, detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name, False, detail)


def _skip(name: str, detail: str) -> CheckResult:
    return CheckResult(name, True, detail, skipped=True)


def _state_dict(path: Path) -> dict[str, torch.Tensor] | None:
    """Load a checkpoint, unwrapping the {'model_state_dict': ...} envelope."""
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


# ----------------------------------------------------------------------------
# Stage 0 — model construction
# ----------------------------------------------------------------------------


def check_experts_identical_at_init(moe_dir: Path) -> CheckResult:
    """Stage 0 copies the base FFN into *both* experts, so they must start equal.

    This is the premise the whole study rests on: any later divergence between
    the experts is caused by training, not by initialisation.
    """
    name = "stage0: experts initialised identically"
    weights = moe_dir / "model.safetensors"
    if not weights.exists():
        return _skip(name, "Stage 0 output not found")

    tensors = load_file(str(weights))
    pairs = 0
    for key in tensors:
        if f"{EXPERT_KEY}0." not in key:
            continue
        twin = key.replace(f"{EXPERT_KEY}0.", f"{EXPERT_KEY}1.")
        if twin not in tensors:
            return _fail(name, f"Expert 0 has {key} but expert 1 has no counterpart")
        if not torch.equal(tensors[key], tensors[twin]):
            return _fail(name, f"{key} differs between experts at initialisation")
        pairs += 1

    if pairs == 0:
        return _fail(name, "No expert weights found in the Stage 0 checkpoint")
    return _ok(name, f"{pairs} tensor pairs identical")


def check_config_declares_moe(moe_dir: Path) -> CheckResult:
    """config.json must say mistral_moe or the checkpoint loads as a dense model."""
    name = "stage0: config declares mistral_moe"
    config_path = moe_dir / "config.json"
    if not config_path.exists():
        return _skip(name, "Stage 0 output not found")

    config = json.loads(config_path.read_text())
    model_type = config.get("model_type")
    if model_type != "mistral_moe":
        return _fail(
            name,
            f"model_type={model_type!r} — the checkpoint would load as a dense "
            "MistralForCausalLM with randomly initialised FFNs",
        )
    return _ok(name, "model_type='mistral_moe'")


# ----------------------------------------------------------------------------
# Stage 2 — expert specialisation under hard routing
# ----------------------------------------------------------------------------


def check_experts_diverged(moe_dir: Path, stage2_ckpt: Path) -> CheckResult:
    """After Stage 2 the experts must no longer be identical.

    They start as exact copies, so if they are still equal the expert parameters
    were never actually updated — a frozen-parameter or optimiser-scope bug that
    a falling loss curve would not reveal.
    """
    name = "stage2: experts diverged from their shared init"
    state = _state_dict(stage2_ckpt)
    if state is None:
        return _skip(name, "Stage 2 checkpoint not found")

    max_delta = 0.0
    compared = 0
    for key in state:
        if f"{EXPERT_KEY}0." not in key:
            continue
        twin = key.replace(f"{EXPERT_KEY}0.", f"{EXPERT_KEY}1.")
        if twin not in state:
            continue
        delta = (state[key].float() - state[twin].float()).abs().max().item()
        max_delta = max(max_delta, delta)
        compared += 1

    if compared == 0:
        return _fail(name, "No expert weight pairs found in the Stage 2 checkpoint")
    if max_delta == 0.0:
        return _fail(
            name,
            "Experts are still bit-identical after training — expert parameters "
            "were not updated (check requires_grad and the optimiser's param group)",
        )
    return _ok(name, f"max |expert0 − expert1| = {max_delta:.3e} across {compared} tensors")


def check_only_experts_trained(moe_dir: Path, stage2_ckpt: Path) -> CheckResult:
    """Stage 2 freezes everything but the experts — nothing else may move."""
    name = "stage2: only expert weights changed"
    weights = moe_dir / "model.safetensors"
    state = _state_dict(stage2_ckpt)
    if state is None or not weights.exists():
        return _skip(name, "Stage 0 or Stage 2 artifacts not found")

    initial = load_file(str(weights))
    moved = []
    for key, tensor in initial.items():
        if EXPERT_KEY in key or key not in state:
            continue
        if not torch.equal(tensor.float(), state[key].float()):
            moved.append(key)

    if moved:
        preview = ", ".join(moved[:3]) + (" …" if len(moved) > 3 else "")
        return _fail(name, f"{len(moved)} frozen tensors changed: {preview}")
    return _ok(name, "all non-expert weights unchanged")


def check_hard_routing_is_exact(moe_dir: Path) -> CheckResult:
    """Under hard routing a token must be handled by exactly one expert.

    Probes MoELayer directly: with a known mask, the layer's output must equal
    the chosen expert's output elementwise. If dispatch or masking regresses,
    this fails even though loss curves would look plausible.
    """
    name = "stage2: hard routing dispatches exactly"
    config_path = moe_dir / "config.json"
    if not config_path.exists():
        return _skip(name, "Stage 0 output not found")

    from models.custom_mistral import MistralMoEConfig
    from models.moe_layer import MoELayer

    config = MistralMoEConfig(**json.loads(config_path.read_text()))
    torch.manual_seed(0)
    layer = MoELayer(config=config, d_model=config.hidden_size, routing_mode="hard").eval()

    batch, seq = 2, 8
    hidden = torch.randn(batch, seq, config.hidden_size)
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[:, : seq // 2] = 0  # first half visual → expert 0
    layer.routing_mask = mask

    with torch.no_grad():
        actual = layer(hidden)
        expected = torch.where(
            mask.unsqueeze(-1).bool(), layer.experts[1](hidden), layer.experts[0](hidden)
        )

    if not torch.allclose(actual, expected, atol=1e-6):
        worst = (actual - expected).abs().max().item()
        return _fail(name, f"routed output differs from the masked expert output (max {worst:.3e})")
    return _ok(name, f"{batch * seq} tokens routed to exactly the masked expert")


# ----------------------------------------------------------------------------
# Stage 2.5 — router training
# ----------------------------------------------------------------------------


def check_only_gates_trained(stage2_ckpt: Path, stage2_5_ckpt: Path) -> CheckResult:
    """Stage 2.5 trains the gate only; experts must be frozen."""
    name = "stage2.5: only router gates changed"
    before = _state_dict(stage2_ckpt)
    after = _state_dict(stage2_5_ckpt)
    if before is None or after is None:
        return _skip(name, "Stage 2 or Stage 2.5 checkpoint not found")

    moved_experts = [
        key
        for key in before
        if EXPERT_KEY in key
        and key in after
        and not torch.equal(before[key].float(), after[key].float())
    ]
    if moved_experts:
        preview = ", ".join(moved_experts[:3]) + (" …" if len(moved_experts) > 3 else "")
        return _fail(name, f"{len(moved_experts)} expert tensors changed: {preview}")

    gates_changed = any(
        GATE_KEY in key
        and key in after
        and not torch.equal(before[key].float(), after[key].float())
        for key in before
    )
    if not gates_changed:
        return _fail(name, "No gate weights changed — the router did not train")
    return _ok(name, "experts frozen, gates updated")


# ----------------------------------------------------------------------------
# Stage 3 — routing metrics
# ----------------------------------------------------------------------------


def check_routing_entropy_bounds(metrics: dict | None, num_experts: int = 2) -> CheckResult:
    """Entropy over N experts is bounded by ln(N); anything above is mis-computed.

    Catches the common slip of summing a per-layer entropy instead of averaging
    it, which produces a number that scales with model depth and is not an
    entropy at all.
    """
    name = "stage3: routing entropy within [0, ln N]"
    if metrics is None:
        return _skip(name, "No Stage 3 expert metrics found")

    ceiling = math.log(num_experts)
    offenders = [
        (entry["layer"], entry["avg_routing_entropy"])
        for entry in metrics["per_layer"]
        if not 0.0 <= entry["avg_routing_entropy"] <= ceiling + 1e-6
    ]
    aggregate = metrics["aggregate"].get("avg_routing_entropy", 0.0)
    if not 0.0 <= aggregate <= ceiling + 1e-6:
        offenders.append(("aggregate", aggregate))

    if offenders:
        shown = ", ".join(f"layer {layer}: {value:.4f}" for layer, value in offenders[:3])
        return _fail(name, f"ln({num_experts}) = {ceiling:.4f}, but {shown}")
    return _ok(name, f"all ≤ ln({num_experts}) = {ceiling:.4f} (aggregate {aggregate:.4f})")


def check_reported_entropy_bounds(
    runs_dir: Path, filename: str, stage: str, num_experts: int = 2
) -> CheckResult:
    """Any metric named 'entropy' must respect the ln(N) ceiling.

    Separate from the Stage 3 check because this reads the per-stage metrics
    history rather than the ExpertUsageTracker dump. A value that scales with
    layer count is the signature of a per-layer entropy that was summed instead
    of averaged.
    """
    name = f"{stage}: reported entropy within [0, ln N]"
    path = runs_dir / filename
    if not path.exists():
        return _skip(name, f"{filename} not found")

    series = json.loads(path.read_text()).get("entropy") or []
    if not series:
        return _skip(name, "No entropy recorded")

    ceiling = math.log(num_experts)
    offenders = [value for value in series if not 0.0 <= value <= ceiling + 1e-6]
    if offenders:
        return _fail(
            name,
            f"ln({num_experts}) = {ceiling:.4f} but recorded {offenders[0]:.4f} — "
            "looks like a per-layer entropy summed rather than averaged",
        )
    return _ok(name, f"all {len(series)} values ≤ ln({num_experts}) = {ceiling:.4f}")


def check_expert_loads_sum_to_100(metrics: dict | None) -> CheckResult:
    """Load shares are percentages of the routed total and must sum to 100."""
    name = "stage3: per-layer expert loads sum to 100%"
    if metrics is None:
        return _skip(name, "No Stage 3 expert metrics found")

    offenders = []
    for entry in metrics["per_layer"]:
        total = sum(entry["expert_load_distribution"].values())
        if entry["expert_load_distribution"] and abs(total - 100.0) > 0.5:
            offenders.append((entry["layer"], total))

    if offenders:
        shown = ", ".join(f"layer {layer}: {total:.2f}%" for layer, total in offenders[:3])
        return _fail(name, f"loads do not sum to 100%: {shown}")
    return _ok(name, f"{len(metrics['per_layer'])} layers each sum to 100%")


def check_all_layers_reported(metrics: dict | None, expected_layers: int) -> CheckResult:
    """Metrics must cover every layer — a truncated report hides collapse."""
    name = "stage3: metrics cover every layer"
    if metrics is None:
        return _skip(name, "No Stage 3 expert metrics found")

    actual = len(metrics["per_layer"])
    if actual != expected_layers:
        return _fail(name, f"model has {expected_layers} layers but metrics report {actual}")
    return _ok(name, f"{actual}/{expected_layers} layers reported")


# ----------------------------------------------------------------------------
# Numerical health, across every stage
# ----------------------------------------------------------------------------


def check_checkpoints_finite(runs_dir: Path) -> CheckResult:
    """A NaN anywhere in a saved checkpoint poisons every downstream stage."""
    name = "all stages: checkpoints contain no NaN/Inf"
    checkpoints = sorted(runs_dir.rglob("*.pth"))
    if not checkpoints:
        return _skip(name, "No checkpoints found")

    for path in checkpoints:
        state = _state_dict(path)
        if state is None:
            continue
        for key, tensor in state.items():
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                return _fail(name, f"{path.name} has non-finite values in {key}")
    return _ok(name, f"{len(checkpoints)} checkpoints all finite")


def check_losses_finite(runs_dir: Path, metric_files: dict[str, str]) -> CheckResult:
    """Every recorded loss must be a finite number."""
    name = "all stages: recorded losses are finite"
    inspected = 0
    for stage, filename in metric_files.items():
        path = runs_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for series in ("train_loss", "val_loss"):
            for value in data.get(series, []):
                inspected += 1
                if not math.isfinite(value):
                    return _fail(name, f"{stage} {series} contains {value}")
    if inspected == 0:
        return _skip(name, "No loss histories found")
    return _ok(name, f"{inspected} recorded loss values all finite")


# ----------------------------------------------------------------------------
# Analysis — the routing claim, run rather than plotted
# ----------------------------------------------------------------------------


def check_routing_ablation_ran(results_path: Path) -> CheckResult:
    """The analysis script must produce a complete, finite result file.

    This is the only invariant that covers the analysis half of the repo. It
    fails if the config seam breaks again — pointing an analysis script at a
    non-default config used to raise on the ``YOUR_PATH_HERE`` placeholder,
    because the loader took an explicit default path and never consulted
    ``MOE_CONFIG``.
    """
    name = "analysis: routing ablation produced results"
    if not results_path.exists():
        return _fail(name, f"{results_path} was not written")

    results = json.loads(results_path.read_text())
    normal = results["normal_routing"]["losses"]
    flipped = results["flipped_routing"]["losses"]

    if not normal or len(normal) != len(flipped):
        return _fail(name, f"{len(normal)} normal vs {len(flipped)} flipped losses")
    if not all(math.isfinite(value) for value in normal + flipped):
        return _fail(name, "the ablation produced a non-finite loss")
    return _ok(name, f"{len(normal)} samples under both routings, all finite")


def check_flipped_routing_is_worse(results_path: Path) -> CheckResult:
    """Swapping the experts must hurt — on every sample, not just on average.

    Stage 2 trains expert 0 on visual positions and expert 1 on text positions
    and nothing else. If sending each modality to the other expert costs
    nothing, the experts did not specialise and the premise of the study is
    gone. Asserting it per sample rather than on the mean makes the check
    independent of how many fixtures the demo happens to build.
    """
    name = "analysis: flipped routing is worse on every sample"
    if not results_path.exists():
        return _skip(name, "No ablation results")

    results = json.loads(results_path.read_text())
    pairs = list(
        zip(results["normal_routing"]["losses"], results["flipped_routing"]["losses"], strict=True)
    )
    if not pairs:
        return _skip(name, "No ablation results")

    better_flipped = [i for i, (normal, flipped) in enumerate(pairs) if flipped <= normal]
    if better_flipped:
        return _fail(
            name,
            f"{len(better_flipped)}/{len(pairs)} samples were no worse under flipped "
            f"routing (indices {better_flipped[:5]}) — the experts did not specialise",
        )
    delta = results["delta"]["percent"]
    return _ok(name, f"{len(pairs)}/{len(pairs)} samples worse when flipped (+{delta:.1f}% mean)")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def run_all(output_root: Path, metric_files: dict[str, str]) -> list[CheckResult]:
    """Run every invariant against a completed demo run."""
    fixtures = output_root / "fixtures"
    runs = output_root / "runs"
    moe_dir = fixtures / "moe_model"
    stage2_ckpt = runs / "stage2_checkpoints" / "llm_stage2_best.pth"
    stage2_5_ckpt = runs / "stage2_5_checkpoints" / "llm_stage2_5_best.pth"
    ablation = output_root / "analysis" / "routing_ablation" / "routing_ablation_results.json"

    metrics_files = sorted((runs / "expert_metrics").glob("expert_metrics_epoch_*.json"))
    metrics = json.loads(metrics_files[-1].read_text()) if metrics_files else None

    expected_layers = 0
    config_path = moe_dir / "config.json"
    if config_path.exists():
        expected_layers = json.loads(config_path.read_text()).get("num_hidden_layers", 0)

    return [
        check_experts_identical_at_init(moe_dir),
        check_config_declares_moe(moe_dir),
        check_hard_routing_is_exact(moe_dir),
        check_experts_diverged(moe_dir, stage2_ckpt),
        check_only_experts_trained(moe_dir, stage2_ckpt),
        check_only_gates_trained(stage2_ckpt, stage2_5_ckpt),
        check_routing_entropy_bounds(metrics),
        check_reported_entropy_bounds(runs, "training_metrics_stage2.5.json", "stage2.5"),
        check_expert_loads_sum_to_100(metrics),
        check_all_layers_reported(metrics, expected_layers),
        check_checkpoints_finite(runs),
        check_losses_finite(runs, metric_files),
        check_routing_ablation_ran(ablation),
        check_flipped_routing_is_worse(ablation),
    ]
