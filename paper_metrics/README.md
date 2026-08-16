# Paper metrics

> **Not populated yet.** The metric files from the paper's Stage 3 run are not
> in this repository. Until they are, `make figures` exits with a message
> pointing here. Everything around them — the regeneration command, this schema,
> and the validation in `tests/test_results.py` — is in place and tested.

The point of this directory: the metrics behind the paper's expert-routing
figures, committed so the figures can be regenerated without a GPU, a
checkpoint, or COCO.

```bash
make figures
```

That writes PNGs and a text report to `results/figures/`, from nothing but the
JSON here and the plotting dependencies in `requirements.txt`.

## What belongs here

| Path | Produced by | Feeds |
|---|---|---|
| `stage3/expert_metrics/expert_metrics_epoch_*.json` | `train_stage_3.py`, one per validation epoch | every expert-routing figure |
| `stage3/training_metrics_stage3.json` | `train_stage_3.py` | the loss-vs-specialisation figure |

Nothing here comes from the demo. `make demo` writes its own metrics in the same
format to `demo_output/runs/` — a worked example of the schema on synthetic
data, not a result. `results/` is where the analysis scripts *write*; it is
generated and git-ignored. This directory is the committed input.

## Schema

### `expert_metrics_epoch_<N>.json`

Written by `ExpertUsageTracker` in `training_scripts/_lib/expert_metrics.py`.
All shares are percentages summing to 100 within their group; entropy is in
nats, so the ceiling is ln(2) ≈ 0.693 for two experts.

```jsonc
{
  "per_layer": [                       // one entry per decoder layer, in order
    {
      "layer": 0,
      "expert_load_distribution": { "expert_0": 53.54, "expert_1": 46.46 },
      "avg_routing_entropy": 0.689,
      "high_confidence_fraction": 0.001,
      "visual_vs_text_routing": {
        "visual": { "expert_0": 51.47, "expert_1": 48.53 },
        "text":   { "expert_0": 53.56, "expert_1": 46.44 }
      }
    }
  ],
  "aggregate": {                       // means over all layers
    "expert_load_distribution": { "expert_0": 43.44, "expert_1": 56.56 },
    "avg_routing_entropy": 0.6603,
    "high_confidence_fraction": 0.0015,
    "visual_routing": { "expert_0": 49.88, "expert_1": 50.12 },
    "text_routing":   { "expert_0": 43.37, "expert_1": 56.63 }
  }
}
```

The filename carries the epoch: `plot_expert_metrics.py` reads the number out of
`expert_metrics_epoch_(\d+).json`, so the files must keep that name.

### `training_metrics_stage3.json`

Four parallel arrays, one element per completed epoch:

```jsonc
{
  "epoch":         [1, 2, 3],
  "train_loss":    [2.822, 2.446, 2.238],
  "val_loss":      [2.499, 2.366, 2.301],
  "learning_rate": [7.5e-4, 2.5e-4, 1.0e-4]
}
```

## Adding the files

Copy them from a completed Stage 3 run:

```bash
cp <run>/expert_metrics/expert_metrics_epoch_*.json paper_metrics/stage3/expert_metrics/
cp <run>/training_metrics_stage3.json               paper_metrics/stage3/
make figures
```

`tests/test_results.py` validates whatever is present against the schema above
and skips when the directory is empty, so a malformed or truncated file is
caught before it reaches a figure. Check the total size before committing —
pre-commit rejects added files over 4 MB.
