# Design notes

Why this code is shaped the way it is. The [README](../README.md) says what to
run; this says what the alternatives were and why they were rejected. It also
says, at the end, what I would do differently.

The study asks a narrow question — **can a modality split be imposed on a
Mixture-of-Experts language model, and does it survive being handed to a learned
router?** Almost every decision below follows from wanting an unambiguous answer
to that, rather than a good captioning model.

---

## 1. Two experts, not eight

Standard sparse-MoE work (Switch, Mixtral) uses 8–64 experts and top-k routing,
because the goal is capacity at fixed FLOPs. That is a different goal.

Here the experts are **hypotheses about structure**: expert 0 is "the vision
one", expert 1 is "the text one". Two is the smallest number that makes the
claim testable and the largest that keeps it interpretable:

- **The routing question becomes binary.** "Did this token go to the vision
  expert?" has an answer. With eight experts and top-2 routing, "did modality
  specialisation emerge?" needs a clustering argument before it needs a
  measurement, and the clustering argument is contestable.
- **The ablation is exact.** Swapping two experts is a complete permutation of
  the routing decision, so `analysis_scripts/routing_ablation_experiment.py`
  measures the *whole* effect of specialisation on the loss, not one arm of it.
  This is the check the CPU demo runs on every push.
- **Entropy has a fixed ceiling.** Routing entropy lives in `[0, ln 2]`, so a
  collapse is visible as an absolute number rather than relative to a baseline.
  Two of the demo's invariants depend on that bound.

`MoELayer` takes `num_experts` as a constructor argument and the soft-routing
path iterates over `self.experts`, so that half generalises. The rest does not:
`_hard_routing_forward` indexes `experts[0]` and `experts[1]` against mask values
0 and 1, and the vision/text metric split assumes a two-way modality partition.
Raising the count is a change of experiment, not a config change.

**Cost:** almost no capacity story. This architecture does not save FLOPs and is
not meant to; every token still passes through one full-width `MistralMLP`.

## 2. A fixed mask before a learned gate

The stage order — hard routing (2), router-only (2.5), end-to-end (3) — is the
central design decision, and it exists because the obvious alternative fails.

**Handing routing to a gate too early collapses it.** Going from the hard mask
straight to end-to-end soft routing sends everything to one expert and lets the
other atrophy — the observation that Stage 2.5 exists to work around, and the
reason its docstring says so. Nothing in the loss objects: one full-width FFN is
a perfectly good FFN, so a collapsed router is a dense model with dead weights
and a constant gate, and its loss curve looks fine.

So specialisation is *imposed* before it is *learned*:

| Stage | Router | Trains | What it establishes |
|---|---|---|---|
| 2 | position mask, no gate | experts only | The experts diverge and become modality-specific |
| 2.5 | learned gate | gate only | A router can be fitted to experts that already differ |
| 3 | learned gate | attention + gate + experts | Whether the split survives end-to-end training |

Stage 2's mask is derived from token position — the first `num_visual_tokens`
of the sequence are the projected image, the rest are text — so it is exact and
free. No gate is trained, and no routing decision is learned. That is the point:
it removes routing as a variable while the experts specialise.

Stage 2.5 exists because **the intermediate step is where the interesting result
is**. It answers "can a router be fitted to experts that are already different?"
separately from "does the split survive training?" — and the answers turn out to
differ, which is only visible because the stages are separate. `MoELayer` builds
the gate in *both* modes, unused in hard routing, so a Stage 2 checkpoint is
structurally identical to a Stage 3 one and cross-stage loading needs no
surgery. `initialize_gate()` resets it at the start of Stage 2.5: after Stage 2
the gate holds its initialisation, and resetting it with a small variance
(`std=0.05`) breaks symmetry deliberately rather than by accident.

**This is also where the headline finding comes from.** Stage 3 collapses
routing. Had the pipeline been one end-to-end stage, that collapse would have
been indistinguishable from "the architecture never worked".

## 3. Straight-Through Gumbel-Softmax, not top-k

Soft routing needs one expert per token in the forward pass and a gradient to
the gate in the backward pass. Those pull in opposite directions: `argmax` has
no useful gradient.

The three real options:

| Approach | Forward | Gradient to gate | Why not |
|---|---|---|---|
| Dense mixture (weighted sum of both experts) | both experts | exact | Not routing. Every token uses both experts, so "which expert handles vision" has no answer. |
| Top-k with an auxiliary load-balancing loss (Switch) | sparse | only via the chosen expert's weight | Designed to keep 64 experts busy. With two experts the balancing term dominates the signal you are trying to measure. |
| **Straight-Through Gumbel-Softmax** | sparse | through the full soft distribution | Chosen. |

The implementation is the standard STE trick in
[`models/moe_layer.py`](../models/moe_layer.py):

```python
router_onehot = hard_onehot - router_probs.detach() + router_probs
```

Forward, this is exactly `hard_onehot` — a one-hot selection. Backward, the
first two terms have no gradient, so the whole gradient flows through
`router_probs`. The gate is therefore trained on the *distribution* it produced,
including the expert it did not pick, while the forward pass stays sparse.

Two details that matter:

- **Gumbel noise only during training.** `if self.training:` adds the noise;
  evaluation is deterministic. Without that, every analysis run would produce
  different routing for the same input, and none of the routing figures would
  be reproducible. The analysis loader also fixes the seed for the same reason.
- **Temperature is annealed, 2.0 → 1.0** (`router_temperature`). High
  temperature early keeps the distribution soft so gradients reach both experts;
  low temperature later sharpens towards the hard selection the forward pass is
  already making. It is set through `_forward_temperature` rather than the
  `forward()` argument so the annealing schedule does not have to thread through
  every HuggingFace decoder call — a wart, and the cause of a real bug: one
  analysis script set `.temperature`, which nothing reads, and silently ran at
  the default.

Stage 2.5 adds a load-balancing term and an **entropy bonus that decays**
(`0.001 × 0.95^epoch`): explore first, specialise later. The bonus is subtracted
from the loss because it is maximised.

## 4. What FSDP forced

The model is Mistral-7B with every FFN doubled, trained on 4× H100 with FSDP.
Three shapes in this code exist only because of that.

### Every expert must be called on every rank

FSDP gathers a sharded parameter when the module that owns it runs. If rank 0
calls expert 1 and rank 1 does not, rank 0 waits at an all-gather that rank 1
never reaches, and **the job hangs** — no error, no traceback, just a wall clock
running until the scheduler kills it.

Sparse routing makes this easy to hit: with a small batch, it is entirely
ordinary for one rank to have no tokens for an expert. So `MoELayer` runs the
expert anyway:

```python
else:
    # FSDP: call expert with a dummy input so all ranks trigger the same
    # all-gather collective. The zero weight ensures no contribution to output.
    dummy_out = expert(hidden_flat[:1])
    final_hidden_states[:1] += dummy_out.to(final_hidden_states.dtype) * 0.0
```

One token through an unused expert, multiplied by zero. It looks like dead code
and reads like a bug; it is the thing standing between a routing imbalance and a
silently hung job. Deleting it would pass every single-GPU test in this repo.

The same reasoning drives `broadcast_flag` in
[`_lib/runtime.py`](../training_scripts/_lib/runtime.py): rank 0 decides from
the filesystem whether a checkpoint exists, and that decision is broadcast, so
every rank takes the same branch. A rank that skips a load the others perform
desynchronises everything after it.

**The general rule this codebase follows: under FSDP, control flow must not
depend on data that differs between ranks.** Not the token counts, not the
filesystem, not a `try/except` that only one rank enters.

### The embedding is not sharded

The training loops call `llm.model.embed_tokens(input_ids)` directly, to build
the `[visual | text]` sequence before it enters the decoder. Reaching into a
sharded module's parameters outside its own forward gets you an unmaterialised
shard.

So the embedding goes in `ignored_modules`, and the reference is cached *before*
wrapping. It must also already be on the target device, or FSDP treats it as a
newly-added parameter. Three constraints, one line each, none of them guessable
from reading the loop.

### Wrapping granularity and precision

`transformer_auto_wrap_policy` with `transformer_layer_cls={MistralMLP}` shards
at **expert** granularity rather than decoder-layer granularity — the experts
are where the doubled parameters are. `use_orig_params=True` keeps
`named_parameters()` meaningful, which is what makes the per-stage
"exactly these parameters are trainable" checks in
`tests/test_training_steps.py` possible at all.

Mixed precision is bfloat16 for parameters, reductions and buffers, and
`GradScaler` is explicitly **disabled** in Stage 3 (`enabled=False`). Loss
scaling exists to stop float16 gradients underflowing; bfloat16 has float32's
exponent range and does not need it. An enabled scaler there would be cargo
cult.

## 5. Why the per-stage forward passes were not unified

Roughly 60–70% of each `train_stage_*.py` is shared boilerplate, and most of it
*was* extracted into `training_scripts/_lib/` — the run context, the FSDP
wrapper, the backbones, the dataloaders, the checkpoint guard, the shifted loss.

The forward pass was deliberately left duplicated. The stages differ in where
the `no_grad` and `autocast` boundaries fall:

- Stage 1 trains the connector, so it must sit **outside** `no_grad`; the
  encoder and token embedding run outside `autocast` too.
- Stages 2, 2.5 and dense freeze the connector and run it **inside** `no_grad`;
  Stage 2 alone keeps the token embedding outside.
- Stage 3 runs only the encoder under `no_grad`.

Folding those into one helper needs three flags, and the flags would encode
exactly the distinctions that matter. The failure mode of getting one wrong is
not a crash — it is a slightly different gradient path, in bfloat16, on a GPU.
**A CPU demo cannot detect that**, which means the change would be unverifiable
by the only mechanism this repo has for verifying changes.

So the duplication stays, the reasoning is recorded at the top of
[`_lib/pipeline.py`](../training_scripts/_lib/pipeline.py), and a test
(`test_stage_1_and_3_keep_their_own_loss_alignment`) fails if someone unifies
them without meaning to. Keeping each stage independently readable and
independently reproducible was worth more here than DRY.

What *did* come out of `train_stage_3.py` is its construction code — model,
data, checkpoint loading and resumption, now in
[`_lib/stage3_setup.py`](../training_scripts/_lib/stage3_setup.py). That is
mechanical setup with no numerics in it, so moving it is verifiable by the
demo's exact-equality baseline; the forward pass is not.

## 6. How correctness is checked without a GPU

The problem: the pipeline needs 4× H100 and a week, and almost every failure
mode above is silent. Loss curves fall whether or not the experts are training.

The approach is a **structurally real miniature**: `demo/` builds a 2-layer
Mistral (hidden 64), a 4-patch CLIP tower, a word-level tokenizer and 24
synthetic images, then runs *the actual training scripts* against them by
pointing `MOE_CONFIG` at a generated config. Nothing is stubbed or
reimplemented. What differs is scale and device — CPU fallbacks for FSDP, 8-bit
loading and FlashAttention, all keyed off a single `on_gpu` flag on
`RunContext`.

On top of that sit **14 executable invariants** (`demo/checks.py`), each tied to
a design decision above, each with a test that proves it can fail:

- the two experts start bit-identical (§1: they are copies of the base FFN) and
  have diverged after Stage 2 (§2);
- hard routing dispatches each token to exactly the masked expert (§2);
- Stage 2 leaves every non-expert weight untouched; Stage 2.5 moves the gates
  and only the gates (§2);
- routing entropy stays within `[0, ln 2]` (§1);
- flipped routing costs more loss on **every** sample (§1's ablation).

That last one is the specialisation claim of the paper, reduced to something a
machine checks on every push.

The honest framing: a randomly-initialised 2-layer model cannot caption
anything, and the demo's routing metrics sit near an even split. It is not a
result. It proves the pipeline, the checkpoint formats and the routing
instrumentation are intact — and it means a reader can verify that themselves in
twenty seconds without a cluster.

---

## What I would do differently

**Checkpoint formats should have been versioned from the start.** Stage 2's
`best` checkpoint began as a bare `state_dict` and later became a full training
checkpoint with the weights nested under `model_state_dict`. The loaders in
Stages 2.5 and 3 were never updated. They load with `strict=False`, so every key
landed in `unexpected_keys`, the model quietly kept its Stage 0 weights, and the
run logged that the checkpoint had loaded successfully.

**This predates the published Stage 3 runs.** Stage 2.5 trained routers over
unspecialised experts and Stage 3 fine-tuned from Stage 0 weights. My own
regression tests caught it, months later, when I started running real training
steps against the demo fixtures. The Stage 2 analyses are unaffected — the
analysis library handled both checkpoint shapes.

The fix is `models/utils/checkpoints.py`: `state_dict_from` accepts either
shape, and `load_matching_weights` **raises** when a non-empty state dict matches
the model in no key at all. `strict=False` is genuinely needed here — a stage
loads a subset of the model on purpose — but "matches nothing" is never a
legitimate outcome, and that is the case that used to pass unnoticed. A
structure test now fails any script that reaches for a bare `strict=False`.

The generalisable lesson is not "write more tests". It is that **`strict=False`
converts a schema mismatch into a silent no-op**, and any code that both loads a
subset and reports success needs a floor under it.

Three smaller ones:

- **The temperature would be a forward argument, not `_forward_temperature`.**
  Threading it through the HuggingFace decoder call was inconvenient, so it
  became an attribute set before the forward. That is why one analysis script
  could set the wrong name and silently run at the default temperature.
- **The analysis scripts would have been given the same seams as the training
  scripts from day one.** They hardcoded their config path, which bypassed the
  `MOE_CONFIG` mechanism entirely, so ~12,000 lines could not be reached by any
  test. Retrofitting that was cheap; not having it cost more than it saved.
- **`ExpertUsageTracker` and the connector would size themselves from the loaded
  models.** Both originally hardcoded the 7B/ViT-L constants — 32 layers, 1024 →
  4096 — which is invisible until you try to run against anything else, and the
  first thing that tries is your own test.

## What this design does not claim

- **It is not an efficiency result.** Two full-width experts per layer is more
  parameters and no fewer FLOPs per token than the dense baseline.
- **It is not a captioning result.** Stage 2 is competitive with a fine-tuned
  LLaVA reference on COCO, which is a useful sanity check and not the point; the
  Stage 3 drop is the studied phenomenon.
- **Two experts do not generalise cleanly to N.** The mask, the metrics and the
  ablation all assume a two-way modality split.
- **`train_dense.py` is the only control.** It matches Stage 2 in data, loss and
  frozen modules and differs in the FFN, so a difference is attributable to the
  MoE layer. There is no ablation over expert count, no ablation over where in
  the sequence the split falls, and no seed sweep.
