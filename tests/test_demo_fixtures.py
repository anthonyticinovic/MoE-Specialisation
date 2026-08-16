"""Tests for the CPU demo's synthetic fixtures.

The demo is the repo's end-to-end safety net, so its fixtures need to stay
valid: the annotations must parse as real COCO, the filenames must match what
COCO_Loader reconstructs from image ids, and the generated config must satisfy
the same validation the training scripts apply. CPU-only, no network.
"""

import json

import pytest
import yaml

from demo import build_fixtures


@pytest.fixture(scope="module")
def fixtures_root(tmp_path_factory):
    """Build the fixtures once for the whole module (model saves are the slow part)."""
    root = tmp_path_factory.mktemp("demo")
    config_path = build_fixtures.build(root, num_images=6)
    return root, config_path


class TestDataset:
    def test_coco_annotations_parse_with_pycocotools(self, fixtures_root):
        """COCO_Loader drives pycocotools, so the file must satisfy it."""
        from pycocotools.coco import COCO

        root, _ = fixtures_root
        coco = COCO(str(root / "fixtures" / "data" / "coco_captions.json"))
        assert len(coco.imgs) == 6
        assert len(coco.getAnnIds()) == 6

    def test_image_filenames_match_loader_convention(self, fixtures_root):
        """COCO_Loader builds paths as f"{image_id:012d}.jpg" — files must match."""
        root, _ = fixtures_root
        image_dir = root / "fixtures" / "data" / "images"
        annotations = json.loads((root / "fixtures" / "data" / "coco_captions.json").read_text())[
            "annotations"
        ]
        for annotation in annotations:
            expected = image_dir / f"{annotation['image_id']:012d}.jpg"
            assert expected.exists(), f"Loader would look for {expected.name} and miss"

    def test_llava_records_have_image_and_conversation(self, fixtures_root):
        """LLaVA_Loader masks question tokens, so both turns must be present."""
        root, _ = fixtures_root
        records = json.loads((root / "fixtures" / "data" / "llava_instruct.json").read_text())
        assert records
        for record in records:
            assert (root / "fixtures" / "data" / "images" / record["image"]).exists()
            speakers = [turn["from"] for turn in record["conversations"]]
            assert speakers == ["human", "gpt"]
            assert "<image>" in record["conversations"][0]["value"]


class TestGeneratedConfig:
    def test_config_passes_the_real_validator(self, fixtures_root):
        """The demo config must survive the same check the training scripts run."""
        from models.utils.common import load_config

        _, config_path = fixtures_root
        config = load_config(str(config_path))
        assert config["paths"]["output_dir"]

    def test_every_input_path_exists(self, fixtures_root):
        """A path typo would surface as a confusing failure mid-pipeline."""
        from pathlib import Path

        # Outputs of the pipeline itself: output_dir is created by the training
        # scripts and moe_model_path by Stage 0, so neither exists yet.
        produced_by_pipeline = {"output_dir", "moe_model_path"}

        _, config_path = fixtures_root
        paths = yaml.safe_load(config_path.read_text())["paths"]
        for key, value in paths.items():
            if key in produced_by_pipeline:
                continue
            assert Path(value).exists(), f"paths.{key} points at a missing {value}"

    def test_all_training_stages_configured(self, fixtures_root):
        """run_demo.py invokes every stage; each needs its config section."""
        _, config_path = fixtures_root
        config = yaml.safe_load(config_path.read_text())
        for section in (
            "training_stage1",
            "training_stage2",
            "training_stage2.5",
            "training_stage3",
            "dense_control",
        ):
            assert section in config, f"Missing config section: {section}"


class TestTinyModels:
    def test_llm_vocab_matches_tokenizer(self, fixtures_root):
        """A vocab mismatch produces an index error deep inside the loss."""
        from transformers import AutoTokenizer, MistralConfig

        root, _ = fixtures_root
        base = root / "fixtures" / "base_llm"
        tokenizer = AutoTokenizer.from_pretrained(str(base))
        config = MistralConfig.from_pretrained(str(base))
        assert config.vocab_size >= tokenizer.vocab_size

    def test_sequence_fits_in_position_embeddings(self, fixtures_root):
        """Visual tokens + the 128-token caption must fit the context window."""
        from transformers import CLIPVisionConfig, MistralConfig

        root, _ = fixtures_root
        vision = CLIPVisionConfig.from_pretrained(str(root / "fixtures" / "clip"))
        llm = MistralConfig.from_pretrained(str(root / "fixtures" / "base_llm"))

        visual_tokens = (vision.image_size // vision.patch_size) ** 2 + 1
        # COCO_Loader pads captions to max_length=128.
        assert visual_tokens + 128 <= llm.max_position_embeddings


class TestRunHygiene:
    """Regression tests for stale-state bugs in the demo runner."""

    def test_fixtures_are_deterministic(self, tmp_path):
        """Two builds must produce byte-identical models.

        Without this, consecutive demo runs train against different random
        weights and their reported numbers cannot be compared.
        """
        import torch
        from safetensors.torch import load_file

        weights = []
        for build_index in range(2):
            root = tmp_path / f"build_{build_index}"
            build_fixtures.build(root, num_images=4)
            weights.append(load_file(str(root / "fixtures" / "base_llm" / "model.safetensors")))

        first, second = weights
        assert first, "No weights were compared"
        assert first.keys() == second.keys()
        for key in first:
            assert torch.equal(first[key], second[key]), f"{key} differs between builds"

    def test_reset_removes_previous_run_outputs(self, tmp_path):
        """Checkpoints must not outlive the fixtures they were trained against.

        The training scripts resume from `*_latest.pth`, so a surviving run
        directory silently mixes weights from two different models — and,
        because the epoch loop is range(start_epoch, NUM_EPOCHS), a completed
        run makes the next one train for zero epochs while reporting success.
        """
        from demo.run_demo import _reset_run_artifacts

        for name in ("runs", "figures", "logs"):
            directory = tmp_path / name
            directory.mkdir(parents=True)
            (directory / "stale.txt").write_text("from a previous run")
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "keep.txt").write_text("rebuilt separately")

        _reset_run_artifacts(tmp_path)

        for name in ("runs", "figures", "logs"):
            assert not (tmp_path / name).exists(), f"{name}/ survived the reset"
        assert (fixtures / "keep.txt").exists(), "fixtures/ must be left to build_fixtures"

    def test_reset_is_safe_on_a_clean_directory(self, tmp_path):
        """First run has nothing to clear."""
        from demo.run_demo import _reset_run_artifacts

        _reset_run_artifacts(tmp_path)
