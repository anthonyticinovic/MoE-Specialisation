"""Unit tests for the shared analysis_scripts._lib helpers."""

import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis_scripts._lib import (
    compute_cosine_similarity_matrix,
    load_analysis_config,
    majority_vote_expert,
)


class TestCosineSimilarityMatrix:
    def test_identity_diagonal_and_symmetry(self):
        reps = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        m = compute_cosine_similarity_matrix(reps)
        assert m.shape == (3, 3)
        for i in range(3):
            assert m[i, i] == 1.0
        assert np.allclose(m, m.T, atol=1e-6)

    def test_orthogonal_and_parallel(self):
        m = compute_cosine_similarity_matrix(
            [np.array([1.0, 0.0]), np.array([0.0, 5.0]), np.array([2.0, 0.0])]
        )
        assert m[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert m[0, 2] == pytest.approx(1.0, abs=1e-6)


class TestMajorityVoteExpert:
    def test_empty_is_unknown(self):
        assert majority_vote_expert([]) == ("unknown", 0.0)

    def test_decisive_winner(self):
        label, frac = majority_vote_expert([1, 1, 1, 0], confidence_threshold=0.6)
        assert label == "Expert 1"
        assert frac == pytest.approx(0.75)

    def test_below_threshold_is_mixed(self):
        label, frac = majority_vote_expert([0, 0, 1, 1, 1], confidence_threshold=0.9)
        assert label == "mixed"
        assert frac == pytest.approx(0.6)


class TestLoadAnalysisConfig:
    def test_applies_defaults_and_returns(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"concepts": ["cat"]}))
        cfg = load_analysis_config(
            str(p), required_fields=["concepts"], defaults={"layers": [31], "pooling": "mean"}
        )
        assert cfg["concepts"] == ["cat"]
        assert cfg["layers"] == [31]
        assert cfg["pooling"] == "mean"

    def test_missing_required_field_raises(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"foo": 1}))
        with pytest.raises(ValueError, match="missing required field"):
            load_analysis_config(str(p), required_fields=["concepts"])


def test_lib_public_api_imports():
    import analysis_scripts._lib as lib

    for name in lib.__all__:
        assert hasattr(lib, name), f"_lib missing exported name: {name}"


ANALYSIS_DIR = Path(__file__).parent.parent / "analysis_scripts"
ANALYSIS_MODULES = sorted(ANALYSIS_DIR.rglob("*.py"))


class TestLoggingMigration:
    """The analysis scripts log like the rest of the repo, not via print().

    Before this migration the core used ``logging`` while these 838 call sites
    used ``print`` — the inconsistency was more noticeable than either choice
    would have been alone.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_print_calls(self, path):
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not calls, (
            f"{path.name}: {len(calls)} print() call(s) at lines "
            f"{[c.lineno for c in calls]} — use the module logger"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_modules_that_log_define_a_logger(self, path):
        source = path.read_text()
        if "logger." not in source:
            return
        assert "logger = logging.getLogger(__name__)" in source, (
            f"{path.name} logs without defining a module logger"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_entry_points_configure_logging(self, path):
        """A script that logs but never configures a handler emits nothing."""
        source = path.read_text()
        if "__main__" not in source or "logger." not in source:
            return
        assert "setup_logging()" in source, (
            f"{path.name} is an entry point that logs but never calls setup_logging()"
        )


class TestFailuresAreVisible:
    """An analysis that fails must say so at a level someone reads.

    `run_comprehensive_analysis` used to wrap five diagnostics in
    `except Exception` and report each failure with `logger.info`, so a run in
    which every diagnostic failed still printed "complete" and exited zero. A
    similar loop swallowed a whole layer. Both are the same shape as the bare
    `except:` and the `strict=False` before it.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_error_is_reported_at_info_or_debug(self, path):
        offenders = []
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ExceptHandler):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in ("info", "debug")
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "logger"
                ):
                    text = ast.unparse(call).lower()
                    if any(word in text for word in ("error", "failed", "exception")):
                        offenders.append(f"line {call.lineno}")
        assert not offenders, (
            f"{path.name}: reports a failure at INFO/DEBUG ({offenders}). Use "
            "logger.error or logger.exception — an error logged as information "
            "is an error nobody sees."
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_traceback_printed_outside_logging(self, path):
        """`traceback.print_exc()` writes to stderr, bypassing the handler.

        The logging migration test looks for `print()`; this is a print by
        another name, and it was the last thing still writing outside the
        logging system.
        """
        source = path.read_text()
        assert "print_exc" not in source, (
            f"{path.name}: use logger.exception() so the traceback goes through logging"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_leftover_debug_scaffolding(self, path):
        """`DEBUG:`-prefixed messages logged at INFO are someone's debugging run.

        Sixteen of them survived in two analysers — "About to call save_results",
        "Exception caught, continuing" — printed on every run at INFO.
        """
        offenders = [
            call.lineno
            for call in ast.walk(ast.parse(path.read_text()))
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "info"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
            and "DEBUG:" in ast.unparse(call)
        ]
        assert not offenders, (
            f"{path.name}: DEBUG: messages logged at INFO on line(s) {offenders}. "
            "Delete them, or log them at logger.debug."
        )


class TestConfigAndDeviceSeam:
    """The analysis scripts must resolve config and device the same way the
    training scripts do.

    An explicit ``configs/training_config.yaml`` default reaches
    ``load_config`` as a real argument, which skips the ``MOE_CONFIG``
    environment variable entirely. That is why the CPU demo could drive all
    five training scripts but not one analysis script: pointing one at the demo
    fixtures failed on the repo config's ``YOUR_PATH_HERE`` placeholders.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_hardcoded_training_config_default(self, path):
        defaults = [
            node.lineno
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith("configs/training_config.yaml")
        ]
        assert not defaults, (
            f"{path.name}: hardcodes the training config at line(s) {defaults}. "
            "Pass None and let models.utils.common.load_config resolve MOE_CONFIG."
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_analysis_script_loads_weights_without_the_guard(self, path):
        """The analysis loaders read the same checkpoints the stages write.

        `strict=False` accepts a state dict that matches the model in no key at
        all: it loads nothing and returns normally. That is how Stages 2.5 and 3
        spent months starting from their Stage 0 experts while logging success,
        and `_lib/model_loading.py` had the same call.
        """
        assert "strict=False" not in path.read_text(), (
            f"{path.name}: use models.utils.checkpoints.load_matching_weights "
            "instead of a bare strict=False load"
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_unconditional_cuda_default(self, path):
        """``device="cuda"`` as a default breaks the programmatic API on CPU.

        Two analysers guarded the argparse default with ``is_available()`` and
        then hardcoded ``"cuda"`` in the constructor, so they worked from the
        command line and failed when imported.
        """
        offenders = []
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            offenders += [
                f"{node.name}() line {d.lineno}"
                for d in defaults
                if isinstance(d, ast.Constant) and d.value == "cuda"
            ]
        assert not offenders, (
            f"{path.name}: {offenders} default to CUDA. Default to None and "
            "resolve with models.utils.common.get_device()."
        )


class TestAnalysisModulesAreImportable:
    """Every analysis module must import as part of the package.

    Five karpathy scripts used ``from karpathy_utils import ...``, which
    resolves only because Python puts a script's *own* directory on
    ``sys.path`` when you run it as a script. They ran, but no test could
    reach ``main()`` — and the repo-root ``sys.path`` edit lived in
    ``karpathy_utils`` itself, so whether a sibling's ``from models...``
    import worked depended on which import ran first.
    """

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_module_imports(self, path):
        dotted = path.relative_to(ANALYSIS_DIR.parent).with_suffix("").as_posix().replace("/", ".")
        assert importlib.import_module(dotted) is not None

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_no_flat_sibling_imports(self, path):
        """``from karpathy_utils import`` only works by accident of cwd."""
        siblings = {p.stem for p in ANALYSIS_DIR.rglob("*.py") if p.name != "__init__.py"}
        flat = [
            node.module
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] in siblings
        ]
        assert not flat, (
            f"{path.name} imports {flat} by bare module name. Use the full "
            "package path so the module is importable from anywhere."
        )

    @pytest.mark.parametrize("path", ANALYSIS_MODULES, ids=lambda p: p.name)
    def test_entry_points_expose_main(self, path):
        """A script whose work is trapped in ``if __name__ == "__main__"`` can
        only be tested by launching a subprocess and reading files off disk."""
        source = path.read_text()
        if "__main__" not in source or "ArgumentParser" not in source:
            return
        tree = ast.parse(source)
        assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), (
            f"{path.name} builds its parser but has no main() to call"
        )


REPO = ANALYSIS_DIR.parent
SOURCE_FILES = sorted(
    path
    for directory in ("models", "data", "training_scripts", "analysis_scripts", "demo", "tests")
    for path in (REPO / directory).rglob("*.py")
)
MAX_LINES = 800


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_source_file_is_oversized(path):
    """No file over 800 lines.

    Four files were over when this was added — two analysers of 1382 and 1071
    lines, a plotting module of 843 and `train_stage_3.py` at 839. The guideline
    is worth enforcing rather than rediscovering: a reviewer opening a
    1,400-line module learns nothing about the design from it, and a file that
    large is where dead code and mangled docstrings survive unnoticed.
    """
    count = len(path.read_text().splitlines())
    assert count <= MAX_LINES, (
        f"{path.relative_to(REPO)} is {count} lines. Split it along a seam that "
        "already exists — extraction vs metrics, analysis vs plotting, setup vs loop."
    )


class TestExtractConceptSamples:
    """The COCO concept sampler, which used to exist in three copies.

    Two were identical (one an override of the other); the third also handled
    compound concepts. The compound-aware one is what survived, so single-word
    behaviour has to be shown unchanged and compound behaviour shown to work.
    """

    @staticmethod
    def _annotations(tmp_path, captions):
        payload = {
            "images": [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(len(captions))],
            "annotations": [
                {"id": i, "image_id": i, "caption": caption} for i, caption in enumerate(captions)
            ],
        }
        path = tmp_path / "captions.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def _run(self, tmp_path, captions, concepts, per_concept=10):
        from analysis_scripts._lib import extract_concept_samples

        return extract_concept_samples(self._annotations(tmp_path, captions), concepts, per_concept)

    def test_single_word_concepts_match_whole_words(self, tmp_path):
        samples = self._run(
            tmp_path, ["a cat on a mat", "a dog outside", "a catapult launches"], ["cat", "dog"]
        )
        assert [s["caption"] for s in samples["cat"]] == ["a cat on a mat"]
        assert [s["caption"] for s in samples["dog"]] == ["a dog outside"]

    def test_ambiguous_captions_are_skipped(self, tmp_path):
        """A caption naming two concepts would appear on both sides of a
        comparison, so it belongs to neither."""
        samples = self._run(tmp_path, ["a cat and a dog", "just a cat"], ["cat", "dog"])
        assert len(samples["cat"]) == 1
        assert samples["dog"] == []

    def test_compound_concepts_match_all_parts_in_any_order(self, tmp_path):
        """`red_apple` matches a caption with both words.

        Two of the three old copies compared `"red_apple"` against the caption's
        words directly. No caption contains an underscore, so they returned an
        empty set for every compound concept without saying why.
        """
        samples = self._run(
            tmp_path,
            ["a red apple on a table", "an apple that is red", "a green apple", "a red car"],
            ["red_apple"],
        )
        assert len(samples["red_apple"]) == 2

    def test_the_per_concept_cap_is_respected(self, tmp_path):
        samples = self._run(tmp_path, [f"a cat number {i}" for i in range(10)], ["cat"], 3)
        assert len(samples["cat"]) == 3

    def test_every_sample_carries_the_keys_the_analyses_read(self, tmp_path):
        samples = self._run(tmp_path, ["a cat"], ["cat"])
        assert set(samples["cat"][0]) == {"image_id", "caption", "image_path", "concept"}

    def test_under_sampling_is_warned_about(self, tmp_path, caplog):
        """The analyses weight concepts equally, so an unbalanced set skews them."""
        with caplog.at_level("WARNING"):
            self._run(tmp_path, ["a cat"], ["cat", "dog"], per_concept=5)
        assert "Under-sampled" in caplog.text


class TestPopeMetrics:
    """POPE scoring, which existed twice under one name.

    `pope_utils.compute_metrics` returns fractions; the copy in
    `compare_priming_strategies` returned percentages with an extra specificity
    column. Both are wanted, so the *counting* is now shared and only the
    presentation differs — which is what stops them disagreeing about what a
    correct answer is.
    """

    ANSWERS = [
        {"answer": "yes", "predicted_answer": "yes"},  # true positive
        {"answer": "yes", "predicted_answer": "no"},  # false negative
        {"answer": "no", "predicted_answer": "yes"},  # false positive
        {"answer": "no", "predicted_answer": "no"},  # true negative
        {"answer": "no", "predicted_answer": "unclear"},  # unreadable
    ]

    def _counts(self):
        from analysis_scripts.pope_evaluation.pope_utils import confusion_counts

        return confusion_counts(self.ANSWERS)

    def test_each_outcome_is_counted_once(self):
        counts = self._counts()
        assert (counts.true_positive, counts.false_positive) == (1, 1)
        assert (counts.true_negative, counts.false_negative) == (1, 1)
        assert counts.unclear == 1

    def test_unreadable_answers_are_excluded_from_the_denominator(self):
        """Scoring an unreadable answer as wrong would understate accuracy."""
        counts = self._counts()
        assert counts.answerable == 4
        assert counts.total == 5

    def test_case_and_whitespace_do_not_change_the_score(self):
        """The `compare_priming_strategies` copy compared raw strings.

        A model answering "Yes" scored zero against a "yes" ground truth, and
        nothing about the output looked wrong.
        """
        from analysis_scripts.pope_evaluation.pope_utils import confusion_counts

        mixed = [
            {"answer": "yes", "predicted_answer": " Yes "},
            {"answer": "no", "predicted_answer": "NO"},
        ]
        counts = confusion_counts(mixed)
        assert (counts.true_positive, counts.true_negative) == (1, 1)
        assert counts.false_positive == counts.false_negative == 0

    def test_the_two_presentations_agree_up_to_a_factor_of_100(self):
        import importlib

        from analysis_scripts.pope_evaluation.pope_utils import compute_metrics

        priming = importlib.import_module(
            "analysis_scripts.pope_evaluation.compare_priming_strategies"
        )
        fractions = compute_metrics(self.ANSWERS)
        percentages = priming.compute_priming_metrics(self.ANSWERS)

        for key in ("accuracy", "precision", "recall", "f1"):
            assert fractions[key] * 100 == pytest.approx(percentages[key]), key

    def test_an_empty_run_scores_zero_rather_than_dividing_by_zero(self):
        from analysis_scripts.pope_evaluation.pope_utils import compute_metrics

        metrics = compute_metrics([{"answer": "no", "predicted_answer": "unclear"}])
        assert metrics["accuracy"] == 0.0
        assert metrics["num_unclear"] == 1


class TestExtractYesNoAnswer:
    """Characterisation tests for the POPE answer extractor.

    A 126-line ladder of string heuristics turning free-form model output into
    yes/no/unclear. Every POPE number depends on it and nothing tested it. These
    pin the behaviour as it stands so the ladder can be reorganised safely —
    they document what it does, not what it ought to do.
    """

    @staticmethod
    def _extract(text, question=None):
        from analysis_scripts.pope_evaluation.pope_utils import extract_yes_no_answer

        return extract_yes_no_answer(text, question)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Yes", "yes"),
            ("yes, there is a dog", "yes"),
            ("No", "no"),
            ("No.", "no"),
            ("  YES  ", "yes"),
            ("There is no dog in the image", "no"),
            ("There are no cats here", "no"),
            ("The object is not visible", "no"),
            ("I cannot see any dog", "no"),
            ("There is a dog on the grass", "yes"),
            ("The image shows a dog", "yes"),
            ("It contains a bicycle", "yes"),
            # Note the ordering effect documented below: this is "yes".
            ("The image features a sunny day", "yes"),
            ("Maybe", "unclear"),
            ("", "unclear"),
        ],
    )
    def test_direct_and_phrase_matches(self, text, expected):
        assert self._extract(text) == expected

    def test_a_question_lets_it_check_the_queried_object(self):
        """Stage 3 answers in prose, so a generic caption must not read as yes."""
        question = "Is there a dog in the image?"
        assert self._extract("The dog is running across the field", question) == "yes"

    def test_a_caption_about_something_else_is_unclear_not_no(self):
        """Weak evidence is reported as weak rather than guessed at."""
        question = "Is there a dog in the image?"
        assert self._extract("A person stands beside a bright red bicycle", question) == "unclear"

    def test_a_truncated_answer_is_unclear(self):
        question = "Is there a dog in the image?"
        assert self._extract("A man and", question) == "unclear"

    @pytest.mark.parametrize(
        "caption",
        [
            "The image features a sunny day",
            "The image shows a person riding a bicycle",
            "The image depicts a busy street",
        ],
    )
    def test_a_generic_caption_scores_yes_regardless_of_the_question(self, caption):
        """**Known defect, pinned rather than fixed.**

        The affirmative phrase list contains "features a", "shows a" and
        "depicts a", and it is scanned *before* the descriptive-pattern list
        that exists to mark "the image features ..." unclear. So any caption of
        that shape counts as "yes" even when the queried object is absent —
        note the third case answers "Is there a dog?" with a street scene.

        Stage 3 collapsed into producing exactly these generic captions, so this
        plausibly inflates its POPE yes-rate. Changing the order would change
        published numbers, so it is recorded here and raised in the improvement
        plan rather than quietly corrected.
        """
        assert self._extract(caption, "Is there a dog in the image?") == "yes"

    def test_a_plural_object_mention_escapes_the_affirmative_list(self):
        """ "features two dogs" has no "a", so it falls through to the object check."""
        assert self._extract("The image features two dogs", "Is there a dog in the image?") == (
            "unclear"
        )

    def test_only_the_opening_of_a_long_answer_is_examined(self):
        """The phrase scans are windowed, so a late negation does not count.

        Worth pinning: it is a deliberate limit, and a reorganisation that
        widened the window would silently change every POPE number.
        """
        padding = "x" * 100
        assert self._extract(f"{padding} there is no dog") == "unclear"


# Functions over 120 lines that predate the limit. Every one is in analysis
# code the demo cannot reach, so a refactor of it cannot be verified by
# anything — splitting them is deliberately deferred until they are executable
# (see the analysis-coverage note in analysis_scripts/README.md).
#
# The list may shrink. It must never grow: adding an entry means writing a new
# 120-line function, which is what this test exists to prevent.
KNOWN_OVERSIZED_FUNCTIONS = {
    "analysis_scripts/attention_routing_analysis.py::_compute_attention_statistics",
    "analysis_scripts/attention_routing_plots.py::plot_attention_routing_evolution",
    "analysis_scripts/attention_routing_plots.py::plot_expert_attention_correlation",
    "analysis_scripts/compositional_case_study.py::run_analysis",
    "analysis_scripts/cross_concept_similarity_matrix.py::main",
    "analysis_scripts/cross_modality_purity.py::_visualize_results",
    "analysis_scripts/cross_modality_purity.py::run_stage3_alignment_analysis",
    "analysis_scripts/karpathy_evaluation/04_generate_captions.py::generate_captions",
    "analysis_scripts/layer_clustering_analysis.py::collect_representations",
    "analysis_scripts/layer_clustering_analysis.py::main",
    "analysis_scripts/layer_clustering_plots.py::plot_clustering_analysis",
    "analysis_scripts/llava_evaluation/01_llava_wild_eval.py::main",
    "analysis_scripts/pope_evaluation/01_generate_pope_questions.py::generate_pope_questions_for_image",
    "analysis_scripts/pope_evaluation/02_generate_pope_answers.py::generate_answers_primed",
    "analysis_scripts/pope_evaluation/02_generate_pope_answers.py::generate_pope_answers",
    "analysis_scripts/pope_evaluation/compare_priming_strategies.py::main",
}
MAX_FUNCTION_LINES = 120


def _oversized_functions() -> set[str]:
    found = set()
    for path in SOURCE_FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.FunctionDef)
                and node.end_lineno - node.lineno > MAX_FUNCTION_LINES
            ):
                found.add(f"{path.relative_to(REPO).as_posix()}::{node.name}")
    return found


def test_no_new_function_exceeds_the_line_limit():
    """A ratchet, not a clean bill of health.

    The training scripts have enforced this limit since their refactor; this
    extends it to everything else without pretending the backlog is gone. New
    offenders fail immediately; the known ones are listed above with the reason
    they are still there.
    """
    new = _oversized_functions() - KNOWN_OVERSIZED_FUNCTIONS
    assert not new, (
        f"{len(new)} new function(s) over {MAX_FUNCTION_LINES} lines: {sorted(new)}. "
        "Split it rather than adding it to KNOWN_OVERSIZED_FUNCTIONS."
    )


def test_the_oversized_list_has_no_stale_entries():
    """Splitting one of them must also remove it from the list.

    Otherwise the backlog looks larger than it is and the next person cannot
    tell which entries are real.
    """
    stale = KNOWN_OVERSIZED_FUNCTIONS - _oversized_functions()
    assert not stale, f"no longer oversized, remove from the list: {sorted(stale)}"
