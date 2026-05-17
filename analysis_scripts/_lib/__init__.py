"""Shared library for analysis scripts.

Consolidates config loading, model loading, representation maths, plotting
helpers and I/O that were previously copy-pasted across the analysis scripts.
"""

from analysis_scripts._lib.config import (
    get_paths,
    load_analysis_config,
    load_training_config,
)
from analysis_scripts._lib.io import (
    format_time,
    load_and_preprocess_image,
    load_json,
    mean_pool_embeddings,
    print_banner,
    save_json,
)
from analysis_scripts._lib.model_loading import (
    LoadedModels,
    load_stage2_models,
    load_stage3_models,
)
from analysis_scripts._lib.representations import (
    compute_cosine_similarity_matrix,
    majority_vote_expert,
)
from analysis_scripts._lib.synthetic_images import SyntheticImageGenerator
from analysis_scripts._lib.viz import set_publication_rcparams, similarity_heatmap

__all__ = [
    "LoadedModels",
    "SyntheticImageGenerator",
    "compute_cosine_similarity_matrix",
    "format_time",
    "get_paths",
    "load_analysis_config",
    "load_and_preprocess_image",
    "load_json",
    "load_stage2_models",
    "load_stage3_models",
    "load_training_config",
    "majority_vote_expert",
    "mean_pool_embeddings",
    "print_banner",
    "save_json",
    "set_publication_rcparams",
    "similarity_heatmap",
]
