"""Small, trainer-independent utilities for set-aligned GRPO."""

from .canonical import (
    CanonicalClaspSetResult,
    CanonicalSetAdvantageResult,
    DenseClaspAuxiliaryResult,
    canonical_finite_batch_advantages,
    compute_canonical_clasp_set_advantages,
    compute_dense_clasp_auxiliary_advantages,
    deployment_failure_u_statistic,
    enumerate_k_subsets,
)
from .controller import (
    SignedBucketDualUpdate,
    update_signed_bucket_lambdas,
    update_signed_bucket_lambdas_from_counts,
)
from .fixed_null import (
    FixedNullAdvantageResult,
    fixed_null_finite_batch_advantages,
)
from .fixed_null_clasp import (
    FixedNullClaspSetResult,
    compute_fixed_null_clasp_set_advantages,
)
from .fixed_null_receipt import (
    FixedNullEvaluatorScores,
    FixedNullScoreReceipt,
    load_fixed_null_score_receipt,
    prompt_sha256,
)
from .frontier_cdf import (
    DEFAULT_QUADRATURE_WEIGHTS,
    DEFAULT_THRESHOLD_MULTIPLIERS,
    FrontierCDFResult,
    compute_frontier_cdf_advantages,
)

__all__ = [
    "CanonicalSetAdvantageResult",
    "CanonicalClaspSetResult",
    "DenseClaspAuxiliaryResult",
    "DEFAULT_QUADRATURE_WEIGHTS",
    "DEFAULT_THRESHOLD_MULTIPLIERS",
    "FixedNullAdvantageResult",
    "FixedNullClaspSetResult",
    "FixedNullEvaluatorScores",
    "FixedNullScoreReceipt",
    "FrontierCDFResult",
    "SignedBucketDualUpdate",
    "canonical_finite_batch_advantages",
    "compute_canonical_clasp_set_advantages",
    "compute_dense_clasp_auxiliary_advantages",
    "compute_frontier_cdf_advantages",
    "deployment_failure_u_statistic",
    "enumerate_k_subsets",
    "fixed_null_finite_batch_advantages",
    "compute_fixed_null_clasp_set_advantages",
    "load_fixed_null_score_receipt",
    "prompt_sha256",
    "update_signed_bucket_lambdas",
    "update_signed_bucket_lambdas_from_counts",
]
