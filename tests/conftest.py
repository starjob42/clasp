"""Session-wide helpers for the published CLASP test suite.

Two responsibilities:

1. Test isolation: some dev-profile helpers (e.g. ``seal_reward_environment``)
   intentionally mutate the process environment. An autouse fixture restores
   ``os.environ`` after every test so that suite order cannot change results.
2. Publishing boundary: modules that exercise frozen campaign bundles, private
   research trees or reviewed historical pins are skipped when those artifacts
   are absent from the working tree (e.g. a fresh public clone). Pass
   ``--run-unpublished-tests`` to force them to run regardless.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Modules that require unpublished private artifacts (research data trees,
# frozen campaign bundles, reviewed contract pins, local checkpoints).
UNPUBLISHED_TEST_MODULES = frozenset("""
test_adapter_endpoint_audit.py
test_additional_evidence_analysis.py
test_aggregate_publication_contract.py
test_aggregate_sota.py
test_audit_sota_advancement.py
test_base_sft_campaign.py
test_build_checkpoint_evidence.py
test_canonical_bandit_experiment.py
test_chunked_confirmation_analysis_v1.py
test_chunked_sft_operational_custody_v1.py
test_clasp_grpo_fresh_dev_artifacts.py
test_collect_timing_campaign_v2.py
test_confirmation_7b_execution_v3.py
test_confirmation_7b_factory_v3.py
test_confirmation_artifact_verifier_v3.py
test_confirmation_campaign_v3.py
test_confirmation_op_metadata_v3.py
test_confirmation_op_paper_render_v3.py
test_confirmation_op_statistics_v3.py
test_confirmation_paper_publisher_v3.py
test_confirmation_paper_results_v1.py
test_confirmation_paper_update_watch_v3.py
test_confirmation_reconciliation_handoff_v4.py
test_cvrp_probe_execution_v2.py
test_cvrp_probe_execution_v3.py
test_cvrp_probe_held_submit_transport_v2.py
test_cvrp_probe_held_submit_transport_v3.py
test_cvrp_probe_operator_handoff_v2.py
test_cvrp_probe_operator_handoff_v3.py
test_cvrp_probe_rejected_submission_history_v3.py
test_cvrp_timing_handoff_v2.py
test_cvrp_timing_operator_v2.py
test_cvrp_timing_probe_authority_v1.py
test_cvrp_timing_probe_history_v1.py
test_cvrp_timing_probe_history_v3.py
test_cvrp_timing_probe_kernel_v1.py
test_cvrp_timing_probe_worker_v1.py
test_evaluation_rescue_source_packet.py
test_execute_evaluation_rescue_v1.py
test_foarl_f638_matched_comparison.py
test_formal_crypto_transport_v1.py
test_formal_source2_operational_v1.py
test_formal_source2_worker_v2.py
test_formal_source_layer_v2.py
test_greedy_op96_layout_v2.py
test_greedy_op96_paper_v1.py
test_grpo_incumbent_heldout300_campaign.py
test_grpo_verified_evaluation_v2.py
test_llmcosolver_f638_contract_v2.py
test_llmcosolver_f638_public_data_manifest_v2.py
test_monitor_native_probe_jobs_v1.py
test_monitor_native_probe_jobs_v2.py
test_monitor_native_probe_jobs_v3.py
test_monitor_timing_campaign_v2.py
test_native_holdout_a100_once_v1.py
test_native_holdout_a100_profile_20260907.py
test_native_holdout_audit_bundle_20260907.py
test_native_holdout_successor_a100_once_v1.py
test_native_holdout_successor_packaging_20260907.py
test_native_holdout_successor_profile_20260907.py
test_native_timing_addendum_v2.py
test_native_timing_standby_peer_v2.py
test_native_timing_standby_v2.py
test_one_shot_analysis_admission_v1.py
test_one_shot_analysis_gate_v1.py
test_one_shot_campaign_v1.py
test_one_shot_confirmation_inputs_v1.py
test_one_shot_confirmation_transfer_v3.py
test_one_shot_local_runner_v1.py
test_one_shot_teacher_candidate_v1.py
test_one_shot_teacher_data_v1.py
test_one_shot_teacher_training_v1.py
test_one_shot_teacher_v2.py
test_one_shot_training_collector_v1.py
test_op_scalability_remote_launch.py
test_op_scalability_sweep.py
test_paper_only_runtime_upgrade_v4c.py
test_prepare_confirmation_round_v1.py
test_prepare_evaluation_rescue_v1.py
test_prepare_native_timing_execution_v2.py
test_publication_docs.py
test_reconciled_paper_wait_v4.py
test_registration_promotion_v1.py
test_remote_chunk_provider_v1.py
test_remote_chunk_provider_v2.py
test_resource_extension_collection_v1.py
test_rloo_decoder_search_analysis.py
test_rloo_decoder_search_data.py
test_rloo_decoder_search_engine.py
test_rloo_decoder_search_worker.py
test_sft_anchor_admission_patch_v3.py
test_sft_anchor_admission_v3_actual.py
test_sft_anchor_admission_v3_collection.py
test_sft_anchor_candidate_selection.py
test_sft_anchor_cohorts.py
test_sft_anchor_collection.py
test_sft_anchor_confirmation_custody.py
test_sft_anchor_confirmatory_analysis.py
test_sft_anchor_evaluation_custody.py
test_sft_anchor_input_overlap.py
test_sft_anchor_paper_gate.py
test_sft_anchor_paper_gate_v2.py
test_sft_anchor_paper_gate_v3.py
test_sft_anchor_pending_qos_transition_v1.py
test_sft_anchor_pending_qos_transition_v2.py
test_sft_anchor_qos_custody.py
test_sft_anchor_recovery_accounting.py
test_sft_anchor_recovery_collection.py
test_sft_anchor_recovery_ops.py
test_sft_anchor_recovery_v3.py
test_sft_anchor_selection_metrics.py
test_small_tsp_canonical_pilot_v1.py
test_small_tsp_matched40_pilot.py
test_subprocess_crypto_verifier_v1.py
test_teacher_7b_execution_v2.py
test_teacher_7b_factory_v2.py
test_teacher_calibration_7b_runner_v2.py
test_teacher_followup_v2.py
test_teacher_followup_watch_v2.py
test_teacher_only_7b_v2.py
test_teacher_op_analysis_v2.py
test_teacher_op_paper_render_v2.py
test_teacher_operational_runtime_v3.py
test_teacher_operational_runtime_v3b.py
test_teacher_operational_runtime_v3c.py
test_teacher_paper_publisher_v2.py
test_teacher_paper_update_watch_v2.py
test_teacher_reconciliation_v3.py
test_training_collection_admission_v1.py
test_training_launcher.py
test_training_telemetry_audit.py
test_tsp_fcdf_beta_pilot.py
test_v4_numerical_continuity_v1.py
test_vocabulary_validation_reconciliation_v3.py
test_workflow_runtime_migration_v4b.py
""".split())

# Markers that a private research workspace is present in this checkout. When
# none of these directories exist (fresh public clone) the unpublished modules
# above are skipped automatically.
PRIVATE_WORKSPACE_MARKERS = (
    "data",
    "checkpoints",
    "outputs",
    "results",
    "models",
    "artifacts",
    "logs",
)


def pytest_addoption(parser):
    group = parser.getgroup("clasp")
    group.addoption(
        "--run-unpublished-tests",
        action="store_true",
        default=False,
        help="Run modules that require unpublished research artifacts even if they are absent.",
    )


@pytest.fixture(autouse=True)
def _restore_process_environment():
    """Undo environment mutations that tests or sealed runtimes leak."""
    baseline = dict(os.environ)
    yield
    for key in list(os.environ):
        if key not in baseline:
            del os.environ[key]
    for key, value in baseline.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def pytest_ignore_collect(collection_path, config):
    """Keep unpublished modules from even being imported in a public clone.

    Several of these modules read private files or import optional runtime
    dependencies at import time, so a plain mark-based skip would still raise
    during collection.
    """
    if config.getoption("--run-unpublished-tests"):
        return None
    root = config.rootpath
    if any((root / marker).is_dir() for marker in PRIVATE_WORKSPACE_MARKERS):
        return None  # private research workspace present: run everything
    relative = os.path.relpath(str(collection_path), str(root / "tests"))
    if relative in UNPUBLISHED_TEST_MODULES:
        return True
    return None


# Belt-and-braces: collect_ignore is honored even for explicitly named paths,
# unlike pytest_ignore_collect in some collection orders.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

collect_ignore = (
    sorted(UNPUBLISHED_TEST_MODULES)
    if not any((_ROOT / marker).is_dir() for marker in PRIVATE_WORKSPACE_MARKERS)
    else []
)


def pytest_configure(config):
    if config.getoption("--run-unpublished-tests"):
        collect_ignore[:] = []
