"""Paper admission/format guards; fabricated rows are formatting fixtures only."""
import copy
import hashlib
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"


def load(path, name):
    body = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


m = load(ROOT / "paper/rloo9_20261008/confirmation_results_v1.py", "paper_confirmation_tests")
a = load(EXP / "confirmatory_analysis.py", "frozen_confirmation_format_math")


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def write(root, relative, raw):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o440)
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def packet(tmp_path, *, real_sources=False):
    folder = tmp_path / "evidence/packet"
    folder.mkdir(parents=True)
    (tmp_path / "evidence/chunks").mkdir()
    bindings = {}
    for name, filename in m.SOURCE_FILES.items():
        body = (EXP / filename).read_bytes() if real_sources else b"raise AssertionError('Unreviewed code must never execute')\n"
        bindings[name] = write(folder, filename, body)
    bindings["contract"] = write(folder, "contract.json", encoded({"contract_sha256": "1" * 64}))
    bindings["registry"] = write(folder, "registry.json", encoded({
        "protocol": m.ANALYSIS_PROTOCOL + "-registry", "schema_version": 1, "entries": []}))
    request = {"protocol": m.PROTOCOL + "-request", "schema_version": 1,
        "evidence_root": "evidence", "packet_root": "packet", "chunk_root": "chunks",
        "bindings": bindings, "expected_registry_sha256": bindings["registry"]["sha256"]}
    # An empty-registry test never reaches the remaining producer artifacts.
    bindings["promotion"] = {"path": "missing_promotion.json", "sha256": "0" * 64, "size_bytes": 1}
    return request, folder


def formatting_fixture(successes=15):
    """Never fed to build_layer or labelled actual/native execution evidence."""
    roster = [{"draw_id": f"draw{i:02}", "train_seed": 4 if i < 2 else 100 + i} for i in range(20)]
    contract = {"contract_sha256": "1" * 64, "candidate_arm": "rloo_cardinal_beta005",
        "seed_roster": roster, "alpha_fraction": [1, 40]}
    metrics = {"N": 500, "n": 499, "feasibility_percent": 99.8, "q_sigma_percent": 45.0,
        "u10_percent": 99.0, "own_signed_gap_percent": -0.125, "own_clipped_gap_percent": 0.5}
    draws = [{**row, "D": int(i < successes), "decoder_results": {"k8": {
        "macro_q_sigma_delta_pp": .01 if i < successes else -.01,
        "per_task": {task: {"feasibility_delta_pp": 0.0, "joint_n": 498,
            "joint_signed_gap_delta_pp": -.125} for task in m.TASKS}}}, "per_slot_descriptive": [
        {"task": task, "arm": arm, "state": "admitted", "physical_metrics": {"k8": copy.deepcopy(metrics)}}
        for task in m.TASKS for arm in (contract["candidate_arm"], "foarl")]}
        for i, row in enumerate(roster)]
    selected = [{"feasible": True, "signed_gap": -.01}, {"feasible": False, "signed_gap": None}]
    report = {"protocol": m.ANALYSIS_PROTOCOL, "schema_version": 1,
        "status": "closed_registered_chunked_round_analyzed", "formal_claim_eligible": True,
        "synthetic_fixture": False, "contract_sha256": contract["contract_sha256"],
        "candidate_arm": contract["candidate_arm"], "primary_decoder": "k8", "global_test_count": 1,
        "seed_draws": draws, "all_registered_draws_retained": True,
        "failed_missing_unadmitted_draws_counted_zero": True, "duplicate_seed_values_retained_as_distinct_draws": True,
        "seed_replacement_performed": False, "resources": {"all82slots_retained": True},
        "once_sft_sidecars": {task: {"metrics": {"k8": {}}, "selected": {"k8": selected}} for task in m.TASKS},
        "global_test": a.exact_global_test(successes),
        "conditional_majority_seed_gate_probability_established": successes >= 15,
        **{field: False for field in ("expected_mean_improvement_established", "population_feasibility_noninferiority_established",
            "published_FOARL_replication_or_SOTA_established", "individual_task_or_greedy_significance_claimed",
            "benchmark_superiority_established", "native_mutations")}}
    report["receipt_sha256"] = m.sha(m.canonical(report))
    return contract, report


def reseal(report):
    report["receipt_sha256"] = m.sha(m.canonical({k: v for k, v in report.items() if k != "receipt_sha256"}))


def test_missing_request_publishes_nothing_and_preserves_real_pdf(tmp_path):
    pdf = ROOT / "paper/rloo9_20261008/main.pdf"
    before = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert m.load_completed(root=tmp_path) is None and m.build_layer(root=tmp_path) is None
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("marker", ["synthetic_fixture", "synthetic_offline_fixture"])
def test_nested_synthetic_request_denied_before_replay(tmp_path, monkeypatch, marker):
    request = {"protocol": m.PROTOCOL + "-request", "schema_version": 1, "hidden": [{marker: True}]}
    write(tmp_path, "request.json", encoded(request))
    monkeypatch.setattr(m, "_replay", lambda *args: pytest.fail("Fixture cannot reach replay"))
    with pytest.raises(ValueError, match="Synthetic"):
        m.build_layer("request.json", root=tmp_path)


@pytest.mark.parametrize("fault", ["writable", "linked", "parent_escape", "absolute", "sha", "size"])
def test_packet_evidence_must_be_immutable_scoped_and_exact(tmp_path, fault):
    binding = write(tmp_path, "owned.json", b"{}")
    if fault == "writable": (tmp_path / "owned.json").chmod(0o640)
    elif fault == "linked":
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        binding["path"] = "alias/owned.json"
    elif fault == "parent_escape": binding["path"] = "../owned.json"
    elif fault == "absolute": binding["path"] = str(tmp_path / "owned.json")
    elif fault == "sha": binding["sha256"] = "0" * 64
    else: binding["size_bytes"] += 1
    with pytest.raises(ValueError): m.PacketReader(tmp_path)(binding)


def test_unreviewed_source_never_executes(tmp_path):
    request, folder = packet(tmp_path)
    write(tmp_path, "request.json", encoded(request))
    with pytest.raises(ValueError, match="Unknown independently reviewed"):
        m.build_layer("request.json", root=tmp_path)


def test_exact_real_core_keeps_empty_native_registry_closed(tmp_path):
    request, folder = packet(tmp_path, real_sources=True)
    before = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.iterdir()}
    write(tmp_path, "request.json", encoded(request))
    with pytest.raises(ValueError, match="Default analysis registry is empty"):
        m.build_layer("request.json", root=tmp_path)
    assert before == {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.iterdir()}
    assert list((tmp_path / "evidence/chunks").iterdir()) == []
    assert a.PRODUCTION_ADMISSION_READY is False


@pytest.mark.parametrize("fault", ["partial", "fixture", "missing_draw", "replaced_seed", "wrong_D", "second_test",
    "wrong_alpha", "dropped_failures", "missing_sft", "dropped_costs", "population_claim", "SOTA", "greedy_test", "mean_claim", "bad_seal"])
def test_replayed_report_cannot_expand_scope_or_drop_draws(fault):
    contract, report = formatting_fixture()
    if fault == "partial": report["status"] = "partial"
    elif fault == "fixture": report["synthetic_fixture"] = True
    elif fault == "missing_draw": report["seed_draws"].pop()
    elif fault == "replaced_seed": report["seed_draws"][0]["train_seed"] += 1
    elif fault == "wrong_D": report["seed_draws"][0]["D"] = True
    elif fault == "second_test": report["global_test_count"] = 2
    elif fault == "wrong_alpha": report["global_test"]["alpha_fraction"] = [1, 20]
    elif fault == "dropped_failures": report["failed_missing_unadmitted_draws_counted_zero"] = False
    elif fault == "missing_sft": report["once_sft_sidecars"].pop("op")
    elif fault == "dropped_costs": report["resources"]["all82slots_retained"] = False
    elif fault == "population_claim": report["population_feasibility_noninferiority_established"] = True
    elif fault == "SOTA": report["published_FOARL_replication_or_SOTA_established"] = True
    elif fault == "greedy_test": report["individual_task_or_greedy_significance_claimed"] = True
    elif fault == "mean_claim": report["expected_mean_improvement_established"] = True
    if fault != "bad_seal": reseal(report)
    else: report["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError): m._validate_report(contract, report)


def test_format_retains_all_twenty_draws_duplicate_seeds_and_own_support():
    contract, report = formatting_fixture()
    m._validate_report(contract, report)  # Structural formatting check only.
    tables = m._format_tables(contract, report)
    assert set(tables) == {"confirmation_global_table.tex", "confirmation_op_draw_table.tex",
        "confirmation_cvrp_draw_table.tex", "confirmation_sft_table.tex", "confirmation_paired_gate_table.tex"}
    for task in m.TASKS:
        text = tables[f"confirmation_{task}_draw_table.tex"]
        assert all(text.count(row["draw_id"] + " &") == 1 for row in report["seed_draws"])
        assert "draw00 & 4 &" in text and "draw01 & 4 &" in text
        assert "499/500" in text and "-0.125" in text and "own feasible support" in text
    assert "15/20" in tables["confirmation_global_table.tex"]
    assert "\\frac{1}{40}" in tables["confirmation_global_table.tex"]
    assert "$1-\\alpha$" in tables["confirmation_global_table.tex"]
    assert "not twenty independent SFT trials" in tables["confirmation_sft_table.tex"]
    assert "0.010" in tables["confirmation_paired_gate_table.tex"]
    assert "498" in tables["confirmation_paired_gate_table.tex"]
    assert "joint feasible support" in tables["confirmation_paired_gate_table.tex"]


def test_failed_slot_stays_visible_and_D_zero():
    contract, report = formatting_fixture()
    row = report["seed_draws"][-1]
    row["per_slot_descriptive"][0].update(state="unadmitted", physical_metrics=None)
    row.pop("decoder_results")
    row["D"] = 0
    reseal(report)
    m._validate_report(contract, report)
    text = m._format_tables(contract, report)["confirmation_op_draw_table.tex"]
    assert "draw19 & 119 & 0 & -- & -- & --" in text
    paired = m._format_tables(contract, report)["confirmation_paired_gate_table.tex"]
    assert "draw19 & 0 & -- & -- & -- & --" in paired


def test_formatter_cannot_be_used_as_publication_source(tmp_path, monkeypatch):
    contract, report = formatting_fixture()
    write(tmp_path, "producer_report.json", encoded(report))
    monkeypatch.setattr(m, "_format_tables", lambda *a: pytest.fail("Report alone cannot reach formatter"))
    assert m.build_layer(root=tmp_path) is None


def test_bad_external_request_sha_denied_before_replay(tmp_path):
    write(tmp_path, "request.json", b"{}")
    with pytest.raises(ValueError, match="request byte identity"):
        m.build_layer("request.json", root=tmp_path, expected_request_sha256="0" * 64)


@pytest.mark.parametrize("producer_value", [0, 0.0])
def test_replay_rejects_boolean_numeric_equivalence(tmp_path, monkeypatch, producer_value):
    # An isolated replay harness exercises the actual comparison branch. Its
    # private packet is never passed to load_completed/build_layer, and cannot
    # satisfy the production report schema or trusted key registry.
    request, folder = packet(tmp_path)
    body = b'''class Reader:
    def __init__(self, root): self.checked = {}
class Collector:
    ChunkReader = Reader
    @staticmethod
    def ChunkStore(root): return None
    @staticmethod
    def verify_source_snapshot(*args): return None
def prepare_context(*args, **kwargs): return {"collector": Collector}
def analyze(*args, **kwargs): return None, {"native_mutations": False}
'''
    request["bindings"]["adapter"] = write(folder, "private_replay_harness.py", body)
    monkeypatch.setattr(m, "KNOWN_PACKETS", {m.sha(body): {
        name: request["bindings"][name]["sha256"] for name in
        ("frozen_analysis", "frozen_collector", "promotion_verifier")}})
    for name in ("source_chunks", "finalized", "sidecars", "attempt_cost"):
        request["bindings"][name] = write(folder, name + ".json", b"{}")
    request["bindings"]["report"] = write(folder, "report.json", encoded({"native_mutations": producer_value}))
    with pytest.raises(ValueError, match="typed canonical confirmation replay differs"):
        m._replay(request, tmp_path / "evidence")


def test_later_alpha_spending_is_printed_exactly():
    contract, report = formatting_fixture()
    report["global_test"] = a.exact_global_test(15, round_number=20)
    contract["alpha_fraction"] = report["global_test"]["alpha_fraction"]
    text = m._format_tables(contract, report)["confirmation_global_table.tex"]
    assert "\\frac{1}{20971520}" in text
    assert "0.000000" not in text and "$1-\\alpha$" in text
