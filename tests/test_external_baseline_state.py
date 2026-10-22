"""Actual stepped optimizer restoration and tensor-level checkpoint fidelity."""
# ruff: noqa: E402

from __future__ import annotations

import copy
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from clasp.models.external_baseline_state import (
    restore_rank_state,
    save_rank_state,
    semantic_state_sha256,
)


def training_objects(offset=0.0, optimizer_class=None):
    parameters = [torch.nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7 + offset),
                  torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64) + offset)]
    optimizer_class = optimizer_class or torch.optim.AdamW
    optimizer = optimizer_class(parameters, lr=0.03, betas=(0.8, 0.95),
                                weight_decay=0.07, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 20)
    return parameters, optimizer, scheduler


def take_step(parameters, optimizer, scheduler, step):
    optimizer.zero_grad(set_to_none=True)
    loss = sum((p.square() * (i + 1) + p * (step + 1) / 10).sum()
               for i, p in enumerate(parameters))
    loss.backward()
    optimizer.step()
    scheduler.step()


def read_receipt(directory, rank=0):
    return json.loads((directory / f"external_rank_state_{rank}.json").read_text())


def write_receipt(directory, value, rank=0):
    (directory / f"external_rank_state_{rank}.json").write_text(json.dumps(value))


def test_semantic_hash_preserves_tensor_dtype_shape_values_and_container_types():
    flat = torch.arange(6, dtype=torch.float64)
    digest = semantic_state_sha256(flat)
    assert digest != semantic_state_sha256(flat.float())
    assert digest != semantic_state_sha256(flat.reshape(2, 3))
    assert digest != semantic_state_sha256(flat + 1)
    assert semantic_state_sha256([1]) != semantic_state_sha256((1,))
    assert semantic_state_sha256(1) != semantic_state_sha256(1.0)
    assert semantic_state_sha256(True) != semantic_state_sha256(1)
    assert semantic_state_sha256({"b": 2, "a": flat}) == semantic_state_sha256({"a": flat, "b": 2})


def test_semantic_hash_uses_logical_tensor_values_including_bfloat16_and_scalars():
    transposed = torch.arange(12).reshape(3, 4).t()
    assert not transposed.is_contiguous()
    assert semantic_state_sha256(transposed) == semantic_state_sha256(transposed.contiguous())
    for tensor in (torch.tensor(1.25), torch.ones((2, 3), dtype=torch.bfloat16),
                   torch.empty((0, 3)), torch.tensor([True, False])):
        assert semantic_state_sha256(tensor) == semantic_state_sha256(tensor.clone())
    with pytest.raises(TypeError, match="Unsupported"):
        semantic_state_sha256({"unsupported": {1, 2}})


def test_stepped_adamw_checkpoint_restores_exactly_and_next_update_matches(tmp_path):
    parameters, optimizer, scheduler = training_objects()
    for step in range(3):
        take_step(parameters, optimizer, scheduler, step)
    moments = copy.deepcopy(optimizer.state_dict())
    schedule = copy.deepcopy(scheduler.state_dict())
    saved = save_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=3)
    assert saved["states"]["optimizer"]["semantic_sha256"] == semantic_state_sha256(moments)
    assert saved["states"]["scheduler"]["semantic_sha256"] == semantic_state_sha256(schedule)
    new_parameters, new_optimizer, new_scheduler = training_objects(offset=10.0)
    with torch.no_grad():
        for old, new in zip(parameters, new_parameters):
            new.copy_(old)
    restored = restore_rank_state(tmp_path, new_optimizer, new_scheduler,
                                  rank=0, world_size=1, step=3)
    assert restored["status"] == "passed"
    assert semantic_state_sha256(new_optimizer.state_dict()) == semantic_state_sha256(moments)
    assert semantic_state_sha256(new_scheduler.state_dict()) == semantic_state_sha256(schedule)
    for state in new_optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float64
        assert state["exp_avg_sq"].dtype == torch.float64
        assert int(state["step"].item()) == 3
    take_step(parameters, optimizer, scheduler, 3)
    take_step(new_parameters, new_optimizer, new_scheduler, 3)
    assert all(torch.equal(old, new) for old, new in zip(parameters, new_parameters))
    assert semantic_state_sha256(optimizer.state_dict()) == semantic_state_sha256(new_optimizer.state_dict())
    assert semantic_state_sha256(scheduler.state_dict()) == semantic_state_sha256(new_scheduler.state_dict())


def test_two_ranks_restore_their_own_different_moments_without_fallback(tmp_path):
    expected = []
    for rank in range(2):
        parameters, optimizer, scheduler = training_objects(offset=rank)
        take_step(parameters, optimizer, scheduler, 0)
        save_rank_state(tmp_path, optimizer, scheduler, rank=rank, world_size=2, step=1)
        expected.append(semantic_state_sha256(optimizer.state_dict()))
    assert expected[0] != expected[1]
    for rank in range(2):
        _, optimizer, scheduler = training_objects()
        restore_rank_state(tmp_path, optimizer, scheduler, rank=rank, world_size=2, step=1)
        assert semantic_state_sha256(optimizer.state_dict()) == expected[rank]
    (tmp_path / "optimizer_rank1.pt").unlink()
    _, optimizer, scheduler = training_objects()
    with pytest.raises(FileNotFoundError):
        restore_rank_state(tmp_path, optimizer, scheduler, rank=1, world_size=2, step=1)


def test_file_tampering_is_rejected_before_optimizer_is_changed(tmp_path):
    parameters, optimizer, scheduler = training_objects()
    take_step(parameters, optimizer, scheduler, 0)
    save_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)
    with (tmp_path / "optimizer_rank0.pt").open("ab") as handle:
        handle.write(b"modified")
    _, new_optimizer, new_scheduler = training_objects()
    before = semantic_state_sha256(new_optimizer.state_dict())
    with pytest.raises(ValueError, match="file changed"):
        restore_rank_state(tmp_path, new_optimizer, new_scheduler, rank=0, world_size=1, step=1)
    assert semantic_state_sha256(new_optimizer.state_dict()) == before


def test_changed_tensor_contents_cannot_pass_with_only_a_recomputed_file_hash(tmp_path):
    parameters, optimizer, scheduler = training_objects()
    take_step(parameters, optimizer, scheduler, 0)
    save_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)
    path = tmp_path / "optimizer_rank0.pt"
    value = torch.load(path, map_location="cpu", weights_only=False)
    next(iter(value["state"].values()))["exp_avg"].add_(0.1)
    torch.save(value, path)
    receipt = read_receipt(tmp_path)
    receipt["states"]["optimizer"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_receipt(tmp_path, receipt)
    _, new_optimizer, new_scheduler = training_objects()
    with pytest.raises(ValueError, match="saved contents differ"):
        restore_rank_state(tmp_path, new_optimizer, new_scheduler, rank=0, world_size=1, step=1)


@pytest.mark.parametrize("field,value", [("rank", 1), ("world_size", 4),
                                         ("global_step", 2), ("status", "failed")])
def test_rank_step_and_world_identity_must_match_checkpoint(tmp_path, field, value):
    parameters, optimizer, scheduler = training_objects()
    take_step(parameters, optimizer, scheduler, 0)
    save_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)
    receipt = read_receipt(tmp_path)
    receipt[field] = value
    write_receipt(tmp_path, receipt)
    with pytest.raises(ValueError, match=field):
        restore_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)


def test_rank_owned_path_binding_cannot_point_to_another_file(tmp_path):
    parameters, optimizer, scheduler = training_objects()
    take_step(parameters, optimizer, scheduler, 0)
    save_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)
    receipt = read_receipt(tmp_path)
    receipt["states"]["optimizer"]["file"] = "optimizer_rank1.pt"
    write_receipt(tmp_path, receipt)
    with pytest.raises(ValueError, match="path differs"):
        restore_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)


def test_loader_that_silently_casts_optimizer_moments_is_rejected(tmp_path):
    class CastingAdamW(torch.optim.AdamW):
        def load_state_dict(self, state):
            super().load_state_dict(state)
            for values in self.state.values():
                values["exp_avg"] = values["exp_avg"].float()

    parameters, optimizer, scheduler = training_objects()
    take_step(parameters, optimizer, scheduler, 0)
    save_rank_state(tmp_path, optimizer, scheduler, rank=0, world_size=1, step=1)
    _, cast_optimizer, cast_scheduler = training_objects(optimizer_class=CastingAdamW)
    with pytest.raises(RuntimeError, match="restoration changed saved values"):
        restore_rank_state(tmp_path, cast_optimizer, cast_scheduler, rank=0, world_size=1, step=1)
