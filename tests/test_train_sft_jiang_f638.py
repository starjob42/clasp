from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.entrypoints import train_sft_jiang_f638 as jiang


class _CharacterTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


class _Dataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.column_names = list(rows[0]) if rows else []
        self.events = []
        self._fingerprint = "fake-fingerprint"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            selected = self.rows[index]
            keys = selected[0].keys() if selected else self.column_names
            return {key: [row[key] for row in selected] for key in keys}
        return self.rows[index]

    def shuffle(self, *, seed):
        self.events.append(("shuffle", seed))
        self.rows.reverse()
        return self

    def map(self, function, *, batched, fn_kwargs, desc):
        self.events.append(("map", batched, desc))
        batch = {key: [row[key] for row in self.rows] for key in self.column_names}
        additions = function(batch, **fn_kwargs)
        for index, text in enumerate(additions["text"]):
            self.rows[index]["text"] = text
        if "text" not in self.column_names:
            self.column_names.append("text")
        return self


class _ProjectedDataset(_Dataset):
    features = None
    from_generator_kwargs = None

    @classmethod
    def from_generator(cls, generator, **kwargs):
        cls.features = kwargs["features"]
        cls.from_generator_kwargs = kwargs
        return cls(list(generator(**kwargs["gen_kwargs"])))


class _Features(dict):
    pass


class _Value:
    def __init__(self, dtype):
        self.dtype = dtype


class _Row(list):
    def tolist(self):
        return list(self)


class _Matrix:
    def __init__(self, rows):
        self.rows = [_Row(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return self.rows[row][column]
        return self.rows[key]

    def __setitem__(self, key, value):
        row, columns = key
        indices = range(*columns.indices(len(self.rows[row])))
        for index in indices:
            self.rows[row][index] = value


class _FakeLanguageModelCollator:
    def __init__(self, *, tokenizer, mlm):
        assert mlm is False

    def torch_call(self, examples):
        rows = [example["input_ids"] for example in examples]
        return {"input_ids": _Matrix(rows), "labels": _Matrix(rows)}


def _row(output="Route: [0, 1, 0], Objective: 2"):
    return {
        "instruction": "Solve exactly.",
        "input": "Node 0; Node 1",
        "output": output,
    }


def _chunk_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in jiang.REQUIRED_CHUNK_CHECKPOINT_FILES:
        if name == "trainer_state.json":
            (checkpoint / name).write_text(
                '{"global_step": %d, "max_steps": 31250}\n' % step,
                encoding="utf-8",
            )
        else:
            (checkpoint / name).write_bytes(b"x")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    return checkpoint


def test_fixed_recipe_is_the_corrected_f638_contract():
    assert jiang.UPSTREAM_COMMIT == "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
    assert jiang.UPSTREAM_MODEL_REVISION == "86e6e1eccea424c7fb77d59f7d2a9125c0673bfd"
    assert jiang.MAX_SEQ_LENGTH == 20_000
    assert jiang.EXPECTED_TRAIN_ROWS == 500_000
    assert jiang.TARGET_STEP == 31_250
    assert jiang.LORA_R == jiang.LORA_ALPHA == 64
    assert jiang.LORA_BIAS == "all"
    # The pinned Qwen2.5-7B base tokenizer has no chat template, and the exact
    # Transformers 4.50.3 Trainer checkpoint therefore does not emit this
    # optional tokenizer artifact.  Requiring it would break every 5k resume.
    assert "chat_template.jinja" not in jiang.REQUIRED_CHUNK_CHECKPOINT_FILES
    assert set(jiang.LORA_TARGET_MODULES) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert jiang.PER_DEVICE_TRAIN_BATCH_SIZE == 4
    assert jiang.GRADIENT_ACCUMULATION_STEPS == 4
    assert jiang.SAVE_STEPS == 5_000
    assert jiang.ALLOWED_CHUNK_STOP_STEPS == (
        5_000,
        10_000,
        15_000,
        20_000,
        25_000,
        30_000,
        31_250,
    )


def test_standalone_stage_contract_contains_every_matched_sft_field():
    expected_keys = {
        "recipe",
        "jiang_source_commit",
        "model_loader",
        "load_in_4bit",
        "full_finetune",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "explicit_bias_trainability_restore_after_unsloth",
        "lora_target_modules",
        "use_gradient_checkpointing",
        "use_rslora",
        "loftq_config",
        "train_lm_head",
        "train_embed_tokens",
        "seed",
        "dataset_shuffle_seed",
        "dataset_shuffle_passes",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "effective_train_batch_size",
        "learning_rate",
        "optim",
        "weight_decay",
        "warmup_steps",
        "lr_scheduler_type",
        "num_train_epochs",
        "max_steps",
        "max_seq_length",
        "packing",
        "min_train_samples_per_task",
        "deduplicate_train_rows",
        "no_reason",
        "prompt_style",
        "released_response_template",
        "response_boundary_prefix",
        "response_boundary_policy",
        "append_eos_token",
        "completion_only_loss",
        "memory_safe_loss",
        "supervised_logits_only",
        "evaluation_max_new_tokens",
        "evaluation_max_prompt_length",
        "evaluation_requires_zero_prompt_truncation",
        "evaluation_no_reason",
        "clasp_tsp_prompt_knn",
        "clasp_tsp_prompt_knn_distances",
        "clasp_tsp_input_order",
        "clasp_tsp_relabel_order",
        "clasp_tsp_route_canonicalize",
        "clasp_tsp_route_only_output",
        "clasp_tsp_label_tta_schedule",
        "clasp_mis_relabel_order",
        "clasp_mis_jiang_response_output",
        "clasp_mis_prompt_node_degrees",
        "clasp_mis_prompt_low_degree_k",
    }
    assert set(jiang.MATCHED_SFT_STAGE_HPARAMETERS) == expected_keys
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["prompt_style"] == "jiang_f638"
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["lora_dropout"] == 0.0
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["lora_bias"] == "all"
    assert (
        jiang.MATCHED_SFT_STAGE_HPARAMETERS[
            "explicit_bias_trainability_restore_after_unsloth"
        ]
        is True
    )
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["dataset_shuffle_passes"] == 1
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["evaluation_max_new_tokens"] == 3_000
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["evaluation_max_prompt_length"] == 128_072
    assert (
        jiang.MATCHED_SFT_STAGE_HPARAMETERS[
            "evaluation_requires_zero_prompt_truncation"
        ]
        is True
    )
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["evaluation_no_reason"] is True
    assert jiang.MATCHED_SFT_STAGE_HPARAMETERS["clasp_mis_jiang_response_output"] == "0"


def test_runtime_audit_requires_every_exact_official_version():
    observed = jiang.audit_runtime_versions(lambda name: jiang.PINNED_RUNTIME_VERSIONS[name])
    assert observed == dict(jiang.PINNED_RUNTIME_VERSIONS)

    def drifted(name):
        return "99.0" if name == "trl" else jiang.PINNED_RUNTIME_VERSIONS[name]

    with pytest.raises(jiang.JiangF638SFTError, match="trl='99.0'"):
        jiang.audit_runtime_versions(drifted)


def test_projected_loader_ignores_heterogeneous_metadata_without_changing_text(tmp_path):
    train_dir = tmp_path / "cvrp" / "train"
    train_dir.mkdir(parents=True)
    rows = [
        {
            "instruction": "  Solve exactly.  ",
            "input": "nodes: α\\nβ",
            "output": " route: [0, 1, 0] ",
            "instance_json": [[[0, 0], [1, 1]], [0, 2], 7],
        },
        {
            "instruction": "Escaped \\\"instruction\\\"",
            "input": "line one\nline two",
            "output": "Objective: 2",
            "instance_json": {"incompatible": "object"},
        },
        {
            "instruction": "Duplicate",
            "input": "same",
            "output": "same",
            "instance_json": "incompatible scalar",
        },
        {
            "instruction": "Duplicate",
            "input": "same",
            "output": "same",
            "instance_json": None,
        },
    ]
    source = train_dir / "train_cvrp.json"
    source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    dataset = jiang.load_projected_training_dataset(
        train_dir,
        cache_dir=tmp_path / "cache",
        expected_rows=len(rows),
        dataset_class=_ProjectedDataset,
        features_class=_Features,
        value_class=_Value,
    )

    expected = [
        {field: row[field] for field in jiang.REQUIRED_TEXT_FIELDS} for row in rows
    ]
    assert dataset.rows == expected
    assert dataset.column_names == list(jiang.REQUIRED_TEXT_FIELDS)
    assert list(_ProjectedDataset.features) == list(jiang.REQUIRED_TEXT_FIELDS)
    assert all(value.dtype == "string" for value in _ProjectedDataset.features.values())
    assert _ProjectedDataset.from_generator_kwargs["keep_in_memory"] is False
    assert _ProjectedDataset.from_generator_kwargs["num_proc"] == 1


def test_projected_loader_rejects_non_string_training_text(tmp_path):
    train_dir = tmp_path / "op" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "train_op.json").write_text(
        json.dumps(
            [
                {
                    "instruction": "Solve",
                    "input": ["not", "text"],
                    "output": "answer",
                    "instance_json": [[[0, 0]], [1], 3.0],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(jiang.JiangF638SFTError, match="invalid required field 'input'"):
        jiang.load_projected_training_dataset(
            train_dir,
            cache_dir=None,
            expected_rows=1,
            dataset_class=_ProjectedDataset,
            features_class=_Features,
            value_class=_Value,
        )


def test_projected_loader_requires_one_regular_json_file(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    with pytest.raises(jiang.JiangF638SFTError, match="exactly one regular .json"):
        jiang.load_projected_training_dataset(
            train_dir,
            cache_dir=None,
            expected_rows=1,
            dataset_class=_ProjectedDataset,
            features_class=_Features,
            value_class=_Value,
        )

    (train_dir / "one.json").write_text("[]\n", encoding="utf-8")
    (train_dir / "two.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(jiang.JiangF638SFTError, match="exactly one regular .json"):
        jiang.load_projected_training_dataset(
            train_dir,
            cache_dir=None,
            expected_rows=1,
            dataset_class=_ProjectedDataset,
            features_class=_Features,
            value_class=_Value,
        )


def test_projected_loader_detects_source_identity_change_during_stream(tmp_path):
    source = tmp_path / "train.json"
    source.write_text(json.dumps([_row("one"), _row("two")]), encoding="utf-8")
    observed = source.stat(follow_symlinks=False)
    identity = tuple(
        int(getattr(observed, field))
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )
    rows = jiang._projected_training_rows(
        str(source), expected_rows=2, expected_identity=identity
    )
    assert next(rows)["output"] == "one"
    source.chmod(0o440)
    with pytest.raises(jiang.JiangF638SFTError, match="changed during projection"):
        list(rows)


def test_projected_loader_preserves_formatted_training_semantics(tmp_path):
    train_dir = tmp_path / "op" / "train"
    train_dir.mkdir(parents=True)
    rows = [
        {**_row("answer one"), "instance_json": [[[0, 0]], [1], 3.0]},
        {**_row("answer two"), "instance_json": [[[0, 0]], [1], 4.0]},
    ]
    (train_dir / "train_op.json").write_text(json.dumps(rows), encoding="utf-8")
    projected = jiang.load_projected_training_dataset(
        train_dir,
        cache_dir=None,
        expected_rows=2,
        dataset_class=_ProjectedDataset,
        features_class=_Features,
        value_class=_Value,
    )
    direct = _Dataset(
        [{field: row[field] for field in jiang.REQUIRED_TEXT_FIELDS} for row in rows]
    )
    tokenizer = _CharacterTokenizer()
    projected_formatted, projected_audit = jiang.prepare_training_dataset(
        projected, tokenizer, expected_rows=2, max_seq_length=2_000
    )
    direct_formatted, direct_audit = jiang.prepare_training_dataset(
        direct, tokenizer, expected_rows=2, max_seq_length=2_000
    )
    assert [row["text"] for row in projected_formatted.rows] == [
        row["text"] for row in direct_formatted.rows
    ]
    assert projected_audit == direct_audit


def test_training_dataset_has_one_explicit_shuffle_then_exact_format_and_audit():
    dataset = _Dataset([_row("answer one"), _row("answer two")])
    tokenizer = _CharacterTokenizer()
    formatted, audit = jiang.prepare_training_dataset(
        dataset,
        tokenizer,
        expected_rows=2,
        max_seq_length=2_000,
    )
    assert formatted is dataset
    assert dataset.events == [
        ("shuffle", 42),
        ("map", True, "Formatting exact Jiang-f638 SFT text"),
    ]
    assert dataset.rows[0]["text"] == jiang.build_jiang_f638_sft_text(
        "Solve exactly.",
        "Node 0; Node 1",
        "answer two",
        eos_token=tokenizer.eos_token,
    )
    assert audit["rows_checked"] == 2
    assert audit["tokenized_header_mismatches"] == 0
    assert audit["truncated_rows"] == 0
    assert audit["formatted_dataset_fingerprint"] == "fake-fingerprint"


def test_all_row_marker_audit_fails_on_ambiguity_and_truncation():
    tokenizer = _CharacterTokenizer()
    duplicate = _Dataset(
        [
            {
                **_row(),
                "text": f"prefix {jiang.RESPONSE_MARKER} x {jiang.RESPONSE_MARKER}<eos>",
            }
        ]
    )
    with pytest.raises(jiang.JiangF638SFTError, match="raw response header is ambiguous"):
        jiang.audit_formatted_dataset(duplicate, tokenizer, expected_rows=1)

    valid = _Dataset(
        [
            {
                **_row(),
                "text": f"prefix {jiang.RESPONSE_MARKER} completion<eos>",
            }
        ]
    )
    with pytest.raises(jiang.JiangF638SFTError, match="would truncate"):
        jiang.audit_formatted_dataset(
            valid,
            tokenizer,
            expected_rows=1,
            max_seq_length=10,
        )


def test_strict_collator_masks_through_marker_and_never_falls_back():
    tokenizer = _CharacterTokenizer()
    marker = tokenizer.encode(jiang.RESPONSE_MARKER, add_special_tokens=False)
    prefix = tokenizer.encode("prompt ", add_special_tokens=False)
    completion = tokenizer.encode("answer<eos>", add_special_tokens=False)
    collator = jiang.build_strict_response_collator(
        tokenizer,
        base_collator_class=_FakeLanguageModelCollator,
    )
    batch = collator.torch_call([{"input_ids": prefix + marker + completion}])
    response_start = len(prefix) + len(marker)
    assert batch["labels"].rows[0][:response_start] == [-100] * response_start
    assert batch["labels"].rows[0][response_start:] == completion

    with pytest.raises(jiang.JiangF638SFTError, match="0 contextual response headers"):
        collator.torch_call([{"input_ids": prefix + completion}])
    with pytest.raises(jiang.JiangF638SFTError, match="2 contextual response headers"):
        collator.torch_call([{"input_ids": marker + marker + completion}])


def test_real_qwen_boundary_supervises_colon_output_crossing_token():
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "local_tsp_sft500k_seed42_instruct"
        / "tokenizer.json"
    )
    if not tokenizer_path.is_file():
        pytest.skip("local Qwen tokenizer is unavailable")
    backend = tokenizers.Tokenizer.from_file(str(tokenizer_path))

    class Wrapper:
        def encode(self, text, add_special_tokens=False):
            return backend.encode(text, add_special_tokens=add_special_tokens).ids

    tokenizer = Wrapper()
    text = jiang.build_jiang_f638_sft_text(
        "Solve.",
        "Nodes: [0, 1]",
        "Set: [0]",
        eos_token="<|endoftext|>",
    )
    input_ids = backend.encode(text, add_special_tokens=True).ids
    header_ids = jiang.jiang_f638_response_header_token_ids(tokenizer)
    supervised_start = jiang.locate_jiang_f638_supervised_start(
        text,
        input_ids,
        header_ids,
    )
    assert input_ids[supervised_start] == 15606
    assert backend.decode([input_ids[supervised_start]], skip_special_tokens=False) == ":Set"

    collator = jiang.build_strict_response_collator(
        tokenizer,
        base_collator_class=_FakeLanguageModelCollator,
    )
    batch = collator.torch_call([{"input_ids": input_ids}])
    assert batch["labels"].rows[0][:supervised_start] == [-100] * supervised_start
    assert batch["labels"].rows[0][supervised_start] == 15606


def test_fresh_output_rejects_training_artifacts_but_allows_read_only_contract(tmp_path):
    absent = tmp_path / "absent"
    assert jiang.assert_fresh_output(absent) == absent.absolute()

    contracted = tmp_path / "contracted"
    contracted.mkdir()
    contract = contracted / "base_sft_run_contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    contract.chmod(0o440)
    assert jiang.assert_fresh_output(contracted) == contracted.resolve()

    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "checkpoint-1").mkdir()
    with pytest.raises(jiang.JiangF638SFTError, match="refuses existing run artifacts"):
        jiang.assert_fresh_output(stale)


def test_chunk_execution_accepts_only_exact_checkpoint_prefix(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "hyperparams.json").write_text("{}\n", encoding="utf-8")
    checkpoint_5000 = _chunk_checkpoint(run, 5_000)
    checkpoint_10000 = _chunk_checkpoint(run, 10_000)

    output, resume, resume_step = jiang.resolve_chunk_execution(
        run,
        stop_after_step=15_000,
        resume_from_checkpoint=checkpoint_10000,
    )
    assert output == run.resolve()
    assert resume == checkpoint_10000.resolve()
    assert resume_step == 10_000

    with pytest.raises(jiang.JiangF638SFTError, match="exact saved prefix"):
        jiang.resolve_chunk_execution(
            run,
            stop_after_step=10_000,
            resume_from_checkpoint=checkpoint_5000,
        )


def test_chunk_execution_fails_closed_on_invalid_boundaries_and_resume_state(tmp_path):
    with pytest.raises(jiang.JiangF638SFTError, match="--stop-after-step"):
        jiang.resolve_chunk_execution(
            tmp_path / "fresh",
            stop_after_step=7_500,
            resume_from_checkpoint=None,
        )

    run = tmp_path / "run"
    run.mkdir()
    (run / "hyperparams.json").write_text("{}\n", encoding="utf-8")
    checkpoint = _chunk_checkpoint(run, 5_000)
    with pytest.raises(jiang.JiangF638SFTError, match="strictly before"):
        jiang.resolve_chunk_execution(
            run,
            stop_after_step=5_000,
            resume_from_checkpoint=checkpoint,
        )

    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(jiang.JiangF638SFTError, match="incomplete"):
        jiang.resolve_chunk_execution(
            run,
            stop_after_step=10_000,
            resume_from_checkpoint=checkpoint,
        )


def test_resume_checkpoint_must_be_canonical_and_inside_the_same_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "hyperparams.json").write_text("{}\n", encoding="utf-8")
    checkpoint = _chunk_checkpoint(run, 5_000)
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(
        jiang.JiangF638SFTError,
        match="invalid directory name|regular directory",
    ):
        jiang.resolve_chunk_execution(
            run,
            stop_after_step=10_000,
            resume_from_checkpoint=alias,
        )

    other = tmp_path / "other"
    other.mkdir()
    outside = _chunk_checkpoint(other, 5_000)
    with pytest.raises(jiang.JiangF638SFTError, match="outside the requested run"):
        jiang.resolve_chunk_execution(
            run,
            stop_after_step=10_000,
            resume_from_checkpoint=outside,
        )


def test_training_source_accepts_corpus_root_or_direct_train_directory(tmp_path):
    nested = tmp_path / "corpus" / "tsp" / "train"
    nested.mkdir(parents=True)
    (nested / "train.json").write_text("[]\n", encoding="utf-8")
    root, source = jiang.resolve_training_source(tmp_path / "corpus", "tsp")
    assert root == (tmp_path / "corpus").resolve()
    assert source == nested.resolve()

    direct = tmp_path / "direct" / "train"
    direct.mkdir(parents=True)
    (direct / "train.json").write_text("[]\n", encoding="utf-8")
    root, source = jiang.resolve_training_source(direct, "tsp")
    assert root == source == direct.resolve()


def test_periodic_checkpoint_callback_uses_existing_fidelity_writer(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "checkpoint-5000"
    checkpoint.mkdir(parents=True)
    calls = []

    class Callback:
        pass

    trainer = SimpleNamespace(model="live-model", accelerator=None)

    def auditor(model, path, **kwargs):
        calls.append((model, Path(path), kwargs))
        return {"status": "ok"}

    callback = jiang.build_checkpoint_fidelity_callback(
        trainer,
        trainer_callback_class=Callback,
        auditor=auditor,
    )
    control = object()
    returned = callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "checkpoints")),
        SimpleNamespace(global_step=5000, is_world_process_zero=True),
        control,
    )
    assert returned is control
    assert calls == [
        (
            "live-model",
            checkpoint,
            {"require_qwen25_7b_publication": True},
        )
    ]

    terminal = tmp_path / "checkpoints" / "checkpoint-31250"
    terminal.mkdir()
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "checkpoints")),
        SimpleNamespace(global_step=31_250, is_world_process_zero=True),
        control,
    )
    assert calls[-1][1] == terminal


def test_chunk_stop_callback_stops_only_after_complete_target_save(tmp_path):
    class Callback:
        pass

    callback = jiang.build_chunk_stop_callback(
        10_000,
        trainer_callback_class=Callback,
    )
    control = SimpleNamespace(should_training_stop=False)
    returned = callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "checkpoints")),
        SimpleNamespace(global_step=5_000, max_steps=31_250),
        control,
    )
    assert returned is control
    assert control.should_training_stop is False
    assert callback.saved_target is False

    _chunk_checkpoint(tmp_path, 10_000)
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "checkpoints")),
        SimpleNamespace(global_step=10_000, max_steps=31_250),
        control,
    )
    assert control.should_training_stop is True
    assert callback.saved_target is True


def test_chunk_stop_callback_rejects_incomplete_save_and_horizon_drift(tmp_path):
    class Callback:
        pass

    callback = jiang.build_chunk_stop_callback(
        5_000,
        trainer_callback_class=Callback,
    )
    control = SimpleNamespace(should_training_stop=False)
    incomplete = tmp_path / "checkpoints" / "checkpoint-5000"
    incomplete.mkdir(parents=True)
    with pytest.raises(jiang.JiangF638SFTError, match="incomplete"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path / "checkpoints")),
            SimpleNamespace(global_step=5_000, max_steps=31_250),
            control,
        )
    assert control.should_training_stop is False

    with pytest.raises(jiang.JiangF638SFTError, match="max_steps"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path / "checkpoints")),
            SimpleNamespace(global_step=1, max_steps=5_000),
            control,
        )


def test_run_uses_fresh_or_exact_resume_call_and_released_unsloth_sft_wrapper():
    tree = ast.parse(inspect.getsource(jiang.run))
    train_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
    ]
    assert len(train_calls) == 2
    fresh_calls = [call for call in train_calls if not call.args and not call.keywords]
    resume_calls = [
        call
        for call in train_calls
        if [keyword.arg for keyword in call.keywords] == ["resume_from_checkpoint"]
    ]
    assert len(fresh_calls) == 1
    assert len(resume_calls) == 1
    resume_value = resume_calls[0].keywords[0].value
    assert isinstance(resume_value, ast.Call)
    assert isinstance(resume_value.func, ast.Name)
    assert resume_value.func.id == "str"

    trainer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SFTTrainer"
    ]
    assert len(trainer_calls) == 1
    keywords = {keyword.arg for keyword in trainer_calls[0].keywords}
    assert {
        "tokenizer",
        "data_collator",
        "train_dataset",
        "args",
        "dataset_text_field",
        "max_seq_length",
        "dataset_num_proc",
        "packing",
    } <= keywords
    assert "processing_class" not in keywords

    sft_config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrainingArguments"
    ]
    assert len(sft_config_calls) == 1
    config_keywords = {keyword.arg for keyword in sft_config_calls[0].keywords}
    assert "logging_dir" in config_keywords
    assert not {"dataset_text_field", "max_seq_length", "dataset_num_proc", "packing"} & (
        config_keywords
    )
    assert not {"per_device_eval_batch_size", "eval_steps"} & config_keywords

    peft_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_peft_model"
    ]
    peft_keywords = {keyword.arg for keyword in peft_calls[0].keywords}
    assert "lora_dropout" in peft_keywords


def test_cli_exposes_only_paths_reporting_reload_and_exact_chunk_switches():
    actions = {action.dest for action in jiang.build_parser()._actions}
    assert actions == {
        "help",
        "task",
        "model_path",
        "data_root",
        "output_dir",
        "tokenizer_path",
        "cache_dir",
        "report_to",
        "post_save_reload_check",
        "stop_after_step",
        "resume_from_checkpoint",
    }
    parser = jiang.build_parser()
    stop_action = next(action for action in parser._actions if action.dest == "stop_after_step")
    assert tuple(stop_action.choices) == jiang.ALLOWED_CHUNK_STOP_STEPS
    assert stop_action.default == 31_250


def test_wandb_identity_is_worker_controlled_and_requires_iclr() -> None:
    expected = {
        "WANDB_PROJECT": "clasp_sft_iclr",
        "WANDB_RUN_GROUP": "campaign_iclr-tsp-seed42",
        "WANDB_NAME": "campaign_iclr-tsp-job123",
    }
    assert jiang._iclr_wandb_identity(expected) == expected
    for key in expected:
        broken = dict(expected)
        broken[key] = "missing-tag"
        with pytest.raises(jiang.JiangF638SFTError, match=key):
            jiang._iclr_wandb_identity(broken)


def test_hyperparams_records_exact_recipe_and_both_corrections(tmp_path):
    args = argparse.Namespace(
        task="tsp",
        model_path=tmp_path / "model",
        tokenizer_path=None,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "run",
        cache_dir=None,
        report_to="none",
        post_save_reload_check=True,
    )

    class TrainingArgs:
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 4
        warmup_steps = 20
        num_train_epochs = 1.0
        learning_rate = 0.0002
        bf16 = True
        logging_steps = 1
        optim = "adamw_8bit"
        weight_decay = 0.01
        lr_scheduler_type = "linear"
        seed = 42
        eval_strategy = "no"
        metric_for_best_model = "eval_loss"
        save_total_limit = 50
        save_steps = 5_000
        dataset_text_field = "text"
        max_seq_length = 20_000
        dataset_num_proc = 16
        packing = False
        per_device_eval_batch_size = 8
        eval_steps = None

        def to_dict(self):
            return {"seed": 42, "output_dir": str(tmp_path / "run" / "checkpoints")}

    payload = jiang._build_hyperparams_payload(
        args=args,
        model_path=tmp_path / "model",
        tokenizer_path=tmp_path / "model",
        data_root=tmp_path / "data",
        train_source=tmp_path / "data" / "tsp" / "train",
        output_dir=tmp_path / "run",
        runtime_versions=jiang.PINNED_RUNTIME_VERSIONS,
        hardware={"world_size": 1},
        base_snapshot={"sha256": "a" * 64},
        marker_audit={"rows_checked": 500_000},
        policy_preflight={"status": "exact"},
        training_args=TrainingArgs(),
    )
    stage = payload["stage_hyperparameters"]
    assert stage["lora_bias"] == "all"
    assert all(
        stage[key] == value for key, value in jiang.MATCHED_SFT_STAGE_HPARAMETERS.items()
    )
    assert stage["per_device_eval_batch_size"] == 8
    assert stage["checkpoint_fidelity_mode"] == "publication"
    assert stage["min_eval_samples_per_task"] == 0
    assert stage["numeric_token_loss_weight"] == 1.0
    assert stage["eval_steps"] is None
    assert stage["expected_terminal_global_step"] == 31_250
    assert stage["single_explicit_dataset_shuffle_seed"] == 42
    assert stage["response_marker_fallback"] is None
    assert stage["response_header"] == "    ### Response"
    assert payload["artifacts"]["checkpoint_dir"].endswith("/run/checkpoints")
    assert payload["intentional_corrections"] == {
        "fresh_train_call": "trainer.train()",
        "released_fresh_call": "trainer.train(resume_from_checkpoint=True)",
        "bias_policy": "all",
        "released_bias_policy": "lora_only",
        "bias_trainability_restore": "explicitly re-enable exact 84 Q/K/V base biases",
        "response_boundary": "contextual indented header without colon",
        "released_response_boundary": "standalone ### Response: token IDs (never match)",
        "trainer_api": (
            "released legacy SFTTrainer kwargs through Unsloth compatibility wrapper"
        ),
        "effective_trainer_config_audit": "required before first optimizer step",
        "reason": (
            "fresh-start/API correctness, explicit QKV-bias training and serialization, "
            "and intended response-only supervision"
        ),
    }
    assert payload["effective_sft_config"]["max_seq_length"] == 20_000
    assert payload["effective_sft_config"]["eval_steps"] is None
    assert payload["args"]["dtype"] == "bfloat16"
    assert payload["args"]["eval_steps"] is None


def test_resume_reopens_identical_hyperparams_without_replacing_them(tmp_path):
    path = tmp_path / "hyperparams.json"
    payload = {"stage": "sft", "fixed_horizon": 31_250}
    jiang.write_or_validate_hyperparams(path, payload, resume=False)
    original = path.read_bytes()
    jiang.write_or_validate_hyperparams(path, payload, resume=True)
    assert path.read_bytes() == original

    with pytest.raises(jiang.JiangF638SFTError, match="immutable fresh-chunk"):
        jiang.write_or_validate_hyperparams(
            path,
            {**payload, "fixed_horizon": 5_000},
            resume=True,
        )
    assert path.read_bytes() == original


def test_effective_sft_config_audit_rejects_wrapper_default_drift() -> None:
    valid = SimpleNamespace(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=1.0,
        learning_rate=0.0002,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        eval_strategy="no",
        metric_for_best_model="eval_loss",
        save_total_limit=50,
        save_steps=5_000,
        dataset_text_field="text",
        max_seq_length=20_000,
        dataset_num_proc=16,
        packing=False,
        per_device_eval_batch_size=8,
        eval_steps=None,
    )
    assert jiang.audit_effective_sft_config(valid)["dataset_num_proc"] == 16
    valid.max_seq_length = 2_048
    with pytest.raises(jiang.JiangF638SFTError, match="unexpected effective defaults"):
        jiang.audit_effective_sft_config(valid)


def test_terminal_status_precedes_optional_reload_for_recoverable_certification() -> None:
    source = inspect.getsource(jiang.run)
    status_offset = source.index("write_final_stage_status(output_dir, trainer_state)")
    reload_offset = source.index("_run_reload_audit(model_path=model_path, output_dir=output_dir)")
    assert status_offset < reload_offset


def test_nonterminal_chunk_returns_before_any_terminal_root_export() -> None:
    source = inspect.getsource(jiang.run)
    branch = source.index("if stop_after_step < TARGET_STEP:")
    terminal_save = source.index("trainer.save_model(str(output_dir))")
    terminal_status = source.index("write_final_stage_status(output_dir, trainer_state)")
    terminal_reload = source.index(
        "_run_reload_audit(model_path=model_path, output_dir=output_dir)"
    )
    assert branch < terminal_save < terminal_status < terminal_reload
    branch_source = source[branch:terminal_save]
    assert "return output_dir" in branch_source
    assert "save_model" not in branch_source
    assert "write_final_stage_status" not in branch_source
    assert "_run_reload_audit" not in branch_source


def test_trainable_bias_closure_is_rechecked_after_trainer_construction() -> None:
    source = inspect.getsource(jiang.run)
    trainer_offset = source.index("trainer = SFTTrainer(")
    post_closure_offset = source.index(
        "post_trainer_closure = audit_trainable_state_closure"
    )
    train_offset = source.index("trainer.train()")
    assert trainer_offset < post_closure_offset < train_offset


def test_final_stage_status_binds_step_31250_to_checkpoint_31250(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "checkpoint-31250"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "checkpoint_fidelity.json",
        "trainer_state.json",
        "adapter_model.safetensors",
    ):
        (checkpoint / name).write_bytes(b"x")
    for name in (
        "adapter_config.json",
        "checkpoint_fidelity.json",
        "adapter_model.safetensors",
    ):
        (run_dir / name).write_bytes(b"x")

    payload = jiang.write_final_stage_status(
        run_dir,
        SimpleNamespace(global_step=31_250, max_steps=31_250, epoch=1.0),
    )
    assert payload == {
        "stage": "sft",
        "completed": True,
        "global_step": 31_250,
        "max_steps": 31_250,
        "epoch": 1.0,
        "has_saved_model": True,
        "latest_checkpoint_dir": str(checkpoint.resolve()),
    }
    assert (run_dir / "stage_status.json").is_file()
