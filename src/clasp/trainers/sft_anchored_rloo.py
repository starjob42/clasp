"""Prospective SFT-anchored B9 RLOO pilot on the pinned TRL 0.24 loss.

The original SFT policy is an independent frozen model. Stock TRL's implicit
PEFT adapter-disabled reference is deliberately never used. Existing RLOO and
quality trainers remain unchanged. Beta zero and control quality credit retain
the original detached RLOO coefficients and fixed-horizon loss exactly.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import trl
from trl import GRPOTrainer

from clasp.models.external_baseline_state import (
    _write_json_atomic, restore_rank_state, save_rank_state, semantic_state_sha256,
)
from clasp.rewards.feasible_quality import (
    decompose_advantages, make_advantage_contract, validate_advantage_contract,
    verified_reward_metadata,
)

PROTOCOL = "sft-anchored-rloo-pilot-20261013-v1"
CONTRACT_FILENAME = "sft_anchored_rloo_contract.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _storage_ids(model):
    return {(str(tensor.device), tensor.untyped_storage().data_ptr())
            for tensor in model.state_dict().values() if tensor.numel()}


class _FrozenSFTReferenceMixin:
    """Shared live reference checks without changing either policy estimator."""

    @staticmethod
    def _validate_independent_reference(policy, reference):
        if policy is reference:
            raise ValueError("Actor and reference must be distinct independent model objects")
        if any(parameter.requires_grad for parameter in reference.parameters()):
            raise ValueError("Original SFT reference must be fully frozen before trainer construction")
        if _storage_ids(policy) & _storage_ids(reference):
            raise ValueError("Actor and reference must not share tensor storage")

    def _gather_objects(self, value):
        if int(self.accelerator.num_processes) == 1:
            return [value]
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed audit requires an initialized process group")
        gathered = [None] * int(self.accelerator.num_processes)
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def _consensus(self, value, label):
        values = self._gather_objects(value)
        if any(item != values[0] for item in values):
            raise RuntimeError(f"SFT-anchored {label} differs across ranks")

    def _collective_check(self, fn, label):
        result, error = None, None
        try:
            result = fn()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        failures = self._gather_objects(error)
        if any(item is not None for item in failures):
            raise RuntimeError(f"SFT-anchored {label} failed across ranks: {failures}")
        return result

    def _reference_version_inventory(self):
        reference = self.accelerator.unwrap_model(self.ref_model)
        return {name: (id(value), value._version, str(value.dtype), tuple(value.shape),
                       str(value.device), value.untyped_storage().data_ptr())
                for name, value in (*reference.named_parameters(), *reference.named_buffers())}

    def reference_state_sha256(self):
        """Full tensor-byte fingerprint, independent of device and object IDs."""
        if self.ref_model is None:
            return None
        return semantic_state_sha256(self.accelerator.unwrap_model(self.ref_model).state_dict())

    def _assert_reference_integrity(self, *, full=False):
        if not self.beta:
            if self.ref_model is not None:
                raise RuntimeError("Beta-zero control acquired a reference model")
            return
        if self.ref_model is None:
            raise RuntimeError("Original SFT reference is missing; adapter-disabled fallback forbidden")
        reference = self.accelerator.unwrap_model(self.ref_model)
        if reference.training or any(p.requires_grad or p.grad is not None for p in reference.parameters()):
            raise RuntimeError("Original SFT reference is no longer frozen in eval mode")
        if self._reference_version_inventory() != self._reference_versions:
            raise RuntimeError("Original SFT reference tensors changed after initialization")
        self._validate_independent_reference(self.accelerator.unwrap_model(self.model), reference)
        if self.optimizer is not None:
            owned = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            if owned & {id(p) for p in reference.parameters()}:
                raise RuntimeError("Optimizer includes frozen SFT reference parameters")
        if full and self.reference_state_sha256() != self._reference_initial_sha256:
            raise RuntimeError("Frozen SFT reference tensor-byte fingerprint changed")

    def _audit_initial_pair(self):
        """Probe both forwards under the same actual attention/AMP/cache inputs.

        During Trainer construction the actor is not yet Accelerate-prepared.
        We compare float32 output projections under shared autocast here, then
        repeat on the actual prepared actor at the first native rollout.
        """
        policy = self.model
        reference = self.ref_model
        raw_policy = self.accelerator.unwrap_model(policy)
        raw_reference = self.accelerator.unwrap_model(reference)
        policy_state, ref_state = raw_policy.state_dict(), raw_reference.state_dict()
        if (set(policy_state) != set(ref_state) or any(
                a.dtype != ref_state[key].dtype or not torch.equal(a, ref_state[key])
                for key, a in policy_state.items())):
            raise ValueError("Initial actor differs from the original SFT reference state")
        attention = [getattr(model.config, "_attn_implementation", None)
                     for model in (raw_policy, raw_reference)]
        if attention[0] != attention[1]:
            raise ValueError("Actor and reference attention implementations differ")
        vocab = int(raw_policy.config.vocab_size)
        ids = torch.tensor([[min(i, vocab - 1) for i in (3, 4, 7, 2)],
                            [min(i, vocab - 1) for i in (3, 7, 4, 2)]],
                           device=self.accelerator.device, dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[1, 0] = 0
        was_training = policy.training
        try:
            policy.eval()
            reference.eval()
            with torch.no_grad(), self.accelerator.autocast():
                p = policy(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                r = reference(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            if not torch.isfinite(p).all() or not torch.equal(p, r):
                delta = (p - r).abs().max().item()
                raise ValueError(f"Prepared initial actor/reference logits differ: max_error={delta}")
            return {"protocol": "same-input-attention-autocast-eval-cache-probe-v1",
                    "bitwise_equal": True, "max_absolute_error": 0.0,
                    "attention_implementation": attention[0], "use_cache": False,
                    "autocast": str(self.accelerator.mixed_precision),
                    "input_sha256": semantic_state_sha256({"ids": ids, "mask": mask}),
                    "logits_sha256": semantic_state_sha256(p)}
        finally:
            policy.train(was_training)
            reference.eval()

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        # Both reference collection and actor optimization enter the same AMP
        # context. TRL already supplies identical attention masks/use_cache=False.
        with self.accelerator.autocast():
            return super()._get_per_token_logps_and_entropies(model, *args, **kwargs)


class SFTAnchoredRLOOTrainer(_FrozenSFTReferenceMixin, GRPOTrainer):
    """B9 detached RLOO, optional feasible-cardinal credit, and fixed SFT KL.

    ``reference_identity`` binds the independently loaded immutable original
    SFT source: ``source_kind='original_sft'``, ``source_sha256=<64 hex chars>``,
    and ``immutable=True``. Additional provenance fields are checkpoint-pinned.
    The caller verifies the source artifact; this trainer verifies live state,
    independent storage, preparation, optimizer exclusion, and no reference
    drift. The nonempty ``recipe_contract`` pins data, initialization and budget.
    """

    def __init__(self, *positional, reference_model=None, reference_identity=None,
                 advantage_contract=None, reward_contract=None, recipe_contract=None,
                 **kwargs):
        if positional or "args" not in kwargs or "model" not in kwargs:
            raise ValueError("Pass instantiated model and GRPOConfig as keyword arguments")
        if trl.__version__ != "0.24.0":
            raise RuntimeError("This prospective trainer requires audited TRL 0.24.0")
        arguments = kwargs["args"]
        beta = arguments.beta
        if isinstance(beta, bool) or beta not in (0.0, 0.05):
            raise ValueError("Pilot beta must be 0.0 or 0.05")
        required = {
            "num_generations": 9, "scale_rewards": "none", "loss_type": "dr_grpo",
            "num_iterations": 1, "importance_sampling_level": "token",
            "use_vllm": False, "use_liger_loss": False, "sync_ref_model": False,
            "mask_truncated_completions": False, "top_entropy_quantile": 1.0,
        }
        for name, expected in required.items():
            if getattr(arguments, name, None) != expected:
                raise ValueError(f"SFT-anchored attribution requires {name}={expected!r}")
        if arguments.steps_per_generation != arguments.gradient_accumulation_steps:
            raise ValueError("Require one full rollout buffer per optimizer step")
        if arguments.eval_strategy != "no":
            raise ValueError("Use separate fixed-decoder policy evaluation")
        if not isinstance(recipe_contract, dict) or not recipe_contract:
            raise ValueError("A nonempty JSON recipe_contract is required")
        self.recipe_contract = json.loads(_canonical(recipe_contract))
        self.advantage_contract = validate_advantage_contract(
            advantage_contract or make_advantage_contract("control"))
        if self.advantage_contract["method"] not in ("control", "cardinal"):
            raise ValueError("This pilot compares only control and cardinal quality credit")
        self.reward_contract = None
        if reward_contract is not None:
            from clasp.baselines.quality_signal_dev import validate_reward_contract
            self.reward_contract = validate_reward_contract(reward_contract)
            if self.reward_contract["mode"] != "reference_sigmoid_v2":
                raise ValueError("Pilot reward must be reference_sigmoid_v2")
        if self.advantage_contract["weight"] and self.reward_contract is None:
            raise ValueError("Cardinal quality requires a calibrated reward contract")
        self.reference_identity = None
        if beta:
            if reference_model is None:
                raise ValueError("Positive beta requires an independent frozen original SFT reference")
            if (not isinstance(reference_identity, dict)
                    or reference_identity.get("source_kind") != "original_sft"
                    or reference_identity.get("immutable") is not True
                    or not isinstance(reference_identity.get("source_sha256"), str)
                    or len(reference_identity["source_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in reference_identity["source_sha256"])):
                raise ValueError("Reference must bind an immutable original_sft source SHA256")
            self.reference_identity = json.loads(_canonical(reference_identity))
            self._validate_independent_reference(kwargs["model"], reference_model)
        elif reference_model is not None or reference_identity is not None:
            raise ValueError("Beta-zero control requires no reference model or identity")

        # Parent initialization otherwise reloads _name_or_path or uses the
        # adapter-disabled base. A copied beta-zero initialization configuration
        # suppresses both branches without altering the caller's final recipe.
        initialization_args = copy.copy(arguments)
        initialization_args.beta = 0.0
        kwargs["args"] = initialization_args
        super().__init__(**kwargs)
        self.args = arguments
        self.beta = float(beta)
        if self.reward_weights.numel() != 1 or self.reward_weights.item() != 1.0:
            raise ValueError("Require one unscaled verifier reward")
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("This pilot supports native CPU/CUDA and DDP; sharded state needs a separate audit")
        self.ref_model = None
        self._reference_versions = None
        self._reference_initial_sha256 = None
        self.initial_reference_audit = None
        self.prepared_reference_audit = None
        self._reference_prepared_checked = False
        self._resuming = False
        if reference_model is not None:
            # Match dropout configuration while preserving actor train mode
            # and keeping the reference in eval mode throughout optimization.
            if arguments.disable_dropout:
                from trl.trainer.grpo_trainer import disable_dropout_in_model
                disable_dropout_in_model(reference_model)
            reference_model.eval()
            self.ref_model = self.accelerator.prepare_model(reference_model, evaluation_mode=True)
            self._reference_initial_sha256 = self.reference_state_sha256()
            self._reference_versions = self._reference_version_inventory()
            self.initial_reference_audit = self._audit_initial_pair()
        self._generation_active = False
        self._reward_buffer = None
        self._metadata_buffer = None
        self._prompt_rows = None
        self._contract_json = _canonical(self._recipe())
        self._contract_sha256 = _digest(self._recipe())
        self._consensus(self._contract_json, "recipe")
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

    def _recipe(self):
        return {"protocol": PROTOCOL, "trl_version": trl.__version__,
                "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_loss_source_sha256": hashlib.sha256(
                    inspect.getsource(GRPOTrainer._compute_loss).encode()).hexdigest(),
                "beta": self.beta, "quality_weight": self.advantage_contract["weight"],
                "num_generations": self.num_generations,
                "max_completion_length": self.max_completion_length,
                "loss_type": self.loss_type, "scale_rewards": self.scale_rewards,
                "num_iterations": self.num_iterations,
                "steps_per_generation": self.args.steps_per_generation,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "disable_dropout": self.args.disable_dropout,
                "max_steps": self.args.max_steps,
                "training_arguments": _training_arguments(self.args),
                "reference_identity": self.reference_identity,
                "reference_state_sha256": self._reference_initial_sha256,
                "advantage_contract": self.advantage_contract,
                "reward_contract": self.reward_contract,
                "recipe_contract": self.recipe_contract}

    def _assert_contract(self):
        if _canonical(self._recipe()) != self._contract_json or self.args.beta != self.beta:
            raise ValueError("Active SFT-anchored recipe contract changed")
        self._assert_reference_integrity()

    def _input_metadata(self, inputs):
        def collect():
            if not isinstance(inputs, list) or not inputs or any(not isinstance(row, dict) for row in inputs):
                raise ValueError("Require nonempty completion-aligned input dictionaries")
            return [hashlib.sha256(json.dumps(
                [row.get("task_type"), row.get("instance_json"), row.get("input"), row.get("prompt")],
                sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest() for row in inputs]
        local = self._collective_check(collect, "prompt metadata")
        parts = self._gather_objects(local)
        if any(len(part) != len(local) for part in parts):
            raise RuntimeError("Rank rollout sizes differ")
        rows = [row for part in parts for row in part]
        if len(rows) % self.num_generations:
            raise RuntimeError("Global rollout contains incomplete B9 groups")
        for start in range(0, len(rows), self.num_generations):
            if any(row != rows[start] for row in rows[start:start + self.num_generations]):
                raise RuntimeError("Native sampler did not preserve sibling prompt identity")
        return rows

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if not self._generation_active or self._reward_buffer is not None:
            raise RuntimeError("Require exactly one reward calculation per rollout")
        metadata = None
        if self.reward_contract is not None:
            local = self._collective_check(lambda: verified_reward_metadata(
                self.reward_contract["task"], prompts, completions, inputs), "quality verification")
            parts = self._gather_objects(local)
            if any(len(part) != len(inputs) for part in parts):
                raise RuntimeError("Rank quality metadata sizes differ")
            metadata = [row for part in parts for row in part]
            if [row["prompt_key"] for row in metadata] != self._prompt_rows:
                raise RuntimeError("Quality metadata differs from native prompt order")
            for start in range(0, len(metadata), self.num_generations):
                group = metadata[start:start + self.num_generations]
                if any(row["reference"] != group[0]["reference"] for row in group):
                    raise RuntimeError("Sibling training references differ")
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if tuple(rewards.shape) != (len(self._prompt_rows), 1) or not torch.isfinite(rewards).all():
            raise RuntimeError("Missing or nonfinite globally gathered scalar verifier rewards")
        if metadata is not None:
            expected = torch.tensor([row["anchor_reward"] for row in metadata],
                                    dtype=rewards.dtype, device=rewards.device).reshape(-1, 1)
            if not torch.equal(rewards, expected):
                raise RuntimeError("Verified metadata differs from gathered anchor rewards")
        self._reward_buffer = rewards.detach().clone()
        self._metadata_buffer = metadata
        return rewards

    def _generate_and_score_completions(self, inputs):
        if not self.model.training or self._generation_active:
            raise RuntimeError("Use non-nested train-only rollout collection")
        self._collective_check(self._assert_contract, "active contract")
        if self.beta and not self._reference_prepared_checked:
            if not self._resuming:
                self.prepared_reference_audit = self._collective_check(
                    self._audit_initial_pair, "prepared initial actor/reference equality")
            self._reference_prepared_checked = True
        self._prompt_rows = self._input_metadata(inputs)
        self._reward_buffer = self._metadata_buffer = None
        self._generation_active = True
        try:
            output = super()._generate_and_score_completions(inputs)
            rewards = self._reward_buffer
            if rewards is None:
                raise RuntimeError("Missing scalar rewards from native rollout")
            grouped = rewards[:, 0].view(-1, self.num_generations)
            base = (grouped - (grouped.sum(1, keepdim=True) - grouped)
                    / (self.num_generations - 1)).reshape(-1).detach()
            auxiliary = torch.zeros_like(base)
            diagnostics = {}
            if self._metadata_buffer is not None:
                parts = decompose_advantages(base.cpu().tolist(), self._metadata_buffer,
                                             self.advantage_contract)
                auxiliary = torch.tensor(parts["quality"], dtype=base.dtype, device=base.device)
                diagnostics = parts["diagnostics"]
            weight = self.advantage_contract["weight"]
            combined = base if weight == 0.0 else base + weight * auxiliary
            if not torch.isfinite(combined).all():
                raise RuntimeError("Nonfinite SFT-anchored RLOO coefficients")
            self._consensus(combined.cpu().tolist(), "global final coefficients")
            start = int(self.accelerator.process_index) * len(inputs)
            output["advantages"] = combined[start:start + len(inputs)].detach()
            recorded, replacement = self._logs["advantages"], combined.cpu().tolist()
            for _ in range(min(len(recorded), len(replacement))):
                recorded.pop()
            recorded.extend(replacement)
            for name, value in {**diagnostics, "anchor/beta": self.beta,
                                "anchor/quality_weight": weight}.items():
                self._metrics["train"][name].append(float(value))
            if self.accelerator.is_main_process:
                record = {"global_step_before_update": int(self.state.global_step),
                          "recorded_at_unix": time.time(), "contract_sha256": self._contract_sha256,
                          "beta": self.beta, "quality_weight": weight,
                          "reference_state_sha256": self._reference_initial_sha256,
                          "base": base.cpu().tolist(), "quality": auxiliary.cpu().tolist(),
                          "combined": combined.cpu().tolist(), "metadata": self._metadata_buffer,
                          "initial_reference_audit": self.initial_reference_audit,
                          "prepared_reference_audit": self.prepared_reference_audit, **diagnostics}
                with (Path(self.args.output_dir) / "sft_anchored_rloo_batches.jsonl").open("a") as handle:
                    handle.write(_canonical(record) + "\n")
            return output
        finally:
            self._generation_active = False
            self._reward_buffer = self._metadata_buffer = self._prompt_rows = None

    def _compute_loss(self, model, inputs):
        def admit_forward():
            self._assert_contract()
            if inputs["advantages"].requires_grad:
                raise ValueError("RLOO advantages must be detached from the policy")
            if self.beta and (inputs.get("ref_per_token_logps") is None
                              or inputs["ref_per_token_logps"].requires_grad):
                raise ValueError("Positive beta requires detached frozen SFT reference scores")

        # Every rank reaches this collective before entering the native actor
        # forward. A rank-local reference or input failure must stop its peers
        # before their DDP forward/backward collectives can block.
        self._collective_check(admit_forward, "pre-forward RLOO loss admission")
        return super()._compute_loss(model, inputs)

    def _checkpoint_payload(self, step):
        self._assert_contract()
        self._assert_reference_integrity(full=True)
        return {"schema_version": 1, "global_step": int(step),
                "world_size": int(self.accelerator.num_processes),
                "contract_sha256": self._contract_sha256,
                "contract": json.loads(self._contract_json)}

    def _save_checkpoint(self, model, trial):
        if self._generation_active:
            raise RuntimeError("Cannot checkpoint an unfinished rollout")
        payload = self._collective_check(
            lambda: self._checkpoint_payload(self.state.global_step), "checkpoint contract")
        self._consensus(payload, "checkpoint contract")
        super()._save_checkpoint(model, trial)
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        self._collective_check(lambda: save_rank_state(
            directory, self.optimizer, self.lr_scheduler,
            rank=int(self.accelerator.process_index), world_size=int(self.accelerator.num_processes),
            step=int(self.state.global_step)), "rank-owned checkpoint state")
        if self.accelerator.is_main_process:
            _write_json_atomic(directory / CONTRACT_FILENAME, payload)
        self.accelerator.wait_for_everyone()

    def _validate_checkpoint(self, checkpoint):
        directory = Path(checkpoint)
        state = json.loads((directory / "trainer_state.json").read_text())
        step = state["global_step"]
        if type(step) is not int or not 0 < step <= self.args.max_steps:
            raise ValueError("Invalid SFT-anchored checkpoint step")
        saved = json.loads((directory / CONTRACT_FILENAME).read_text())
        if _canonical(saved) != _canonical(self._checkpoint_payload(step)):
            raise ValueError("Resume SFT-anchored recipe or frozen reference identity differs")
        return step

    def _load_optimizer_and_scheduler(self, checkpoint):
        self._collective_check(self._assert_contract, "resume active contract")
        if checkpoint:
            step = self._collective_check(lambda: self._validate_checkpoint(checkpoint),
                                          "resume recipe validation")
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            self._collective_check(lambda: restore_rank_state(
                checkpoint, self.optimizer, self.lr_scheduler,
                rank=int(self.accelerator.process_index), world_size=int(self.accelerator.num_processes),
                step=step), "rank-owned optimizer restoration")
            self._resuming = True
            self._reference_prepared_checked = True
        return result


def _training_arguments(arguments):
    """Portable numerical training recipe; output destinations may differ."""
    names = (
        "seed", "data_seed", "learning_rate", "weight_decay", "optim",
        "lr_scheduler_type", "warmup_steps", "warmup_ratio", "max_grad_norm",
        "per_device_train_batch_size", "gradient_checkpointing", "bf16", "fp16",
        "epsilon", "epsilon_high", "delta", "temperature", "top_p", "top_k",
        "min_p", "repetition_penalty", "max_prompt_length", "reward_weights",
        "importance_sampling_level", "top_entropy_quantile", "shuffle_dataset",
        "use_vllm", "use_liger_loss", "sync_ref_model", "mask_truncated_completions",
    )
    return {name: getattr(arguments, name, None) for name in names}


class SFTReferenceGRPOTrainer(_FrozenSFTReferenceMixin, GRPOTrainer):
    """Stock TRL credit and loss with an explicit independent frozen SFT anchor.

    This is the matched baseline reference plumbing. It supports FOARL's
    released multi-head reward, group standard deviation and response-length
    loss. It never substitutes RLOO/cardinal coefficients for stock GRPO credit.
    """

    def __init__(self, *positional, reference_model, reference_identity,
                 recipe_contract, **kwargs):
        if positional or "args" not in kwargs or "model" not in kwargs:
            raise ValueError("Pass instantiated model and GRPOConfig as keyword arguments")
        if trl.__version__ != "0.24.0":
            raise RuntimeError("Explicit SFT reference requires audited TRL 0.24.0")
        arguments = kwargs["args"]
        beta = arguments.beta
        if beta != 0.05 or isinstance(beta, bool):
            raise ValueError("This matched stock baseline fixes beta=0.05")
        if arguments.sync_ref_model or arguments.use_liger_loss:
            raise ValueError("A frozen explicit SFT reference forbids synchronization/Liger")
        if (not isinstance(reference_identity, dict)
                or reference_identity.get("source_kind") != "original_sft"
                or reference_identity.get("immutable") is not True
                or not isinstance(reference_identity.get("source_sha256"), str)
                or len(reference_identity["source_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in reference_identity["source_sha256"])):
            raise ValueError("Reference must bind an immutable original_sft source SHA256")
        if not isinstance(recipe_contract, dict) or not recipe_contract:
            raise ValueError("A nonempty recipe_contract is required")
        self.reference_identity = json.loads(_canonical(reference_identity))
        self.recipe_contract = json.loads(_canonical(recipe_contract))
        self._validate_independent_reference(kwargs["model"], reference_model)
        initialization_args = copy.copy(arguments)
        initialization_args.beta = 0.0
        kwargs["args"] = initialization_args
        super().__init__(**kwargs)
        self.args = arguments
        self.beta = float(beta)
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("Explicit SFT baseline supports CPU/CUDA and DDP; sharding needs a separate audit")
        if arguments.disable_dropout:
            from trl.trainer.grpo_trainer import disable_dropout_in_model
            disable_dropout_in_model(reference_model)
        reference_model.eval()
        self.ref_model = self.accelerator.prepare_model(reference_model, evaluation_mode=True)
        self._reference_initial_sha256 = self.reference_state_sha256()
        self._reference_versions = self._reference_version_inventory()
        self.initial_reference_audit = self._audit_initial_pair()
        self.prepared_reference_audit = None
        self._reference_prepared_checked = False
        self._resuming = False
        self._generation_active = False
        self._contract_json = _canonical(self._recipe())
        self._contract_sha256 = _digest(self._recipe())
        self._consensus(self._contract_json, "stock recipe")

    def _recipe(self):
        return {"protocol": "stock-grpo-explicit-sft-reference-20261013-v1",
                "trl_version": trl.__version__,
                "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_loss_source_sha256": hashlib.sha256(
                    inspect.getsource(GRPOTrainer._compute_loss).encode()).hexdigest(),
                "beta": self.beta, "num_generations": self.num_generations,
                "max_completion_length": self.max_completion_length,
                "max_steps": self.args.max_steps, "loss_type": self.loss_type,
                "scale_rewards": self.scale_rewards, "num_iterations": self.num_iterations,
                "steps_per_generation": self.args.steps_per_generation,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "disable_dropout": self.args.disable_dropout,
                "training_arguments": _training_arguments(self.args),
                "reward_func_names": self.reward_func_names,
                "reference_identity": self.reference_identity,
                "reference_state_sha256": self._reference_initial_sha256,
                "recipe_contract": self.recipe_contract}

    def _generate_and_score_completions(self, inputs):
        self._collective_check(self._assert_contract, "stock active contract")
        if not self._reference_prepared_checked:
            if not self._resuming:
                self.prepared_reference_audit = self._collective_check(
                    self._audit_initial_pair, "stock prepared initial equality")
            self._reference_prepared_checked = True
        self._generation_active = True
        try:
            output = super()._generate_and_score_completions(inputs)
            if not torch.isfinite(output["advantages"]).all():
                raise RuntimeError("Stock baseline produced nonfinite advantages")
            return output
        finally:
            self._generation_active = False

    def _compute_loss(self, model, inputs):
        def admit_forward():
            self._assert_contract()
            if inputs["advantages"].requires_grad:
                raise ValueError("Stock advantages must be detached from the policy")
            if inputs.get("ref_per_token_logps") is None or inputs["ref_per_token_logps"].requires_grad:
                raise ValueError("Stock KL requires detached independent frozen SFT scores")

        self._collective_check(admit_forward, "pre-forward stock loss admission")
        return super()._compute_loss(model, inputs)

    # Identical immutable-reference checkpoint mechanics; no estimator methods
    # are inherited from the RLOO trainer.
    _assert_contract = SFTAnchoredRLOOTrainer._assert_contract
    _checkpoint_payload = SFTAnchoredRLOOTrainer._checkpoint_payload
    _validate_checkpoint = SFTAnchoredRLOOTrainer._validate_checkpoint

    def _save_checkpoint(self, model, trial):
        if self._generation_active:
            raise RuntimeError("Cannot checkpoint an unfinished stock rollout")
        payload = self._collective_check(
            lambda: self._checkpoint_payload(self.state.global_step), "stock checkpoint contract")
        self._consensus(payload, "stock checkpoint contract")
        super()._save_checkpoint(model, trial)
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        self._collective_check(lambda: save_rank_state(
            directory, self.optimizer, self.lr_scheduler,
            rank=int(self.accelerator.process_index), world_size=int(self.accelerator.num_processes),
            step=int(self.state.global_step)), "stock rank-owned checkpoint state")
        if self.accelerator.is_main_process:
            _write_json_atomic(directory / CONTRACT_FILENAME, payload)
        self.accelerator.wait_for_everyone()

    def _load_optimizer_and_scheduler(self, checkpoint):
        self._collective_check(self._assert_contract, "stock resume active contract")
        if checkpoint:
            step = self._collective_check(lambda: self._validate_checkpoint(checkpoint),
                                          "stock resume recipe validation")
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            self._collective_check(lambda: restore_rank_state(
                checkpoint, self.optimizer, self.lr_scheduler,
                rank=int(self.accelerator.process_index), world_size=int(self.accelerator.num_processes),
                step=step), "stock rank-owned optimizer restoration")
            self._resuming = self._reference_prepared_checked = True
        return result
