"""Model and tokenizer loading, including PEFT/LoRA handling.

Loads a base model or adapter checkpoint at the requested dtype, detects whether a path is
a PEFT checkpoint, and merges adapters into the base weights
(:func:`merge_peft_checkpoint`). LoRA target-module selection is driven by the
``CLASP_LORA_TARGET_MODULES`` environment variable: ``default`` targets attention only,
while ``jiang`` and ``full`` target attention plus MLP.
"""

from __future__ import annotations

import json
import os

from peft import (  # type: ignore[import-not-found]
    LoraConfig,
    PeftConfig,
    PeftModel,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

from clasp.runtime import env_world_size

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
JIANG_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _align_special_token_ids_with_tokenizer(model, tokenizer) -> None:
    for field_name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, field_name, None)
        if hasattr(model.config, field_name):
            setattr(model.config, field_name, token_id)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, field_name):
            setattr(generation_config, field_name, token_id)


def assert_exact_paired_special_token_ids(policy, reference, tokenizer) -> None:
    """Require both paired surfaces to equal the bound tokenizer exactly."""

    vocabulary_sizes = {}
    for role, model in (("policy", policy), ("reference", reference)):
        vocabulary_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size <= 0
        ):
            raise RuntimeError(
                f"exact paired {role} model has invalid vocab_size: {vocabulary_size!r}"
            )
        vocabulary_sizes[role] = vocabulary_size

    try:
        tokenizer_size = len(tokenizer)
    except TypeError:
        tokenizer_size = getattr(tokenizer, "vocab_size", None)
    if (
        isinstance(tokenizer_size, bool)
        or not isinstance(tokenizer_size, int)
        or tokenizer_size <= 0
    ):
        raise RuntimeError(f"exact paired tokenizer has invalid size: {tokenizer_size!r}")

    tokenizer_values = {}
    for token_field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if not hasattr(tokenizer, token_field):
            raise RuntimeError(f"exact paired tokenizer lacks {token_field}")
        value = getattr(tokenizer, token_field, None)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= tokenizer_size
            or any(value >= vocabulary_size for vocabulary_size in vocabulary_sizes.values())
        ):
            raise RuntimeError(f"exact paired tokenizer has invalid {token_field}: {value!r}")
        tokenizer_values[token_field] = value

    for attribute, label in (
        ("config", "model config"),
        ("generation_config", "generation config"),
    ):
        policy_config = getattr(policy, attribute, None)
        reference_config = getattr(reference, attribute, None)
        if policy_config is None or reference_config is None:
            raise RuntimeError(
                f"exact paired {label} is unavailable: "
                f"policy={policy_config is not None}, reference={reference_config is not None}"
            )
        for token_field in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if not hasattr(policy_config, token_field) or not hasattr(
                reference_config, token_field
            ):
                raise RuntimeError(f"exact paired {label} lacks {token_field}")
            policy_value = getattr(policy_config, token_field, None)
            reference_value = getattr(reference_config, token_field, None)
            for role, value in (("policy", policy_value), ("reference", reference_value)):
                if value is None:
                    continue
                values = value if isinstance(value, (list, tuple)) else (value,)
                if not values or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or token_id >= vocabulary_sizes[role]
                    for token_id in values
                ):
                    raise RuntimeError(
                        f"exact paired {role} {label} has invalid {token_field}: {value!r}"
                    )
            tokenizer_value = tokenizer_values[token_field]
            if policy_value != tokenizer_value or reference_value != tokenizer_value:
                raise RuntimeError(
                    f"exact paired {label} mismatch for {token_field}: "
                    f"policy={policy_value!r}, reference={reference_value!r}, "
                    f"tokenizer={tokenizer_value!r}"
                )


def _ensure_input_require_grads(model) -> None:
    """Keep autograd alive for PEFT + gradient-checkpointing training.

    When we resume from an existing LoRA/PEFT checkpoint, the base model is mostly
    frozen. With gradient checkpointing enabled, PyTorch can warn that none of the
    checkpointed inputs require gradients, which then detaches the loss and causes
    backward() to fail. Enabling input grads on the embeddings is the standard fix.
    """

    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()
        return

    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return

    input_embeddings = get_input_embeddings()
    if input_embeddings is None:
        return

    def _make_outputs_require_grad(_module, _inputs, output):
        if hasattr(output, "requires_grad_"):
            output.requires_grad_(True)

    input_embeddings.register_forward_hook(_make_outputs_require_grad)


def _resolve_tokenizer_path(model_path: str, tokenizer_path: str | None = None) -> str:
    candidate = str(tokenizer_path or "").strip()
    if candidate:
        return candidate
    return str(model_path)


def resolve_tokenizer_path(model_path: str, tokenizer_path: str | None = None) -> str:
    return _resolve_tokenizer_path(model_path, tokenizer_path=tokenizer_path)


def _load_tokenizer(tokenizer_source_path: str):
    base_kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_source_path,
            fix_mistral_regex=True,
            **base_kwargs,
        )
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(tokenizer_source_path, **base_kwargs)


def _load_model_with_dtype(from_pretrained_fn, model_path: str, *, dtype, **kwargs):
    try:
        return from_pretrained_fn(model_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return from_pretrained_fn(model_path, torch_dtype=dtype, **kwargs)


def load_peft_causal_lm_with_explicit_adapter_autocast(
    adapter_path: str,
    *,
    dtype,
    device_map,
    is_trainable: bool,
    trust_remote_code: bool,
    local_files_only: bool = False,
    expected_base_model_path: str | None = None,
):
    """Load a PEFT causal LM with an explicit, symmetric adapter dtype policy.

    PEFT's auto loader does not let callers explicitly pass
    ``autocast_adapter_dtype``: that keyword is also forwarded to the base
    Transformers constructor, where it is rejected.  Loading the base and
    adapter in two explicit stages avoids relying on PEFT's implicit default
    and guarantees that trainable policies and frozen references both promote
    FP16/BF16 LoRA tensors to FP32.

    The base model may use the low-memory meta-device load path.  The adapter
    deliberately may not: pinned PEFT 0.17 moves an adapter loaded through that
    path to the BF16 base-layer dtype before ``autocast_adapter_dtype`` promotes
    it back to FP32.  That F32 -> BF16 -> F32 round trip changes saved F32 LoRA
    values.  Loading adapter tensors normally preserves their serialized values
    exactly while retaining the explicit autocast policy.
    """

    config_kwargs = {"local_files_only": True} if local_files_only else {}
    peft_config = PeftConfig.from_pretrained(adapter_path, **config_kwargs)
    configured_base = str(peft_config.base_model_name_or_path or "").strip()
    if not configured_base:
        raise ValueError(f"PEFT adapter lacks base_model_name_or_path: {adapter_path}")
    if expected_base_model_path is not None:
        configured = os.path.realpath(os.path.expanduser(configured_base))
        expected = os.path.realpath(os.path.expanduser(expected_base_model_path))
        if configured != expected:
            raise ValueError(
                "PEFT adapter base differs from the exact expected base: "
                f"configured={configured!r}, expected={expected!r}"
            )

    base_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **config_kwargs,
    }
    base_revision = getattr(peft_config, "revision", None)
    if isinstance(base_revision, str) and base_revision.strip():
        base_kwargs["revision"] = base_revision
    base_model = _load_model_with_dtype(
        AutoModelForCausalLM.from_pretrained,
        configured_base,
        dtype=dtype,
        **base_kwargs,
    )
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        config=peft_config,
        is_trainable=is_trainable,
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=False,
        **config_kwargs,
    )


def load_model_and_tokenizer(
    model_path: str,
    dtype: str,
    *,
    padding_side: str,
    tokenizer_path: str | None = None,
):
    import torch  # type: ignore[import-not-found]

    if padding_side not in {"left", "right"}:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    tokenizer = _load_tokenizer(_resolve_tokenizer_path(model_path, tokenizer_path=tokenizer_path))
    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True
    tokenizer.padding_side = padding_side

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    device_map = None
    if torch.cuda.is_available() and env_world_size() <= 1:
        device_map = {"": f"cuda:{local_rank}"}

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_adapter_checkpoint = os.path.isfile(adapter_config_path)
    is_peft_model = False

    if is_adapter_checkpoint:
        model = load_peft_causal_lm_with_explicit_adapter_autocast(
            model_path,
            dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True,
            is_trainable=True,
        )
        is_peft_model = True
    else:
        model = _load_model_with_dtype(
            AutoModelForCausalLM.from_pretrained,
            model_path,
            dtype=resolved_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )
        # Keep compatibility with environments where AutoModel may already return
        # an injected PEFT model object.
        is_peft_model = bool(getattr(model, "peft_config", None))

    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    _align_special_token_ids_with_tokenizer(model, tokenizer)
    if is_peft_model:
        _ensure_input_require_grads(model)

    return model, tokenizer, is_peft_model


def path_looks_like_peft_checkpoint(model_path: str) -> bool:
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def infer_base_model_path_from_peft_checkpoint(model_path: str) -> str | None:
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    candidate = payload.get("base_model_name_or_path")
    if not candidate:
        return None
    return str(candidate)


def peft_checkpoint_bias_policy(model_path: str) -> str | None:
    """Return the saved PEFT bias policy, or ``None`` for a full model path."""

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read PEFT adapter config: {adapter_config_path}") from exc
    bias = str(payload.get("bias", "none") or "none").strip().lower()
    if bias not in {"none", "lora_only", "all"}:
        raise ValueError(f"Unsupported PEFT bias policy {bias!r} in {adapter_config_path}")
    return bias


def merge_peft_checkpoint(
    model_path: str,
    output_dir: str,
    dtype: str,
) -> str:
    import torch  # type: ignore[import-not-found]

    os.makedirs(output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    device_map = None
    if torch.cuda.is_available():
        device_map = {"": f"cuda:{local_rank}"}

    model = load_peft_causal_lm_with_explicit_adapter_autocast(
        model_path,
        dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=True,
        is_trainable=False,
    )
    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    merged_model = model.merge_and_unload()
    _align_special_token_ids_with_tokenizer(merged_model, tokenizer)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del merged_model
    del model
    del tokenizer
    return output_dir


def lora_target_modules_from_env() -> list[str]:
    raw_value = str(os.environ.get("CLASP_LORA_TARGET_MODULES", "") or "").strip()
    normalized = raw_value.lower().replace("-", "_")
    if normalized in {"", "attention", "attn", "default"}:
        return list(DEFAULT_LORA_TARGET_MODULES)
    if normalized in {"jiang", "attn_mlp", "attention_mlp", "attention_plus_mlp", "full"}:
        return list(JIANG_LORA_TARGET_MODULES)

    pieces = [piece.strip() for piece in raw_value.replace(";", ",").split(",")]
    modules = [piece for piece in pieces if piece]
    if not modules:
        return list(DEFAULT_LORA_TARGET_MODULES)
    return modules


def lora_bias_from_env() -> str:
    """Resolve the PEFT LoRA bias policy.

    Legacy CLASP checkpoints used ``none``.  Jiang et al.'s released training
    configuration uses ``lora_only``; keeping this explicit in the environment
    lets new specialist runs match that capacity without changing how existing
    checkpoints are loaded.
    """

    raw_value = str(os.environ.get("CLASP_LORA_BIAS", "none") or "none").strip().lower()
    aliases = {
        "none": "none",
        "no": "none",
        "false": "none",
        "lora": "lora_only",
        "lora_only": "lora_only",
        "all": "all",
    }
    if raw_value not in aliases:
        allowed = ", ".join(sorted({"none", "lora_only", "all"}))
        raise ValueError(f"CLASP_LORA_BIAS must be one of {allowed}; got {raw_value!r}")
    return aliases[raw_value]


def build_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias_from_env(),
        target_modules=lora_target_modules_from_env(),
        task_type="CAUSAL_LM",
    )
