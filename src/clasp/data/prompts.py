"""Alpaca-style prompt templates.

:func:`build_formatted_prompt` renders a task row into a prompt. Styles: ``alpaca`` (plain
instruction format), ``alpaca_chat`` (wraps the Alpaca content with
``tokenizer.apply_chat_template``), ``alpaca_hint`` (adds a no-repair construction and
checking instruction), ``jiang_f638`` (the byte-exact released Jiang et al. inference
prompt), and ``auto`` (picks ``alpaca_chat`` for Gemma, else ``alpaca``). Each generic
style also has a ``no_reason`` variant.
"""

from __future__ import annotations

import os
from typing import Any

ALPACA_PROMPT_WITH_REASONING = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.

Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective. You can reason about how to achieve that in the <reasoning></reasoning> section.

### Instruction:{}

### Input:{}

### Response:
"""

ALPACA_PROMPT_NO_REASONING = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.

Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

### Instruction:{}

### Input:{}

### Response:
"""

ALPACA_HINT_PROMPT_NO_REASONING = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.

Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective. Before writing the response, internally construct and check a high-quality solution. For routing instances, prefer short edges, avoid obvious crossings, and check for local improvements. For graph instances, verify selected vertices are unique, in range, and pairwise nonadjacent; if a selected edge exists, choose a conflict-free alternative before output. Do not include analysis; output only the requested solution format.

### Instruction:{}

### Input:{}

### Response:
"""

ALPACA_HINT_PROMPT_WITH_REASONING = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.

Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective. Before writing the response, construct and check a high-quality solution. For routing instances, prefer short edges, avoid obvious crossings, and check for local improvements. For graph instances, verify selected vertices are unique, in range, and pairwise nonadjacent; if a selected edge exists, choose a conflict-free alternative before output. You can reason about how to achieve that in the <reasoning></reasoning> section.

### Instruction:{}

### Input:{}

### Response:
"""

# Byte-faithful runtime value of ``alpaca_prompt`` in Jiang et al.'s released
# ``main_train.py`` at commit f6388991d243c0aa91d9ee7e2fa18ed1c9e72477.
# The space before the first newline, four leading spaces on the remaining
# non-empty lines, and lack of a newline after ``### Response:`` are material.
# Keep this separate from the generic prompt-style registry: it is an opt-in
# reproduction contract, not a new default CLASP prompt style.
JIANG_F638_SFT_TEMPLATE = (
    "Below is an instruction describing a combinatorial optimization problem. It is paired "
    "with an input that provides the data of the instance. \n"
    "    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    "the given objective.\n"
    "\n"
    "    ### Instruction:{}\n"
    "\n"
    "    ### Input:{}\n"
    "\n"
    "    ### Response:{}"
)

# Byte-faithful runtime value of ``alpaca_prompt`` in the released ``eval.py``
# at the same commit.  It intentionally ends at the response colon with no
# trailing newline or EOS token.  Keep it distinct from the SFT template so
# evaluation cannot accidentally include a gold output.
JIANG_F638_INFERENCE_TEMPLATE = (
    "Below is an instruction describing a combinatorial optimization problem. It is paired "
    "with an input that provides the data of the instance. \n"
    "    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    "the given objective.\n"
    "\n"
    "    ### Instruction:{}\n"
    "\n"
    "    ### Input:{}\n"
    "\n"
    "    ### Response:"
)

# Byte-faithful ``alpaca_prompt`` used by the released ``utils.get_dataset``
# for FOARL.  Unlike the SFT/eval prompt, this one explicitly permits a
# reasoning section.  Dispatching on ``no_reason`` prevents future RL from
# silently reusing the materially different SFT/eval prefix.
JIANG_F638_RL_TEMPLATE = (
    "Below is an instruction describing a combinatorial optimization problem. It is paired "
    "with an input that provides the data of the instance. \n"
    "    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    "the given objective. You can reason about how to achieve that in the "
    "<reasoning></reasoning> section.\n"
    "\n"
    "    ### Instruction:{}\n"
    "\n"
    "    ### Input:{}\n"
    "\n"
    "    ### Response:"
)

# The upstream collator searched for the standalone tokenization of
# ``### Response:``.  It does not occur in the exact rendered Qwen text: the
# indentation changes the first marker token and the colon can merge with the
# first output subword.  The corrected publication lane therefore anchors on
# this byte-exact contextual header without the colon.  Its next token is the
# first supervised token, and may jointly encode the colon plus an output
# prefix (for example Qwen's ``:Set`` token).
JIANG_F638_RESPONSE_HEADER = "    ### Response"
JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE = "### Response:"
JIANG_F638_SUPERVISED_BOUNDARY = "contextual_response_header_next_token"
JIANG_F638_FIRST_SUPERVISED_TOKEN_POLICY = (
    "include_response_colon_token_which_may_span_output_prefix"
)

# Backward-compatible alias used by tests and legacy imports.
ALPACA_PROMPT = ALPACA_PROMPT_WITH_REASONING

SUPPORTED_PROMPT_STYLES = (
    "auto",
    "alpaca",
    "alpaca_chat",
    "alpaca_hint",
    "jiang_f638",
)


def normalize_prompt_style(prompt_style: str | None) -> str:
    style = str(prompt_style or "auto").strip().lower()
    if style not in SUPPORTED_PROMPT_STYLES:
        raise ValueError(
            f"Unsupported prompt_style={prompt_style!r}. Expected one of {SUPPORTED_PROMPT_STYLES}."
        )
    return style


def build_alpaca_prompt(instruction: str, input_text: str, *, no_reason: bool = False) -> str:
    template = ALPACA_PROMPT_NO_REASONING if no_reason else ALPACA_PROMPT_WITH_REASONING
    return template.format(str(instruction), str(input_text))


def build_jiang_f638_sft_text(
    instruction: object,
    input_text: object,
    output: object,
    *,
    eos_token: str,
) -> str:
    """Render the full SFT ``text`` field exactly as Jiang's f638 trainer.

    The released formatter embedded the target immediately after
    ``### Response:`` and then concatenated ``tokenizer.eos_token`` verbatim.
    In particular, this helper does not add a response newline, strip fields,
    coerce or deduplicate EOS markers, or change the generic prompt APIs.
    """

    return JIANG_F638_SFT_TEMPLATE.format(instruction, input_text, output) + eos_token


def build_jiang_f638_inference_prompt(instruction: object, input_text: object) -> str:
    """Render the released f638 evaluation prompt with no target or EOS."""

    return JIANG_F638_INFERENCE_TEMPLATE.format(instruction, input_text)


def build_jiang_f638_rl_prompt(instruction: object, input_text: object) -> str:
    """Render the released f638 FOARL prompt with no completion or EOS."""

    return JIANG_F638_RL_TEMPLATE.format(instruction, input_text)


def jiang_f638_response_header_token_ids(tokenizer: Any) -> list[int]:
    """Encode the corrected contextual response header without special tokens."""

    token_ids = tokenizer.encode(
        JIANG_F638_RESPONSE_HEADER,
        add_special_tokens=False,
    )
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise ValueError("Jiang-f638 response header encoded to invalid token IDs")
    return token_ids


def locate_jiang_f638_supervised_start(
    text: str,
    input_ids: list[int],
    response_header_token_ids: list[int],
) -> int:
    """Return the first supervised token under the corrected f638 boundary.

    The exact contextual header must occur once in both raw text and token
    space.  The returned index is immediately after the header token sequence,
    so its token may span the response colon and the first output subword.
    """

    raw_occurrences = text.count(JIANG_F638_RESPONSE_HEADER)
    if raw_occurrences != 1:
        description = (
            "missing" if raw_occurrences == 0 else f"ambiguous ({raw_occurrences} matches)"
        )
        raise ValueError(f"Jiang-f638 raw response header is {description}")
    if not response_header_token_ids:
        raise ValueError("Jiang-f638 response header token sequence is empty")

    starts: list[int] = []
    cursor = 0
    width = len(response_header_token_ids)
    final_start = len(input_ids) - width
    while cursor <= final_start:
        try:
            start = input_ids.index(response_header_token_ids[0], cursor, final_start + 1)
        except ValueError:
            break
        if input_ids[start : start + width] == response_header_token_ids:
            starts.append(start)
        cursor = start + 1
    if len(starts) != 1:
        description = "missing" if not starts else f"ambiguous ({len(starts)} matches)"
        raise ValueError(f"Jiang-f638 response header token sequence is {description}")

    supervised_start = starts[0] + width
    if supervised_start >= len(input_ids):
        raise ValueError("no supervised token follows the Jiang-f638 response header")
    return supervised_start


def build_alpaca_hint_prompt(instruction: str, input_text: str, *, no_reason: bool = False) -> str:
    template = ALPACA_HINT_PROMPT_NO_REASONING if no_reason else ALPACA_HINT_PROMPT_WITH_REASONING
    return template.format(str(instruction), str(input_text))


def _apply_chat_template(tokenizer: Any, prompt_text: str) -> str | None:
    if tokenizer is None:
        return None
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_fn):
        return None

    messages = [{"role": "user", "content": str(prompt_text)}]
    base_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    try:
        return str(apply_fn(messages, enable_thinking=False, **base_kwargs))
    except TypeError:
        pass
    except Exception:
        return None

    try:
        return str(apply_fn(messages, **base_kwargs))
    except Exception:
        return None


def tokenizer_supports_chat_template(tokenizer: Any) -> bool:
    return _apply_chat_template(tokenizer, "ping") is not None


def resolve_prompt_style(
    prompt_style: str | None,
    *,
    model_path: str,
    tokenizer: Any | None,
) -> str:
    style = normalize_prompt_style(prompt_style)
    if style != "auto":
        return style

    model_name = os.path.basename(str(model_path).rstrip(os.sep)).lower()
    if "gemma" in model_name and tokenizer_supports_chat_template(tokenizer):
        return "alpaca_chat"
    return "alpaca"


def prompt_template_tag(prompt_style: str, *, no_reason: bool) -> str:
    style = normalize_prompt_style(prompt_style)
    if style == "auto":
        style = "alpaca"
    return f"{style}_no_reason" if no_reason else style


def build_formatted_prompt(
    instruction: str,
    input_text: str,
    *,
    no_reason: bool = False,
    prompt_style: str = "alpaca",
    tokenizer: Any | None = None,
    strict_chat_prompt: bool = False,
) -> str:
    style = normalize_prompt_style(prompt_style)
    if style == "auto":
        style = resolve_prompt_style(style, model_path="", tokenizer=tokenizer)

    if style == "alpaca_hint":
        return build_alpaca_hint_prompt(instruction, input_text, no_reason=no_reason)
    if style == "jiang_f638":
        if no_reason:
            return build_jiang_f638_inference_prompt(instruction, input_text)
        return build_jiang_f638_rl_prompt(instruction, input_text)

    alpaca_prompt = build_alpaca_prompt(instruction, input_text, no_reason=no_reason)
    if style != "alpaca_chat":
        return alpaca_prompt

    formatted = _apply_chat_template(tokenizer, alpaca_prompt)
    if formatted is not None:
        return formatted
    if strict_chat_prompt:
        raise ValueError(
            "prompt_style='alpaca_chat' was requested but tokenizer.apply_chat_template "
            "is unavailable or has no chat template. "
            "Provide a tokenizer that includes a chat template (for Gemma 4, typically a *-it tokenizer), "
            "or switch to --prompt_style alpaca."
        )
    return alpaca_prompt
