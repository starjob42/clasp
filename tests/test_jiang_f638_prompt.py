from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from clasp.data.prompts import (
    ALPACA_PROMPT_WITH_REASONING,
    JIANG_F638_RESPONSE_HEADER,
    JIANG_F638_INFERENCE_TEMPLATE,
    JIANG_F638_RL_TEMPLATE,
    JIANG_F638_SFT_TEMPLATE,
    JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE,
    build_alpaca_prompt,
    build_formatted_prompt,
    build_jiang_f638_inference_prompt,
    build_jiang_f638_rl_prompt,
    build_jiang_f638_sft_text,
    locate_jiang_f638_supervised_start,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
QWEN_TOKENIZER_JSON_SHA256 = "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"

INSTRUCTION = "Solve."
INPUT_TEXT = "Nodes: [0, 1]"
OUTPUT = "Set: [0]"
EOS_TOKEN = "<|endoftext|>"

# This literal was transcribed independently from ``main_train.py`` at
# UPSTREAM_COMMIT. Keeping it outside production code makes whitespace drift a
# byte-level test failure instead of comparing the helper to its own constant.
EXPECTED_TEXT_BYTES = (
    b"Below is an instruction describing a combinatorial optimization problem. It is paired "
    b"with an input that provides the data of the instance. \n"
    b"    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    b"the given objective.\n"
    b"\n"
    b"    ### Instruction:Solve.\n"
    b"\n"
    b"    ### Input:Nodes: [0, 1]\n"
    b"\n"
    b"    ### Response:Set: [0]"
    b"<|endoftext|>"
)
EXPECTED_QWEN25_TOKEN_IDS = [
    38214,
    374,
    458,
    7600,
    22692,
    264,
    3614,
    17272,
    530,
    25262,
    3491,
    13,
    1084,
    374,
    34426,
    448,
    458,
    1946,
    429,
    5707,
    279,
    821,
    315,
    279,
    2867,
    13,
    715,
    262,
    4615,
    3383,
    374,
    311,
    8193,
    264,
    42203,
    6291,
    429,
    7553,
    4756,
    320,
    1065,
    318,
    4756,
    476,
    30027,
    4756,
    8,
    279,
    2661,
    16538,
    382,
    262,
    16600,
    29051,
    75369,
    3948,
    382,
    262,
    16600,
    5571,
    25,
    12288,
    25,
    508,
    15,
    11,
    220,
    16,
    2533,
    262,
    16600,
    5949,
    15606,
    25,
    508,
    15,
    60,
    151643,
]
EXPECTED_UPSTREAM_MARKER_TOKEN_IDS = [14374, 5949, 25]
EXPECTED_CONTEXTUAL_HEADER_TOKEN_IDS = [262, 16600, 5949]


def _formatted_text(*, output: object = OUTPUT, eos_token: str = EOS_TOKEN) -> str:
    return build_jiang_f638_sft_text(
        INSTRUCTION,
        INPUT_TEXT,
        output,
        eos_token=eos_token,
    )


def test_jiang_f638_inference_prompt_is_exact_released_eval_prefix() -> None:
    rendered = build_jiang_f638_inference_prompt(INSTRUCTION, INPUT_TEXT)
    expected = EXPECTED_TEXT_BYTES.decode("utf-8").split("Set: [0]", 1)[0]

    assert rendered == expected
    assert rendered == build_formatted_prompt(
        INSTRUCTION,
        INPUT_TEXT,
        no_reason=True,
        prompt_style="jiang_f638",
    )
    assert rendered.endswith("    ### Response:")
    assert not rendered.endswith("\n")
    assert EOS_TOKEN not in rendered
    assert "{}" in JIANG_F638_INFERENCE_TEMPLATE
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == (
        "985f4c777c5b6ca1e677fa74029b14e7ae35890d9802b266d8ac4971744f65bf"
    )


def test_jiang_f638_reasoning_prompt_is_exact_released_foarl_prefix() -> None:
    rendered = build_jiang_f638_rl_prompt(INSTRUCTION, INPUT_TEXT)

    assert rendered == build_formatted_prompt(
        INSTRUCTION,
        INPUT_TEXT,
        no_reason=False,
        prompt_style="jiang_f638",
    )
    assert "<reasoning></reasoning> section." in rendered
    assert rendered.endswith("    ### Response:")
    assert not rendered.endswith("\n")
    assert EOS_TOKEN not in rendered
    assert "{}" in JIANG_F638_RL_TEMPLATE
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == (
        "4b26a647400e7294c287cd0111dd341f5916724525c0b1a3f326e6b6c74121d3"
    )


def test_jiang_f638_sft_text_is_byte_exact_and_appends_eos_verbatim() -> None:
    rendered = _formatted_text()

    assert rendered.encode("utf-8") == EXPECTED_TEXT_BYTES
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == (
        "49ac52e99d63da4166962c5d36955159870a806ca27b58b456e954b714af9356"
    )
    assert hashlib.sha256(JIANG_F638_SFT_TEMPLATE.encode("utf-8")).hexdigest() == (
        "0e4ff1b30b6b81b3ba4d84fd68c791f7a600dd470a2aae4dfe129fca8ffe6883"
    )
    assert rendered.endswith("    ### Response:Set: [0]<|endoftext|>")
    assert "instance. \n    Your task" in rendered

    # Match upstream's plain string concatenation: an EOS already present in
    # the output is not normalized away, and an empty EOS remains empty.
    assert _formatted_text(output=OUTPUT + EOS_TOKEN).endswith(EOS_TOKEN * 2)
    assert _formatted_text(eos_token="").endswith("    ### Response:Set: [0]")
    with pytest.raises(TypeError):
        _formatted_text(eos_token=None)  # type: ignore[arg-type]


def test_default_alpaca_prompt_contract_is_unchanged() -> None:
    rendered = build_alpaca_prompt(INSTRUCTION, INPUT_TEXT)

    assert rendered == ALPACA_PROMPT_WITH_REASONING.format(INSTRUCTION, INPUT_TEXT)
    assert "\nYour task" in rendered
    assert "\n    Your task" not in rendered
    assert rendered.endswith("### Response:\n")


def test_real_qwen25_token_golden_exposes_and_corrects_f638_boundary() -> None:
    upstream_starts = [
        index
        for index in range(
            len(EXPECTED_QWEN25_TOKEN_IDS) - len(EXPECTED_UPSTREAM_MARKER_TOKEN_IDS) + 1
        )
        if EXPECTED_QWEN25_TOKEN_IDS[index : index + len(EXPECTED_UPSTREAM_MARKER_TOKEN_IDS)]
        == EXPECTED_UPSTREAM_MARKER_TOKEN_IDS
    ]
    assert upstream_starts == []

    supervised_start = locate_jiang_f638_supervised_start(
        _formatted_text(),
        EXPECTED_QWEN25_TOKEN_IDS,
        EXPECTED_CONTEXTUAL_HEADER_TOKEN_IDS,
    )
    assert supervised_start == 72
    # Qwen merges the response colon and first output subword into ``:Set``;
    # starting at this token retains the complete target rather than dropping
    # its first subword.
    assert EXPECTED_QWEN25_TOKEN_IDS[supervised_start] == 15606
    assert _formatted_text().count(JIANG_F638_RESPONSE_HEADER) == 1


def _find_local_qwen_tokenizer_json() -> Path | None:
    override = os.environ.get("CLASP_JIANG_F638_TOKENIZER_JSON")
    candidates = [
        Path(override).expanduser() if override else None,
        ROOT / "models/local_tsp_sft500k_seed42_instruct/tokenizer.json",
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def test_jiang_f638_qwen25_token_ids_match_golden() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer_json = _find_local_qwen_tokenizer_json()
    if tokenizer_json is None:
        pytest.skip(
            "Qwen2.5 tokenizer.json is not stored in git; set "
            "CLASP_JIANG_F638_TOKENIZER_JSON for the token golden"
        )

    assert hashlib.sha256(tokenizer_json.read_bytes()).hexdigest() == (QWEN_TOKENIZER_JSON_SHA256)
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    encoded = tokenizer.encode(_formatted_text(), add_special_tokens=True)
    inference_encoded = tokenizer.encode(
        build_jiang_f638_inference_prompt(INSTRUCTION, INPUT_TEXT),
        add_special_tokens=True,
    )

    assert encoded.ids == EXPECTED_QWEN25_TOKEN_IDS
    assert inference_encoded.ids == EXPECTED_QWEN25_TOKEN_IDS[:72] + [25]
    assert encoded.ids[-1] == 151643
    assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == _formatted_text()

    upstream_marker = tokenizer.encode(
        JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE,
        add_special_tokens=False,
    ).ids
    contextual_header = tokenizer.encode(
        JIANG_F638_RESPONSE_HEADER,
        add_special_tokens=False,
    ).ids
    assert upstream_marker == EXPECTED_UPSTREAM_MARKER_TOKEN_IDS
    assert contextual_header == EXPECTED_CONTEXTUAL_HEADER_TOKEN_IDS
    assert (
        locate_jiang_f638_supervised_start(
            _formatted_text(),
            encoded.ids,
            contextual_header,
        )
        == 72
    )
