from __future__ import annotations

import ast
import re
import shlex
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fenced_block_after(text: str, marker: str) -> str:
    tail = text.split(marker, 1)[1]
    return tail.split("```bash", 1)[1].split("```", 1)[0]


def _documented_clasp_eval_commands() -> list[tuple[Path, int, list[str]]]:
    documentation = [ROOT / "README.md"]
    for directory in ("checkpoints", "docs", "experiments"):
        documentation.extend((ROOT / directory).rglob("*.md"))

    commands: list[tuple[Path, int, list[str]]] = []
    fence = re.compile(
        r"^```(?:bash|sh|shell)\s*$\n(?P<body>.*?)^```\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )
    for path in sorted(set(documentation)):
        text = path.read_text(encoding="utf-8")
        for match in fence.finditer(text):
            body = match.group("body")
            if not re.search(r"^\s*clasp-eval(?:\s|$)", body, flags=re.MULTILINE):
                continue
            command = "\n".join(
                line for line in body.splitlines() if not line.lstrip().startswith("#")
            ).replace("\\\n", " ")
            argv = shlex.split(command)
            line_number = text[: match.start()].count("\n") + 1
            commands.append((path.relative_to(ROOT), line_number, argv))
    return commands


def _eval_parser_options() -> tuple[set[str], set[str]]:
    source = (ROOT / "src" / "clasp" / "entrypoints" / "evaluate.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    parser_function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_eval_parser"
    )
    options: set[str] = set()
    boolean_options: set[str] = set()
    for node in ast.walk(parser_function):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        declared = {
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("-")
        }
        options.update(declared)
        boolean_optional = any(
            keyword.arg == "action"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "BooleanOptionalAction"
            for keyword in node.keywords
        )
        if boolean_optional:
            boolean_options.update(declared)
    return options, boolean_options


def test_readme_confirmation_template_includes_campaign_manifest() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = _fenced_block_after(readme, "A GRPO confirmation invocation")
    assert "clasp-eval" in block
    assert "--benchmark_manifest" in block
    assert "--campaign_manifest" in block
    assert "--checkpoint_selection_receipt" in block


def test_readme_does_not_call_contaminated_author_eval_held_out() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "they are the same held-out benchmark instances" not in readme
    assert "299/300 CVRP, 300/300 MIS, and 299/300 OP" in normalized
    assert "llmcosolver_f638_public_data_addendum_v2.md" in readme
    assert "preregistered fresh, disjoint IID" in readme


def test_protocol_separates_evaluator_and_full_grid_aggregate_commands() -> None:
    protocol = (ROOT / "experiments" / "sota_20260814" / "PROTOCOL.md").read_text(encoding="utf-8")
    evaluator = _fenced_block_after(protocol, "Invoke the evaluator explicitly")
    assert "clasp-eval" in evaluator
    assert "--campaign_manifest" in evaluator
    for aggregate_only in ("--run ", "--baseline ", "--challenger ", "--output "):
        assert aggregate_only not in evaluator

    aggregate = _fenced_block_after(protocol, "all 30 exact")
    assert "python scripts/aggregate_sota.py" in aggregate
    assert "--campaign-manifest" in aggregate
    assert aggregate.count("--run ") == 30
    assert "--baseline " in aggregate
    assert "--challenger " in aggregate
    assert "--output " in aggregate


def test_every_documented_clasp_eval_block_matches_parser_and_confirmation_contract() -> None:
    commands = _documented_clasp_eval_commands()
    assert Counter(path for path, _line, _argv in commands) == Counter(
        {
            Path("README.md"): 2,
            Path("checkpoints/README.md"): 1,
            Path("docs/sota_advancement_and_selection.md"): 1,
            Path("experiments/sota_20260814/PROTOCOL.md"): 1,
        }
    )

    parser_options, boolean_options = _eval_parser_options()
    for path, line_number, argv in commands:
        location = f"{path}:{line_number}"
        assert argv[0] == "clasp-eval", location
        for token in argv[1:]:
            if not token.startswith("--"):
                continue
            option = token.split("=", 1)[0]
            positive = "--" + option.removeprefix("--no-")
            assert option in parser_options or (
                option.startswith("--no-") and positive in boolean_options
            ), f"{location}: undocumented parser option {option}"

        purpose = (
            argv[argv.index("--evaluation_purpose") + 1]
            if "--evaluation_purpose" in argv
            else "development"
        )
        if purpose != "confirmation":
            continue
        for required in ("--benchmark_manifest", "--campaign_manifest"):
            assert required in argv, f"{location}: missing {required}"
        stage = argv[argv.index("--stage") + 1]
        if stage == "grpo":
            assert "--checkpoint_selection_receipt" in argv, location
        for required_flag in (
            "--save_results",
            "--save_details",
            "--resume",
            "--no-prefer_subset_files",
        ):
            assert required_flag in argv, f"{location}: missing {required_flag}"
        assert argv[argv.index("--split") + 1] == "eval", location
        assert argv[argv.index("--eval_start_index") + 1] == "0", location
        assert argv[argv.index("--max_eval_samples_per_task") + 1] == "0", location


def test_current_grpo_lane_is_not_documented_as_foarl() -> None:
    baseline = (ROOT / "docs" / "llmcosolver_f638_corrected_baseline.md").read_text(
        encoding="utf-8"
    )
    assert "`METHOD=grpo` lane is also not FOARL" in baseline
    assert "uses three strict heads and `dr_grpo`" in baseline
    assert "Official-code operational semantics" in baseline
    assert "Scientifically corrected frozen step-zero SFT reference" in baseline

    protocol = (ROOT / "experiments" / "sota_20260814" / "PROTOCOL.md").read_text(encoding="utf-8")
    assert "ACCORD has an official code release" in protocol
    assert "`METHOD=grpo` lane is not an LLMCoSolver/FOARL reproduction" in protocol
