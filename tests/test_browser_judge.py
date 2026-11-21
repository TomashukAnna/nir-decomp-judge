"""Exercise execution evidence, bounded feedback and browser-first routing without an API."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from decomp_judge import browser_judge as judge
from decomp_judge.judge_validator import Requirement


def verdict(rid: int, status: str, evidence: list[int]) -> dict[str, Any]:
    return {
        "requirement_id": rid,
        "verdict": status,
        "reason": "Observed result",
        "evidence": evidence,
    }


def dialogue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replies: list[str],
    observations: list[dict[str, Any]],
    *,
    mode: str = "command",
    requirement_ids: tuple[int, ...] = (0,),
) -> tuple[list[judge.FunctionalVerdict], list[list[dict[str, str]]]]:
    requests = []
    responses = iter(replies)

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        requests.append([dict(row) for row in args[1]])
        return next(responses), {"tokens": {}, "finish_reason": "length"}

    monkeypatch.setattr(judge, "call_model", call)
    result = judge.functional_dialogue(
        tmp_path,
        tmp_path,
        "Specification",
        [Requirement(requirement_id=rid, criteria=f"Criterion {rid}") for rid in requirement_ids],
        mode,
        ["python", "train.py"],
        observations,
        {},
        None,
        "",
        {},
        lambda usage: None,
        "test-key",
        judge.FunctionalSettings(max_actions=2),
    )
    return result, requests


def test_repair_preserves_accepted_failure_when_another_verdict_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = json.dumps({"verdicts": [verdict(0, "FAIL", [0]), verdict(1, "PASS", [1])]})
    result, requests = dialogue(
        tmp_path,
        monkeypatch,
        [response] * 4,
        [
            {"kind": "python_launch", "launch": {"exit_code": 1}},
            {"kind": "source", "text": "implementation"},
        ],
        requirement_ids=(0, 1),
    )
    assert [row.verdict for row in result] == ["FAIL", "UNRESOLVED"]
    assert len(requests) == 4
    assert "Requirement 1" in requests[1][-1]["content"]
    session = json.loads((tmp_path / "functional-session.json").read_text())
    assert session["repairs"] == 3
    assert session["accepted_ids"] == [0]


def test_empty_and_non_json_responses_are_returned_for_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = json.dumps({"verdicts": [verdict(0, "FAIL", [0])]})
    result, requests = dialogue(
        tmp_path,
        monkeypatch,
        ["I will inspect files", "  ", response],
        [
            {"kind": "python_launch", "launch": {"exit_code": 1}},
        ],
    )
    assert result[0].verdict == "FAIL"
    assert len(requests) == 3
    assert "Empty model response; finish_reason=length" in requests[2][-1]["content"]
    assert (tmp_path / "response-1.txt").read_text() == "  "
    assert json.loads((tmp_path / "response-1-usage.json").read_text())["finish_reason"] == "length"


def test_web_action_can_produce_functional_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = []

    def execute(response: dict[str, Any], *args: Any) -> dict[str, Any]:
        actions.append(response)
        return {"kind": "browser", "matched": True, "text": "Title is required"}

    monkeypatch.setattr(judge, "execute_functional_action", execute)
    result, _ = dialogue(
        tmp_path,
        monkeypatch,
        [
            json.dumps({"action": "step", "step": {"op": "click", "selector": "button"}}),
            json.dumps({"verdicts": [verdict(0, "PASS", [1])]}),
        ],
        [{"kind": "browser", "text": "Form"}],
        mode="web",
    )
    assert result[0].verdict == "PASS"
    assert actions[0]["action"] == "step"


@pytest.mark.parametrize(
    "observation,status,expected",
    [
        ({"kind": "source"}, "PASS", False),
        ({"kind": "validation_error"}, "FAIL", False),
        ({"kind": "browser", "matched": False}, "PASS", False),
        ({"kind": "browser", "matched": False}, "FAIL", True),
        ({"kind": "python_launch", "launch": {"exit_code": 1}}, "PASS", False),
        ({"kind": "python_launch", "launch": {"exit_code": 1}}, "FAIL", True),
        ({"kind": "probe", "exit_code": 0}, "PASS", True),
        ({"kind": "probe", "error": "Timeout"}, "FAIL", False),
    ],
)
def test_evidence_must_match_execution_outcome(
    observation: dict[str, Any],
    status: str,
    expected: bool,
) -> None:
    assert judge.supports_verdict(observation, status) == expected


@pytest.mark.parametrize("ids", [[], [0, 0], [0, 2], [True, 1]])
def test_missing_duplicate_unknown_and_boolean_ids_are_rejected(ids: list[Any]) -> None:
    with pytest.raises(ValueError):
        judge.validate_functional_response(
            {"verdicts": [verdict(rid, "UNRESOLVED", []) for rid in ids]}, {0, 1}, []
        )


def test_web_cannot_replace_execution_with_source_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Use browser actions"):
        judge.execute_functional_action(
            {"action": "inspect", "path": "app.py"},
            "web",
            tmp_path,
            {},
            tmp_path,
            0,
            object(),
            "http://127.0.0.1:8000",
        )


def test_python_probe_executes_an_independent_component(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("def predict(value):\n    return value * 2\n")
    observation = judge.execute_functional_action(
        {"action": "probe", "module": "model", "function": "predict", "args": [3]},
        "command",
        tmp_path,
        {"model.py": "hash"},
        tmp_path,
        0,
        None,
        "",
    )
    assert observation["exit_code"] == 0
    assert observation["output_tail"].splitlines() == ["int", "6"]
    assert judge.supports_verdict(observation, "PASS")


@pytest.mark.skipif(os.environ.get("RUN_BROWSER_SMOKE") != "1", reason="Optional local Edge smoke")
def test_real_browser_reads_controls_and_executes_a_check(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(channel="msedge")
        try:
            page = browser.new_page()
            page.set_content(
                '<body style="background: ivory"><form><input name="title" required>'
                "<button>Submit</button></form></body>"
            )
            snapshot = judge.page_snapshot(page)
            assert any(control["name"] == "title" for control in snapshot["controls"])
            result = judge.execute_functional_action(
                {"action": "styles", "selector": "body", "property": "background-color"},
                "web",
                tmp_path,
                {},
                tmp_path,
                0,
                page,
                "",
            )
            assert result["values"] == ["rgb(255, 255, 240)"]
            checked = judge.execute_functional_action(
                {"action": "step", "step": {"op": "fill", "selector": "input", "value": "Sale"}},
                "web",
                tmp_path,
                {},
                tmp_path,
                1,
                page,
                "",
            )
            assert checked["matched"] is True
            assert page.locator("input").input_value() == "Sale"
        finally:
            browser.close()
