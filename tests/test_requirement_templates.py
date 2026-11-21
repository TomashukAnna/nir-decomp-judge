"""Check the first reusable rule and its links to the annotated requirements."""

from pathlib import Path

import pytest

from decomp_judge.requirement_templates import check_background_color, load_bank


@pytest.mark.parametrize(
    ("observed", "verdict"),
    [
        (["rgb(245, 245, 220)", "rgb(245,245,220)"], "PASS"),
        (["rgb(245, 245, 220)", "rgb(255, 255, 255)"], "FAIL"),
        ([], "UNRESOLVED"),
        (["rgba(0, 0, 0, 0)"], "UNRESOLVED"),
    ],
)
def test_background_rule(observed: list[str], verdict: str) -> None:
    assert check_background_color("#F5F5DC", observed) == verdict


def test_template_bank_links_to_existing_annotations() -> None:
    root = Path(__file__).resolve().parents[1]
    templates = load_bank(root / "configs/requirement-bank.json", root / "data/requirements")
    assert {item.rule for item in templates} == {"code", "llm"}
