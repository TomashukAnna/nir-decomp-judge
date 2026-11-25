"""Check the first reusable rule and its links to the annotated requirements."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from decomp_judge.requirement_templates import (
    Template,
    check_background_color,
    check_component_color,
    load_bank,
)


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


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        (["rgb(34, 139, 34)"], "PASS"),
        (["rgb(34, 139, 34)", "rgb(255, 255, 255)"], "FAIL"),
        ([], "UNRESOLVED"),
        (["rgba(34, 139, 34, 0.5)"], "UNRESOLVED"),
    ],
)
def test_component_colors(observations: list[str], expected: str) -> None:
    assert check_component_color("#228B22", observations) == expected


def planned_card() -> dict[str, Any]:
    return {
        "id": "form.required_fields",
        "criterion": "Required fields are checked.",
        "applies_when": "Required fields are specified.",
        "source_ids": ["webgen-026-R2.1"],
        "rule": "code",
        "parameters": {"required_fields": "Fields from the specification."},
        "checker_status": "planned",
        "instructions": "Compare observed blocked submissions with required fields.",
    }


def test_planned_checks_cannot_claim_an_implemented_function() -> None:
    card = planned_card()
    parsed = Template.model_validate(card)
    assert parsed.checker is None
    assert parsed.checker_status == "planned"
    assert parsed.validation_status == "not_evaluated"
    with pytest.raises(ValidationError):
        Template.model_validate({**card, "checker": "check_background_color"})
    with pytest.raises(ValidationError):
        Template.model_validate({**card, "checker_status": "implemented"})


@pytest.mark.parametrize(
    "changes",
    [
        {"source_ids": ["webgen-026-R2.1", "webgen-026-R2.1"]},
        {"source_notes": {"unknown-source": "Not in source_ids"}},
        {"parameters": {"field": " "}},
        {"instructions": " "},
        {"validation_status": "validated"},
        {"rule": "llm", "checker_status": "implemented", "checker": "check_component_color"},
    ],
)
def test_invalid_template_metadata_is_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Template.model_validate({**planned_card(), **changes})


def test_first_version_color_card_still_loads() -> None:
    card = planned_card()
    card.update(
        {
            "id": "ui.background_color",
            "checker": "check_background_color",
            "parameters": {"expected_color": "HEX", "observed_colors": "Computed RGB"},
        }
    )
    del card["checker_status"]
    del card["instructions"]
    assert Template.model_validate(card).checker_status == "implemented"


def test_bank_schema_matches_model() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "configs/requirement-bank.schema.json").read_text("utf-8"))
    assert schema == TypeAdapter(list[Template]).json_schema()


@pytest.mark.parametrize("invalid", ["unknown_source", "missing_parent", "cycle"])
def test_broken_bank_links_are_rejected(tmp_path: Path, invalid: str) -> None:
    card = planned_card()
    other = {**planned_card(), "id": "form.other"}
    if invalid == "unknown_source":
        card["source_ids"] = ["webgen-train-999999-R1"]
    elif invalid == "missing_parent":
        card["specializes"] = "form.nonexistent"
    else:
        card["specializes"] = other["id"]
        other["specializes"] = card["id"]
    path = tmp_path / "bank.json"
    path.write_text(json.dumps([card, other]), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError):
        load_bank(path, root / "data/requirements")
