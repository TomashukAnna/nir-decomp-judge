"""Reusable checks and traceability validation for requirement templates."""

import argparse
import json
import re
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decomp_judge.requirement_similarity import catalog


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
    criterion: str = Field(min_length=1)
    applies_when: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    rule: Literal["code", "llm"]
    parameters: dict[str, str]
    checker: str | None = None
    instructions: str | None = None
    checker_status: Literal["implemented", "planned", "not_applicable"] | None = None
    validation_status: Literal["not_evaluated"] = "not_evaluated"
    source_notes: dict[str, str] = Field(default_factory=dict)
    specializes: str | None = None

    @field_validator("criterion", "applies_when", "instructions")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Template text must not be blank")
        return value

    @model_validator(mode="after")
    def check_protocol(self) -> Self:
        if len(set(self.source_ids)) != len(self.source_ids) or any(
            not source.strip() for source in self.source_ids
        ):
            raise ValueError("Source IDs must be nonblank and unique")
        if any(not key.strip() or not value.strip() for key, value in self.parameters.items()):
            raise ValueError("Parameter names and descriptions must not be blank")
        if not set(self.source_notes) <= set(self.source_ids) or any(
            not note.strip() for note in self.source_notes.values()
        ):
            raise ValueError("Source notes must refer to listed source IDs")
        # Accept first-version cards without an explicit implementation status.
        if self.checker_status is None:
            self.checker_status = (
                "not_applicable"
                if self.rule == "llm"
                else "implemented"
                if self.checker
                else "planned"
            )
        if self.rule == "llm":
            if self.checker is not None or self.checker_status != "not_applicable":
                raise ValueError("LLM templates must not declare a code checker")
            if not self.instructions:
                raise ValueError("LLM templates need verification instructions")
        elif self.checker_status == "implemented":
            if self.checker not in {"check_background_color", "check_component_color"}:
                raise ValueError("Unknown implemented checker")
            if not {"expected_color", "observed_colors"} <= self.parameters.keys():
                raise ValueError("Color checkers need expected_color and observed_colors")
        elif self.checker_status == "planned":
            if self.checker is not None or not self.instructions:
                raise ValueError(
                    "Planned checks need instructions and must not claim executable code"
                )
        else:
            raise ValueError("Code templates need implemented or planned status")
        return self


def check_background_color(
    expected_color: str, observed_colors: list[str]
) -> Literal["PASS", "FAIL", "UNRESOLVED"]:
    """Compare computed browser colors for the previously identified background region."""
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", expected_color):
        raise ValueError("expected_color must be #RRGGBB")
    expected = tuple(int(expected_color[i : i + 2], 16) for i in (1, 3, 5))
    if not observed_colors:
        return "UNRESOLVED"
    for color in observed_colors:
        match = re.fullmatch(r"rgb\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)", color)
        if not match or any(int(channel) > 255 for channel in match.groups()):
            return "UNRESOLVED"
        if tuple(map(int, match.groups())) != expected:
            return "FAIL"
    return "PASS"


def check_component_color(
    expected_color: str, observed_colors: list[str]
) -> Literal["PASS", "FAIL", "UNRESOLVED"]:
    """Compare computed colors of the component regions selected from the specification."""
    return check_background_color(expected_color, observed_colors)


def load_bank(path: Path, source: Path) -> list[Template]:
    templates = [
        Template.model_validate(row) for row in json.loads(path.read_text(encoding="utf-8"))
    ]
    if any((source / "train").glob("*.json")):
        from decomp_judge.bank_candidates import build_catalog

        rows, _ = build_catalog(source.resolve().parents[1])
        known_ids = {row.global_id for row in rows}
    else:
        known_ids = {row["global_id"] for row in catalog(source)}
    if not templates or len({item.id for item in templates}) != len(templates):
        raise ValueError("Bank needs unique template IDs")
    for item in templates:
        if (
            len(set(item.source_ids)) != len(item.source_ids)
            or not set(item.source_ids) <= known_ids
        ):
            raise ValueError(f"Invalid source IDs for {item.id}")
        if item.specializes is not None and (
            item.specializes == item.id or item.specializes not in {card.id for card in templates}
        ):
            raise ValueError(f"Invalid specialization for {item.id}")
    tuple(
        TopologicalSorter(
            {item.id: [item.specializes] if item.specializes else [] for item in templates}
        ).static_order()
    )
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("configs/requirement-bank.json"))
    parser.add_argument("--source", type=Path, default=Path("data/requirements"))
    args = parser.parse_args()
    print(f"Validated {len(load_bank(args.bank, args.source))} requirement templates")


if __name__ == "__main__":
    main()
