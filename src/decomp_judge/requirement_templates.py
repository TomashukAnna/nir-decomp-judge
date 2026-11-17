"""Reusable checks and traceability validation for requirement templates."""

import argparse
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from decomp_judge.requirement_similarity import catalog


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    criterion: str
    applies_when: str
    source_ids: list[str] = Field(min_length=1)
    rule: Literal["code", "llm"]
    parameters: dict[str, str]
    checker: str | None = None
    instructions: str | None = None


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


def load_bank(path: Path, source: Path) -> list[Template]:
    templates = [
        Template.model_validate(row) for row in json.loads(path.read_text(encoding="utf-8"))
    ]
    known_ids = {row["global_id"] for row in catalog(source)}
    if not templates or len({item.id for item in templates}) != len(templates):
        raise ValueError("Bank needs unique template IDs")
    for item in templates:
        if (
            len(set(item.source_ids)) != len(item.source_ids)
            or not set(item.source_ids) <= known_ids
        ):
            raise ValueError(f"Invalid source IDs for {item.id}")
        if item.rule == "code":
            if item.checker != "check_background_color" or item.instructions is not None:
                raise ValueError(f"Invalid checker for {item.id}")
        elif not item.instructions or item.checker is not None:
            raise ValueError(f"Invalid LLM instructions for {item.id}")
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("configs/requirement-bank.json"))
    parser.add_argument("--source", type=Path, default=Path("data/requirements"))
    args = parser.parse_args()
    print(f"Validated {len(load_bank(args.bank, args.source))} requirement templates")


if __name__ == "__main__":
    main()
