"""Shared input and evidence protocol for static requirements judges."""

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

RequirementId = StrictStr | StrictInt


class Requirement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    requirement_id: RequirementId
    criteria: str = Field(min_length=1)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str = Field(min_length=1)
    line_start: StrictInt = Field(ge=1)
    line_end: StrictInt = Field(ge=1)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: RequirementId
    verdict: Literal["PASS", "FAIL", "NA"]
    evidence: list[Evidence]
    reason: str

    @field_validator("reason")
    @classmethod
    def one_line(cls, text: str) -> str:
        if not text.strip() or "\n" in text or "\r" in text:
            raise ValueError("reason must be one nonempty line")
        return text


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[Verdict] = Field(min_length=1)


def load_requirements(path: Path) -> list[Requirement]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Requirements must be a nonempty JSON array")
    requirements = [Requirement.model_validate(item) for item in data]
    ids = [(type(r.requirement_id), r.requirement_id) for r in requirements]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate requirement IDs")
    return requirements


def normalize_response(raw: str) -> tuple[str, bool]:
    """Remove only one whole-response Markdown fence, never repair JSON content."""
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", raw.strip(), flags=re.DOTALL)
    if match is None:
        return raw, False
    body = match.group(1)
    if any(line.lstrip().startswith("```") for line in body.splitlines()):
        return raw, False
    return body, True


def validate_judgment(raw: str, requirements: list[Requirement], app: Path) -> Judgment:
    result = Judgment.model_validate_json(raw)
    expected = {(type(r.requirement_id), r.requirement_id) for r in requirements}
    actual = [(type(v.requirement_id), v.requirement_id) for v in result.verdicts]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Return each input requirement ID exactly once, preserving its type")
    root = app.resolve()
    for verdict in result.verdicts:
        if verdict.verdict == "PASS" and not verdict.evidence:
            raise ValueError("PASS requires evidence")
        for evidence in verdict.evidence:
            name = PurePosixPath(evidence.file)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in evidence.file
                or ":" in evidence.file
                or ".git" in name.parts
                or name.name == ".env"
                or name.name.startswith(".env.")
            ):
                raise ValueError("Evidence path must be relative and inside the application")
            target = (root / evidence.file).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError("Evidence file does not exist inside the application")
            lines = len(target.read_text(encoding="utf-8").splitlines())
            if not 1 <= evidence.line_start <= evidence.line_end <= lines:
                raise ValueError("Evidence line range is outside the file")
    return result
