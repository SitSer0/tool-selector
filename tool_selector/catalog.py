import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    # Search representation stays internal; API returns only callable definitions.
    retrieval_text: str = Field(min_length=1, exclude=True)

    @field_validator("input_schema")
    @classmethod
    def validate_schema(cls, value):
        validator_for(value).check_schema(value)
        if value.get("type") != "object":
            raise ValueError("input_schema must describe an object")
        return value


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return rows


def load_catalog(path: str | Path) -> list[Tool]:
    tools = [Tool.model_validate(row) for row in read_jsonl(path)]
    if not tools:
        raise ValueError("Catalog must not be empty")
    for field in ("tool_id", "name"):
        values = [getattr(tool, field) for tool in tools]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field} in catalog")
    if any(not tool.retrieval_text.strip() for tool in tools):
        raise ValueError("Search descriptions must not be blank")
    return tools


def catalog_version(tools: list[Tool]) -> str:
    rows = [{**tool.model_dump(), "retrieval_text": tool.retrieval_text} for tool in tools]
    data = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()
