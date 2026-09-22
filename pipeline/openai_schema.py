"""
openai_schema.py
Convert a Pydantic-generated JSON schema into OpenAI strict-mode form.

OpenAI structured outputs (`response_format.json_schema.strict = true`) reject
schemas unless every object node has:
  - additionalProperties: false
  - required: [every property name]
and they do not accept `default` keywords or siblings next to `$ref`.
A non-conforming schema makes the Batch API fail every item with HTTP 400.

Pydantic emits all three problems for ordinary models:
  - Optional fields  -> anyOf [...] plus "default": null
  - nested models    -> {"$ref": ..., "description": ...}

Invariant: pure function, never mutates its input.
"""

from __future__ import annotations

import copy

_NESTED_KEYS = ("anyOf", "allOf", "oneOf", "prefixItems")


def make_openai_strict(schema: dict) -> dict:
    """Return a strict-mode copy of `schema` (see module docstring for the rules)."""
    return _strict(copy.deepcopy(schema))


def _strict(node: dict) -> dict:
    if "$ref" in node:
        return {"$ref": node["$ref"]}

    node.pop("default", None)

    if "$defs" in node:
        node["$defs"] = {k: _strict(v) for k, v in node["$defs"].items()}

    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
        node["properties"] = {k: _strict(v) for k, v in node["properties"].items()}

    if isinstance(node.get("items"), dict):
        node["items"] = _strict(node["items"])

    for key in _NESTED_KEYS:
        if key in node:
            node[key] = [_strict(v) for v in node[key]]

    return node
