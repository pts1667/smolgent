"""Type-hint inference and decorator compatibility, without provider requests."""

from __future__ import annotations

import asyncio
import unittest
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from smolgent import Tool, tool


class ToolSchemaTests(unittest.TestCase):
    def test_bare_decorator_and_primitive_arguments(self):
        @tool
        def calculate(a: int, b: float, label: str, enabled: bool) -> object:
            """Calculate a result."""
            return a + b

        self.assertIsInstance(calculate, Tool)
        self.assertEqual(calculate.name, "calculate")
        self.assertEqual(calculate.description, "Calculate a result.")
        self.assertEqual(calculate.parameters, {
            "type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "number"},
                "label": {"type": "string"}, "enabled": {"type": "boolean"},
            }, "required": ["a", "b", "label", "enabled"], "additionalProperties": False,
        })

    def test_defaults_keyword_only_and_nullable_required(self):
        @tool(name="lookup", description="Custom description")
        def search(query: str | None, limit: int = 10, *, tags: list[str] | None = None):
            return [query, limit, tags]

        self.assertEqual(search.name, "lookup")
        self.assertEqual(search.description, "Custom description")
        self.assertEqual(search.parameters["required"], ["query"])
        self.assertEqual(search.parameters["properties"]["query"],
                         {"anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(asyncio.run(search._invoke('{"query": null}')), '[null, 10, null]')

    def test_nested_types_and_typing_equivalents(self):
        @tool()
        def store(rows: List[Dict[str, Union[int, str]]],
                  labels: list[dict[str, int | str]], choice: Literal["a", "b"],
                  flag: Literal[True, 1, None], value: Optional[float],
                  metadata: Annotated[str, {"ignored": True}],
                  anything: Any, items: list, mapping: dict, empty: None):
            pass

        properties = store.parameters["properties"]
        self.assertEqual(properties["rows"], {"type": "array", "items": {
            "type": "object", "additionalProperties": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
            },
        }})
        self.assertEqual(properties["rows"], properties["labels"])
        self.assertEqual(properties["choice"], {"enum": ["a", "b"]})
        self.assertEqual(properties["flag"], {"enum": [True, 1, None]})
        self.assertEqual(properties["value"],
                         {"anyOf": [{"type": "number"}, {"type": "null"}]})
        self.assertEqual(properties["metadata"], {"type": "string"})
        self.assertEqual(properties["anything"], {})
        self.assertEqual(properties["items"], {"type": "array", "items": {}})
        self.assertEqual(properties["mapping"], {"type": "object", "additionalProperties": {}})
        self.assertEqual(properties["empty"], {"type": "null"})

    def test_no_arguments_and_async_function(self):
        @tool()
        async def ready() -> bool:
            return True

        self.assertEqual(ready.parameters, {"type": "object", "properties": {},
                                           "required": [], "additionalProperties": False})
        self.assertEqual(asyncio.run(ready._invoke('{}')), 'true')

    def test_return_annotation_is_not_resolved(self):
        @tool
        def echo(value: "int") -> "UnknownReturnType":
            return value

        self.assertEqual(echo.parameters["properties"], {"value": {"type": "integer"}})

    def test_explicit_schema_overrides_annotations(self):
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"]}

        @tool(parameters=schema)
        def echo(value: "UnresolvableType"):
            return value

        self.assertIs(echo.parameters, schema)
        self.assertEqual(asyncio.run(echo._invoke('{"value": "hello"}')), '"hello"')

        @tool(parameters=schema)
        def untyped(value):
            return value

        self.assertIs(untyped.parameters, schema)

    def test_missing_hint_requires_annotation_or_explicit_schema(self):
        def missing(value=42):
            return value

        with self.assertRaisesRegex(TypeError, "'value'.*type annotation.*parameters="):
            tool(missing)

    def test_unsupported_types_identify_parameter(self):
        for annotation in (bytes, tuple[int, ...], set[str], dict[int, str],
                           list[complex], Literal[1.5]):
            def unsupported(value):
                return value
            unsupported.__annotations__ = {"value": annotation}
            with self.subTest(annotation=annotation):
                with self.assertRaisesRegex(TypeError, "'value'.*parameters="):
                    tool(unsupported)

    def test_unresolved_argument_hint_is_actionable(self):
        def unresolved(value: "UnknownArgumentType"):
            return value

        with self.assertRaisesRegex(TypeError, "cannot resolve.*parameters="):
            tool(unresolved)

    def test_incompatible_signatures(self):
        def positional(value: int, /):
            pass

        def variadic(*values: int):
            pass

        def keywords(**values: int):
            pass

        for handler in (positional, variadic, keywords):
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(TypeError, "named arguments"):
                    tool(handler)


if __name__ == "__main__":
    unittest.main()
