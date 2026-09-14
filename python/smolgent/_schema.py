"""JSON Schema inference for keyword-based Python tools."""

from __future__ import annotations

import inspect
import types
from typing import Annotated, Any, Callable, Literal, Union, get_args, get_origin, get_type_hints


def _type_schema(annotation: Any) -> dict[str, Any]:
    if annotation is Any:
        return {}
    for python_type, json_type in ((str, "string"), (int, "integer"),
                                   (float, "number"), (bool, "boolean"),
                                   (type(None), "null")):
        if annotation is python_type:
            return {"type": json_type}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _type_schema(args[0])
    if origin in (Union, types.UnionType):
        return {"anyOf": [_type_schema(arg) for arg in args]}
    if origin is Literal:
        if not args or any(type(value) not in (str, int, bool, type(None)) for value in args):
            raise TypeError("Literal values must be strings, integers, booleans, or None")
        return {"enum": list(args)}
    if annotation is list or origin is list:
        return {"type": "array", "items": _type_schema(args[0]) if args else {}}
    if annotation is dict or origin is dict:
        if args and args[0] is not str:
            raise TypeError("dictionary keys must be annotated as str")
        return {"type": "object",
                "additionalProperties": _type_schema(args[1]) if args else {}}
    raise TypeError(f"unsupported type annotation {annotation!r}")


def infer_parameters(handler: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(handler)
    for parameter in signature.parameters.values():
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY):
            raise TypeError(
                f"cannot infer tool parameter {parameter.name!r}: tools receive named "
                "arguments; positional-only parameters, *args, and **kwargs are unsupported. "
                "Use a wrapper with named parameters or supply parameters= explicitly"
            )
        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(f"tool parameter {parameter.name!r} needs a type annotation; "
                            "add one or supply parameters= explicitly")

    # Resolve only argument hints, so return annotations cannot affect the input
    # schema. A function carrier also avoids Python 3.10's implicit Optional
    # conversion for parameters whose default is None in get_type_hints().
    def argument_hints():
        pass

    argument_hints.__annotations__ = {
        name: parameter.annotation for name, parameter in signature.parameters.items()
    }
    try:
        hints = get_type_hints(argument_hints,
                               globalns=getattr(inspect.unwrap(handler), "__globals__", {}),
                               include_extras=True)
    except (NameError, TypeError, SyntaxError) as error:
        raise TypeError("cannot resolve tool argument type hints; use types available in "
                        "the function's module or supply parameters= explicitly") from error

    properties = {}
    required = []
    for name, parameter in signature.parameters.items():
        try:
            properties[name] = _type_schema(hints[name])
        except TypeError as error:
            raise TypeError(f"cannot infer tool parameter {name!r}: {error}; "
                            "supply parameters= explicitly") from error
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}
