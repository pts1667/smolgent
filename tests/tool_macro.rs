use schemars::JsonSchema;
use serde::Deserialize;
use smolgent::{ToolCall, ToolCallFunction, ToolRegistry};

/// Multiply two integers.
#[smolgent::tool]
fn multiply(a: i64, b: i64) -> i64 {
    a * b
}

#[derive(Clone)]
struct Offset(i64);

#[derive(Deserialize, JsonSchema)]
/// Arguments for offset addition.
struct AddArgs {
    /// Value to add to the captured offset.
    value: i64,
}

const ADD_DESCRIPTION: &str = "Add a value to a captured offset.";

#[smolgent::tool(
    description = ADD_DESCRIPTION,
    fallible,
    factory_visibility = "pub"
)]
fn add_offset(
    #[tool(context)] offset: &Offset,
    #[tool(arguments)] args: AddArgs,
) -> smolgent::Result<i64> {
    if args.value < 0 {
        return Err(smolgent::Error::Tool("value must be non-negative".into()));
    }
    Ok(offset.0 + args.value)
}

#[derive(Deserialize, JsonSchema)]
#[smolgent::tool_definition(
    name = "inspect_value",
    description = "Inspect a value.",
    function = "inspect_value_definition",
    visibility = "pub"
)]
struct InspectValueArgs {
    /// Value that should be inspected.
    #[allow(dead_code)]
    value: String,
}

#[tokio::test]
async fn tool_macro_uses_rustdoc_and_executes_through_registry() {
    let tool = multiply_tool();
    assert_eq!(tool.definition().function.name, "multiply");
    assert_eq!(
        tool.definition().function.description,
        "Multiply two integers."
    );
    assert!(
        tool.definition().function.parameters["properties"]
            .as_object()
            .unwrap()
            .contains_key("a")
    );

    let registry = ToolRegistry::new().with_tool(tool);
    let result = registry
        .execute_call(&ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: "multiply".to_string(),
                arguments: r#"{"a":6,"b":7}"#.to_string(),
            },
        })
        .await
        .unwrap();

    assert_eq!(result.content, "42");
}

#[tokio::test]
async fn tool_macro_supports_context_argument_structs_and_fallible_handlers() {
    let tool = add_offset_tool(Offset(10));
    let definition = add_offset_tool_definition();

    assert_eq!(definition.function.description, ADD_DESCRIPTION);
    assert_eq!(
        definition.function.parameters["description"],
        "Arguments for offset addition."
    );
    assert_eq!(
        definition.function.parameters["properties"]["value"]["description"],
        "Value to add to the captured offset."
    );
    assert!(
        definition.function.parameters["properties"]
            .get("offset")
            .is_none()
    );

    assert_eq!(
        tool.call(serde_json::json!({"value": 5})).await.unwrap(),
        15
    );
    let error = tool
        .call(serde_json::json!({"value": -1}))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("value must be non-negative"));
}

#[test]
fn definition_macro_uses_argument_field_rustdoc() {
    let definition = inspect_value_definition();

    assert_eq!(definition.function.name, "inspect_value");
    assert_eq!(
        definition.function.parameters["properties"]["value"]["description"],
        "Value that should be inspected."
    );
}
