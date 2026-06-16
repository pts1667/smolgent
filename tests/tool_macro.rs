use smolgent::{ToolCall, ToolCallFunction, ToolRegistry};

/// Multiply two integers.
#[smolgent::tool]
fn multiply(a: i64, b: i64) -> i64 {
    a * b
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
