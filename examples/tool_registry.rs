use smolgent::{ToolCall, ToolCallFunction, ToolRegistry};

/// Add two integers.
#[smolgent::tool]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[tokio::main]
async fn main() -> smolgent::Result<()> {
    let registry = ToolRegistry::new().with_tool(add_tool());
    let call = ToolCall {
        id: "call_1".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunction {
            name: "add".to_string(),
            arguments: r#"{"a":2,"b":3}"#.to_string(),
        },
    };

    let result = registry.execute_call(&call).await?;
    println!("{}", result.content);

    Ok(())
}
