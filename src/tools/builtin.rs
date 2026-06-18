pub use crate::tools::file::{
    ApplyPatchArgs, CreateFileArgs, DeleteFileArgs, ReadArgs, RgArgs, apply_patch_tool,
    create_file_tool, delete_file_tool, read_tool, ripgrep_tool,
};

use crate::state::AgentState;

/// Default built-in registry for filesystem-oriented agents.
///
/// Includes read/search tools for readable roots and create/delete/patch tools for writable roots.
pub fn builtin_registry(state: AgentState) -> crate::ToolRegistry {
    crate::ToolRegistry::new()
        .with_tool(read_tool(state.clone()))
        .with_tool(create_file_tool(state.clone()))
        .with_tool(delete_file_tool(state.clone()))
        .with_tool(apply_patch_tool(state.clone()))
        .with_tool(ripgrep_tool(state))
}
