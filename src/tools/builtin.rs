pub use crate::tools::file::{
    ApplyPatchArgs, ReadArgs, RgArgs, apply_patch_tool, read_tool, ripgrep_tool,
};

use crate::state::AgentState;

pub fn builtin_registry(state: AgentState) -> crate::ToolRegistry {
    crate::ToolRegistry::new()
        .with_tool(read_tool(state.clone()))
        .with_tool(apply_patch_tool(state.clone()))
        .with_tool(ripgrep_tool(state))
}
