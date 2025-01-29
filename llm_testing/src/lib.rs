#[allow(unused_imports)]
pub use anyhow::{anyhow, bail, Result};

pub use llm_client::*;

pub use backends::*;
pub use test_loader::*;
pub use test_types::*;
#[allow(unused_imports)]
pub use tracing::{debug, error, info, span, trace, warn, Level};

pub mod backends;
pub mod speed_bench;
pub mod test_loader;
pub mod test_types;

const PRINT_PRIMITIVE_RESULT: bool = true;
const PRINT_WORKFLOW_RESULT: bool = true;
const PRINT_PROMPT: bool = true;

pub fn print_results<T: std::fmt::Debug>(
    prompt: &LlmPrompt,
    workflow_result: &Option<impl std::fmt::Display>,
    primitive_result: &Option<T>,
) {
    if PRINT_PROMPT {
        println!("{}", prompt);
    }
    if PRINT_WORKFLOW_RESULT {
        if let Some(result) = workflow_result {
            println!("{}", result);
        }
    }
    if PRINT_PRIMITIVE_RESULT {
        println!("{:?}", primitive_result);
    }
}
