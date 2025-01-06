mod api_backends;
mod basic_completion_tests;
mod basic_primitive_tests;
mod decision_tests;
mod extract_tests;
mod llama_cpp;
#[cfg(feature = "mistral_rs_backend")]
mod mistral_rs;
mod reason_tests;

use llm_client::prelude::*;
use serial_test::serial;
