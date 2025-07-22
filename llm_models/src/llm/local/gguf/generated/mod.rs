pub mod organizations;
pub mod preset_models;

use std::borrow::Cow;

use crate::llm::{
    base::LlmModelBase,
    local::{
        gguf::gguf_preset::{GgufPreset, GgufPresetConfig, GgufPresetQuant},
        organizations::LocalLlmOrganization,
    },
};
