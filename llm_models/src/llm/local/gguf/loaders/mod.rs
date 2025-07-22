pub mod hf;
pub mod local;
pub mod preset;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    hf_loader::{HfTokenTrait, HuggingFaceLoader},
    llm::{
        base::{LlmModelBase, DEFAULT_CONTEXT_LENGTH},
        local::{
            chat_template::LlmChatTemplate,
            gguf::{
                gguf_model::{GgufModel, GgufQuant},
                loaders::local::GgufLocalLoader,
            },
            metadata::LocalLlmMetadata,
            organizations::LocalLlmOrganization,
        },
    },
};
