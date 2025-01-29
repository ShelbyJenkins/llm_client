// Public modules
pub mod chat_template;
pub mod gguf;
pub mod hf_loader;
pub mod metadata;

// Internal imports
use super::LlmModelBase;
use crate::LlmChatTemplate;
use gguf::GgufLoader;
use metadata::LocalLlmMetadata;

pub struct LocalLlmModel {
    pub model_base: LlmModelBase,
    pub local_model_path: std::path::PathBuf,
    pub model_metadata: LocalLlmMetadata,
    pub chat_template: LlmChatTemplate,
}

impl Default for LocalLlmModel {
    fn default() -> Self {
        let mut loader = GgufLoader::default();
        loader.load().expect("Failed to load LlmPreset")
    }
}

impl std::fmt::Debug for LocalLlmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("LocalLlmModel");
        debug_struct.field("model_id", &self.model_base.model_id);
        debug_struct.field("local_model_path", &self.local_model_path);
        debug_struct.field("model_metadata", &self.model_metadata);
        debug_struct.field("chat_template", &self.chat_template);
        debug_struct.finish()
    }
}
