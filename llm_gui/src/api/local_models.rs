use dioxus::fullstack::prelude::*;

#[server]
pub async fn get_models() -> Result<Vec<LlmGuiLocalModel>, ServerFnError> {
    server::get_models()
}

#[server]
pub async fn get_default_model_id() -> Result<String, ServerFnError> {
    Ok(server::default_model_id())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ModelState {
    Selected(LoadState),
    Downloaded,
    Downloadable,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum LoadState {
    Unloaded,
    Loaded,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LlmGuiModelBase {
    pub model_id: String,
    pub friendly_name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LlmGuiLocalModel {
    pub model_state: ModelState,
    pub base: LlmGuiModelBase,
    pub number_of_parameters: Option<f64>,
    pub organization: Option<String>,
    pub model_repo_link: Option<String>,
    pub gguf_repo_link: Option<String>,
    pub recommended_quantization_level: Option<u8>,
    pub quants: Vec<LlmGuiLocalQuant>,
}

impl LlmGuiLocalModel {
    pub fn organization(&self) -> &str {
        self.organization
            .as_deref()
            .unwrap_or("unknown organization")
    }

    pub fn number_of_parameters(&self) -> String {
        if let Some(params) = self.number_of_parameters {
            format!("{}b", params)
        } else {
            "".to_string()
        }
    }

    pub fn model_repo_link(&self) -> String {
        if let Some(model_repo_link) = &self.model_repo_link {
            format!("https://huggingface.co/{}", model_repo_link)
        } else {
            "".to_string()
        }
    }

    pub fn gguf_repo_link(&self) -> String {
        if let Some(gguf_repo_link) = &self.gguf_repo_link {
            format!("https://huggingface.co/{}", gguf_repo_link)
        } else {
            "".to_string()
        }
    }

    pub fn estimated_memory_usage(&self) -> String {
        if let Some(recommended_quant) = self.get_recommended_quant() {
            recommended_quant.estimated_memory_usage()
        } else {
            "".to_string()
        }
    }

    pub fn recommended_quantization_level(&self) -> Option<u8> {
        if let Some(recommended_quant) = self.get_recommended_quant() {
            Some(recommended_quant.quantization_level)
        } else {
            None
        }
    }

    pub fn get_recommended_quant(&self) -> Option<&LlmGuiLocalQuant> {
        if let Some(recommended_quantization_level) = self.recommended_quantization_level {
            self.quants
                .iter()
                .find(|quant| quant.quantization_level == recommended_quantization_level)
        } else {
            None
        }
    }

    pub fn reset_model_state(&mut self) {
        if self.quants.iter().any(|quant| quant.downloaded) {
            self.model_state = ModelState::Downloaded;
        } else {
            self.model_state = ModelState::Downloadable;
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LlmGuiLocalQuant {
    pub quantization_level: u8,
    pub file_name: String,
    pub downloaded: bool,
    pub on_disk_file_size_bytes: Option<u64>,
    pub total_file_size_bytes: u64,
    pub estimated_memory_usage_bytes: u64,
}

impl LlmGuiLocalQuant {
    fn bytes_to_giga_bytes(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn estimated_memory_usage(&self) -> String {
        format!(
            "{:.2} GB",
            Self::bytes_to_giga_bytes(self.estimated_memory_usage_bytes)
        )
    }
}

#[cfg(feature = "server")]
pub mod server {
    use super::*;
    use llm_client::llm_models::gguf_presets::{GgufPreset, GgufQuant};

    pub fn default_model_id() -> String {
        GgufPreset::default().model_id.to_owned()
    }

    pub fn get_models() -> Result<Vec<LlmGuiLocalModel>, ServerFnError> {
        let default_model_id = default_model_id();
        let all_server_models = GgufPreset::all_models();
        let all_models: Vec<LlmGuiLocalModel> = all_server_models
            .into_iter()
            .map(|preset| LlmGuiLocalModel::new(preset, &default_model_id))
            .collect();
        Ok(all_models)
    }

    impl LlmGuiLocalModel {
        fn new(preset: GgufPreset, default_model_id: &str) -> Self {
            let available_memory_bytes: u64 =
                crate::api::llm_client::server::available_memory_bytes();
            let quants: Vec<LlmGuiLocalQuant> = preset
                .get_quants(None)
                .into_iter()
                .map(|quant| LlmGuiLocalQuant::new(quant))
                .collect();

            let model_state = if preset.model_id == default_model_id {
                ModelState::Selected(LoadState::Unloaded)
            } else if quants.iter().any(|quant| quant.downloaded) {
                ModelState::Downloaded
            } else {
                ModelState::Downloadable
            };
            let recommended_quantization_level = preset
                .select_quant_for_available_memory(None, available_memory_bytes)
                .ok();
            Self {
                model_state,
                base: LlmGuiModelBase {
                    model_id: preset.model_id.to_owned(),
                    friendly_name: preset.friendly_name.to_owned(),
                },
                number_of_parameters: Some(preset.number_of_parameters),
                organization: Some(preset.organization.friendly_name.to_owned()),
                model_repo_link: Some(preset.model_repo_id.to_owned()),
                gguf_repo_link: Some(preset.gguf_repo_id.to_owned()),
                recommended_quantization_level,
                quants,
            }
        }
    }

    impl LlmGuiLocalQuant {
        fn new(quant: GgufQuant) -> Self {
            Self {
                quantization_level: quant.q_lvl,
                file_name: quant.file_name.to_owned(),
                downloaded: quant.downloaded,
                on_disk_file_size_bytes: quant.on_disk_file_size_bytes.map(|size| size as u64),
                total_file_size_bytes: quant.total_file_size_bytes as u64,
                estimated_memory_usage_bytes: quant.estimated_memory_usage_bytes as u64,
            }
        }
    }
}
