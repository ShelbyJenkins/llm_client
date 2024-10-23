//! Adding a preset model checklist:
//! 1. Add a new variant to the `LlmPreset` enum via generate_models! macro
//! 2. Create directory for the new model in `llm_client/llm_models/src/local_model/gguf/preset`
//! 3. Add model_macro_data.json to the new model's directory
//! 4. Add the model's config.json to the new model's directory
//! 5. (Optional) Add the model's tokenizer_config.json to the new model's directory
//! 6. (Optional) Add the model's tokenizer.json to the new model's directory
//! 7. Add a test to llm_client/llm_models/tests/it/preset.rs for the new model
//! 8. Add a test_base_generation_prefix test case to llm_client/llm_models/tests/it/metadata.rs for the new model
use crate::local_model::{
    gguf::loaders::preset::GgufPresetLoader, hf_loader::HuggingFaceLoader,
    metadata::config_json::ConfigJson, GgufLoader, LocalLlmModel,
};

fn presets_path() -> std::path::PathBuf {
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(cargo_manifest_dir)
        .join("src")
        .join("local_model")
        .join("gguf")
        .join("preset")
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmPresetData {
    pub model_id: String,
    pub gguf_repo_id: String,
    pub number_of_parameters: u64,
    pub f_name_for_q_bits: QuantizationConfig,
    pub tokenizer_preset_data: TokenizerPresetData,
    pub tokenizer_config_preset_data: TokenizerConfigPresetData,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenizerPresetData {
    pub local_path: Option<String>,
    pub hf_repo: Option<String>,
    pub hf_filename: Option<String>,
}
impl TokenizerPresetData {
    pub fn load(&self, hf_loader: &HuggingFaceLoader) -> crate::Result<std::path::PathBuf> {
        if let Some(local_path) = self.local_path.clone() {
            let path = presets_path().join(local_path);
            match std::fs::File::open(&path) {
                Ok(_) => Ok(path),
                Err(_) => crate::bail!("Failed to open tokenizer.json at {}", path.display()),
            }
        } else {
            if let (Some(hf_repo), Some(hf_filename)) =
                (self.hf_repo.clone(), self.hf_filename.clone())
            {
                hf_loader.load_file(hf_filename, hf_repo)
            } else {
                crate::bail!("No local tokenizer.json, or hf_repo and hf_filename provided")
            }
        }
    }
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenizerConfigPresetData {
    pub local_path: Option<String>,
    pub hf_repo: Option<String>,
    pub hf_filename: Option<String>,
}
impl TokenizerConfigPresetData {
    pub fn load(&self, hf_loader: &HuggingFaceLoader) -> crate::Result<std::path::PathBuf> {
        if let Some(local_path) = self.local_path.clone() {
            let path = presets_path().join(local_path);
            match std::fs::File::open(&path) {
                Ok(_) => Ok(path),
                Err(_) => {
                    crate::bail!("Failed to open tokenizer_config.json at {}", path.display())
                }
            }
        } else {
            if let (Some(hf_repo), Some(hf_filename)) =
                (self.hf_repo.clone(), self.hf_filename.clone())
            {
                hf_loader.load_file(hf_filename, hf_repo)
            } else {
                crate::bail!("No local tokenizer_config.json, or hf_repo and hf_filename provided")
            }
        }
    }
}

impl LlmPresetData {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> LlmPresetData {
        let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = std::path::PathBuf::from(cargo_manifest_dir)
            .join("src")
            .join("local_model")
            .join("gguf")
            .join("preset")
            .join(path)
            .join("model_macro_data.json");
        let mut file = std::fs::File::open(&path)
            .unwrap_or_else(|_| panic!("Failed to open file at {}", path.display()));
        let mut contents = String::new();
        std::io::Read::read_to_string(&mut file, &mut contents).expect("Failed to read file");
        serde_json::from_str(&contents).expect("Failed to parse JSON")
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizationConfig {
    pub q8: Option<String>,
    pub q7: Option<String>,
    pub q6: Option<String>,
    pub q5: Option<String>,
    pub q4: Option<String>,
    pub q3: Option<String>,
    pub q2: Option<String>,
    pub q1: Option<String>,
}

macro_rules! generate_models {
    ($enum_name:ident {
        $($variant:ident => $path:expr),* $(,)?
    }) => {
        #[derive(Debug, Clone)]
        pub enum $enum_name {
            $($variant),*
        }

        impl $enum_name {
            pub fn get_data(&self) -> &'static LlmPresetData {
                match self {
                    $(
                        Self::$variant => {
                            static DATA: std::sync::LazyLock<LlmPresetData> = std::sync::LazyLock::new(|| {
                                LlmPresetData::new($path)
                            });
                            &DATA
                        }
                    ),*
                }
            }

            pub fn model_id(&self) -> String {
                self.get_data().model_id.to_string()
            }

            pub fn gguf_repo_id(&self) -> &str {
                &self.get_data().gguf_repo_id
            }

            pub fn config_json(&self) -> crate::Result<ConfigJson> {
                ConfigJson::from_local_path(&self.config_json_path())
            }

            pub fn f_name_for_q_bits(&self, q_bits: u8) -> Option<String> {
                match q_bits {
                    8 => self.get_data().f_name_for_q_bits.q8.clone(),
                    7 => self.get_data().f_name_for_q_bits.q7.clone(),
                    6 => self.get_data().f_name_for_q_bits.q6.clone(),
                    5 => self.get_data().f_name_for_q_bits.q5.clone(),
                    4 => self.get_data().f_name_for_q_bits.q4.clone(),
                    3 => self.get_data().f_name_for_q_bits.q3.clone(),
                    2 => self.get_data().f_name_for_q_bits.q2.clone(),
                    1 => self.get_data().f_name_for_q_bits.q1.clone(),
                    _ => panic!("Quantization bits must be between 1 and 8"),
                }

            }

            pub fn number_of_parameters(&self) -> f64 {
                self.get_data().number_of_parameters as f64 * 1_000_000_000.0
            }



            fn preset_dir_path(&self) -> std::path::PathBuf {
                match self {
                    $(
                        Self::$variant => {
                            presets_path()
                                .join($path)
                        }
                    ),*
                }
            }

            pub fn config_json_path(&self) -> std::path::PathBuf {
                let preset_config_path = self.preset_dir_path();
                preset_config_path.join("config.json")
            }

            pub fn load_tokenizer(&self,hf_loader: &HuggingFaceLoader) -> crate::Result<std::path::PathBuf> {
                self.get_data().tokenizer_preset_data.load(hf_loader)
            }

            pub fn load_tokenizer_config(&self,hf_loader: &HuggingFaceLoader) -> crate::Result<std::path::PathBuf> {
                self.get_data().tokenizer_config_preset_data.load(hf_loader)
            }



            pub fn load(&self) -> crate::Result<LocalLlmModel> {
                let mut loader = GgufLoader::default();
                loader.gguf_preset_loader.llm_preset = self.clone();
                loader.load()
            }
        }

        pub trait GgufPresetTrait {
            fn preset_loader(&mut self) -> &mut GgufPresetLoader;

            fn preset_with_available_vram_gb(mut self, preset_with_available_vram_gb: u32) -> Self
            where
                Self: Sized,
            {
                self.preset_loader().preset_with_available_vram_gb = Some(preset_with_available_vram_gb);
                self
            }


            fn preset_with_quantization_level(mut self, level: u8) -> Self
            where
                Self: Sized,
            {
                self.preset_loader().preset_with_quantization_level = Some(level);
                self
            }

            $(
                paste::paste! {
                    fn [<$variant:snake>](mut self) -> Self
                    where
                        Self: Sized,
                    {
                        self.preset_loader().llm_preset = $enum_name::$variant;
                        self
                    }
                }
            )*

        }


    };
}

generate_models!(
    LlmPreset {
        SuperNovaMedius13b => "arcee/supernova_medius",
        Llama3_1_8bInstruct => "llama/llama3_1_8b_instruct",
        Llama3_2_3bInstruct => "llama/llama3_2_3b_instruct",
        Llama3_2_1bInstruct => "llama/llama3_2_1b_instruct",
        Mistral7bInstructV0_3 => "mistral/mistral7b_instruct_v0_3",
        Mixtral8x7bInstructV0_1 => "mistral/mixtral8x7b_instruct_v0_1",
        MistralNemoInstruct2407 => "mistral/mistral_nemo_instruct_2407",
        MistralSmallInstruct2409 => "mistral/mistral_small_instruct_2409",
        Phi3Medium4kInstruct => "phi/phi3_medium4k_instruct",
        Phi3Mini4kInstruct => "phi/phi3_mini4k_instruct",
        Phi3_5MiniInstruct => "phi/phi3_5_mini_instruct",
        Granite3_8bInstruct => "granite/granite3_8b_instruct",
        Granite3_2bInstruct => "granite/granite3_2b_instruct",
        Qwen2_5_32bInstruct => "qwen/qwen2_5_32b_instruct",
        Qwen2_5_14bInstruct => "qwen/qwen2_5_14b_instruct",
        Qwen2_5_7bInstruct => "qwen/qwen2_5_7b_instruct",
        Qwen2_5_3bInstruct => "qwen/qwen2_5_3b_instruct",
        Llama3_1_70bNemotronInstruct => "nvidia/llama3_1_70b_nemotron_instruct",
        MistralNemoMinitron8bInstruct => "nvidia/mistral_nemo_minitron_8b_instruct",
        StableLm2_12bChat => "stabilityai/stablelm_2_12b_chat",
    }
);
