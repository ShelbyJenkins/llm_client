//! Adding a preset model checklist:
//! 1. Add a new variant to the `LlmPreset` enum via generate_models! macro
//! 2. Create directory for the new model in `llm_utils/src/models/local_model/preset`
//! 3. Add model_macro_data.json to the new model's directory
//! 4. Add the model's config.json to the new model's directory
//! 5. (Optional) Add the model's tokenizer_config.json to the new model's directory
//! 6. (Optional) Add the model's tokenizer.json to the new model's directory
//! 7. Add a test to llm_utils/src/models/local_model/preset/mod.rs/tests for the new model
//! 8. Add a test_base_generation_prefix test case to llm_utils/src/models/local_model/chat_template.rs/tests for the new model
use crate::local_model::{
    gguf::loaders::preset::GgufPresetLoader, metadata::config_json::ConfigJson, GgufLoader,
    LocalLlmModel,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmPresetData {
    pub model_id: String,
    pub gguf_repo_id: String,
    pub number_of_parameters: u64,
    pub f_name_for_q_bits: QuantizationConfig,
    pub base_generation_prefix: String,
    pub tokenizer_json_path: Option<String>,
    pub tokenizer_config_json_path: Option<String>,
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
            fn get_data(&self) -> &'static LlmPresetData {
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

            fn presets_path(&self) -> std::path::PathBuf {
                let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
                std::path::PathBuf::from(cargo_manifest_dir)
                    .join("src")
                    .join("local_model")
                    .join("gguf")
                    .join("preset")
            }

            fn preset_dir_path(&self) -> std::path::PathBuf {
                match self {
                    $(
                        Self::$variant => {
                            self.presets_path()
                                .join($path)
                        }
                    ),*
                }
            }

            pub fn config_json_path(&self) -> std::path::PathBuf {
                let preset_config_path = self.preset_dir_path();
                preset_config_path.join("config.json")
            }

            pub fn tokenizer_path(&self) -> Option<std::path::PathBuf> {
                if let Some(tokenizer_json_path) = self.get_data().tokenizer_json_path.clone() {
                    let path = self.presets_path().join(tokenizer_json_path);
                    match std::fs::File::open(&path) {
                        Ok(_) => Some(path),
                        Err(_) => None,
                    }
                } else {
                    None
                }
            }

            pub fn tokenizer_config_path(&self) -> Option<std::path::PathBuf> {
                if let Some(tokenizer_config_json_path) = self.get_data().tokenizer_config_json_path.clone() {
                    let path = self.presets_path().join(tokenizer_config_json_path);
                    match std::fs::File::open(&path) {
                        Ok(_) => Some(path),
                        Err(_) => None,
                    }
                } else {
                    None
                }
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
        Llama3_1_70bNemotronInstruct => "llama/llama3_1_70b_nemotron_instruct",
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

    }
);
