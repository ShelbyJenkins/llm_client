use super::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum GgufPresetId {
    Llama321BInstruct,
    Llama323BInstruct,
    Llama318BInstructGGUF,
    Mixtral8x7BInstructV01,
    Mistral7BInstructV03,
    MistralNemoInstruct2407,
    MistralSmall24BInstruct2501,
    MistralSmallInstruct2409,
    Stablelm212bChat,
    Qwen257BInstruct,
    Qwen2532BInstruct,
    Qwen2514BInstruct,
    Qwen253BInstruct,
    Granite308bInstruct,
    Granite302bInstruct,
    SuperNovaMedius,
    Llama31Nemotron70BInstruct,
    Llama31Nemotron51BInstruct,
    MistralNeMoMinitron8BInstruct,
    Phi35MiniInstruct,
    Phi3Medium4kInstruct,
    Phi4MiniInstruct,
    Phi3Mini4kInstruct,
    Phi4,
    Phi35MoEInstruct,
}
impl GgufPresetId {
    pub fn preset(&self) -> GgufPreset {
        match self {
            Self::Llama321BInstruct => GgufPreset::LLAMA_3_2_1B_INSTRUCT,
            Self::Llama323BInstruct => GgufPreset::LLAMA_3_2_3B_INSTRUCT,
            Self::Llama318BInstructGGUF => GgufPreset::LLAMA_3_1_8B_INSTRUCT,
            Self::Mixtral8x7BInstructV01 => GgufPreset::MIXTRAL_8X7B_INSTRUCT_V0_1,
            Self::Mistral7BInstructV03 => GgufPreset::MISTRAL_7B_INSTRUCT_V0_3,
            Self::MistralNemoInstruct2407 => GgufPreset::MISTRAL_NEMO_INSTRUCT_2407,
            Self::MistralSmall24BInstruct2501 => GgufPreset::MISTRAL_SMALL_24B_INSTRUCT_2501,
            Self::MistralSmallInstruct2409 => GgufPreset::MISTRAL_SMALL_INSTRUCT_2409,
            Self::Stablelm212bChat => GgufPreset::STABLE_LM_2_12B_CHAT,
            Self::Qwen257BInstruct => GgufPreset::QWEN2_5_7B_INSTRUCT,
            Self::Qwen2532BInstruct => GgufPreset::QWEN2_5_32B_INSTRUCT,
            Self::Qwen2514BInstruct => GgufPreset::QWEN2_5_14B_INSTRUCT,
            Self::Qwen253BInstruct => GgufPreset::QWEN2_5_3B_INSTRUCT,
            Self::Granite308bInstruct => GgufPreset::GRANITE_3_0_8B_INSTRUCT,
            Self::Granite302bInstruct => GgufPreset::GRANITE_3_0_2B_INSTRUCT,
            Self::SuperNovaMedius => GgufPreset::SUPERNOVA_MEDIUS,
            Self::Llama31Nemotron70BInstruct => GgufPreset::LLAMA_3_1_NEMOTRON_70B_INSTRUCT,
            Self::Llama31Nemotron51BInstruct => GgufPreset::LLAMA_3_1_NEMOTRON_51B_INSTRUCT,
            Self::MistralNeMoMinitron8BInstruct => GgufPreset::MISTRAL_NEMO_MINITRON_8B_INSTRUCT,
            Self::Phi35MiniInstruct => GgufPreset::PHI_3_5_MINI_INSTRUCT,
            Self::Phi3Medium4kInstruct => GgufPreset::PHI_3_MEDIUM_4K_INSTRUCT,
            Self::Phi4MiniInstruct => GgufPreset::PHI_4_MINI_INSTRUCT_,
            Self::Phi3Mini4kInstruct => GgufPreset::PHI_3_MINI_4K_INSTRUCT,
            Self::Phi4 => GgufPreset::PHI_4,
            Self::Phi35MoEInstruct => GgufPreset::PHI_3_5_MOE_INSTRUCT,
        }
    }
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Llama321BInstruct => "Llama-3.2-1B-Instruct",
            Self::Llama323BInstruct => "Llama-3.2-3B-Instruct",
            Self::Llama318BInstructGGUF => "Llama-3.1-8B-Instruct-GGUF",
            Self::Mixtral8x7BInstructV01 => "Mixtral-8x7B-Instruct-v0.1",
            Self::Mistral7BInstructV03 => "Mistral-7B-Instruct-v0.3",
            Self::MistralNemoInstruct2407 => "Mistral-Nemo-Instruct-2407",
            Self::MistralSmall24BInstruct2501 => "Mistral-Small-24B-Instruct-2501",
            Self::MistralSmallInstruct2409 => "Mistral-Small-Instruct-2409",
            Self::Stablelm212bChat => "stablelm-2-12b-chat",
            Self::Qwen257BInstruct => "Qwen2.5-7B-Instruct",
            Self::Qwen2532BInstruct => "Qwen2.5-32B-Instruct",
            Self::Qwen2514BInstruct => "Qwen2.5-14B-Instruct",
            Self::Qwen253BInstruct => "Qwen2.5-3B-Instruct",
            Self::Granite308bInstruct => "granite-3.0-8b-instruct",
            Self::Granite302bInstruct => "granite-3.0-2b-instruct",
            Self::SuperNovaMedius => "SuperNova-Medius",
            Self::Llama31Nemotron70BInstruct => "Llama-3.1-Nemotron-70B-Instruct",
            Self::Llama31Nemotron51BInstruct => "Llama-3_1-Nemotron-51B-Instruct",
            Self::MistralNeMoMinitron8BInstruct => "Mistral-NeMo-Minitron-8B-Instruct",
            Self::Phi35MiniInstruct => "Phi-3.5-mini-instruct",
            Self::Phi3Medium4kInstruct => "Phi-3-medium-4k-instruct",
            Self::Phi4MiniInstruct => "phi-4-mini-instruct ",
            Self::Phi3Mini4kInstruct => "Phi-3-mini-4k-instruct",
            Self::Phi4 => "phi-4",
            Self::Phi35MoEInstruct => "Phi-3.5-MoE-instruct",
        }
    }
}
impl GgufPreset {
    pub const ALL_MODELS: [GgufPreset; 25usize] = [
        Self::LLAMA_3_2_1B_INSTRUCT,
        Self::LLAMA_3_2_3B_INSTRUCT,
        Self::LLAMA_3_1_8B_INSTRUCT,
        Self::MIXTRAL_8X7B_INSTRUCT_V0_1,
        Self::MISTRAL_7B_INSTRUCT_V0_3,
        Self::MISTRAL_NEMO_INSTRUCT_2407,
        Self::MISTRAL_SMALL_24B_INSTRUCT_2501,
        Self::MISTRAL_SMALL_INSTRUCT_2409,
        Self::STABLE_LM_2_12B_CHAT,
        Self::QWEN2_5_7B_INSTRUCT,
        Self::QWEN2_5_32B_INSTRUCT,
        Self::QWEN2_5_14B_INSTRUCT,
        Self::QWEN2_5_3B_INSTRUCT,
        Self::GRANITE_3_0_8B_INSTRUCT,
        Self::GRANITE_3_0_2B_INSTRUCT,
        Self::SUPERNOVA_MEDIUS,
        Self::LLAMA_3_1_NEMOTRON_70B_INSTRUCT,
        Self::LLAMA_3_1_NEMOTRON_51B_INSTRUCT,
        Self::MISTRAL_NEMO_MINITRON_8B_INSTRUCT,
        Self::PHI_3_5_MINI_INSTRUCT,
        Self::PHI_3_MEDIUM_4K_INSTRUCT,
        Self::PHI_4_MINI_INSTRUCT_,
        Self::PHI_3_MINI_4K_INSTRUCT,
        Self::PHI_4,
        Self::PHI_3_5_MOE_INSTRUCT,
    ];
    pub const LLAMA_3_2_1B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::META,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Llama-3.2-1B-Instruct"),
            friendly_name: Cow::Borrowed("Llama 3.2 1B Instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("meta-llama/Llama-3.2-1B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Llama-3.2-1B-Instruct-GGUF"),
        number_of_parameters: 3f64,
        tokenizer_file_name: Some(Cow::Borrowed("meta.llama3.1.8b.instruct.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 2048u64,
            feed_forward_length: Some(8192u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 16u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 128256u64,
            architecture: Cow::Borrowed("llama"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Llama-3.2-1B-Instruct-IQ3_M.gguf"),
                total_bytes: 657289344u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
                total_bytes: 807694464u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Llama-3.2-1B-Instruct-Q5_K_M.gguf"),
                total_bytes: 911503488u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Llama-3.2-1B-Instruct-Q6_K.gguf"),
                total_bytes: 1021800576u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Llama-3.2-1B-Instruct-Q8_0.gguf"),
                total_bytes: 1321083008u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Llama321BInstruct,
    };
    pub const LLAMA_3_2_3B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::META,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Llama-3.2-3B-Instruct"),
            friendly_name: Cow::Borrowed("Llama 3.2 3B Instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("meta-llama/Llama-3.2-3B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Llama-3.2-3B-Instruct-GGUF"),
        number_of_parameters: 3f64,
        tokenizer_file_name: Some(Cow::Borrowed("meta.llama3.1.8b.instruct.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 3072u64,
            feed_forward_length: Some(8192u64),
            head_count: 24u64,
            head_count_kv: Some(8u64),
            block_count: 28u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 128256u64,
            architecture: Cow::Borrowed("llama"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Llama-3.2-3B-Instruct-IQ3_M.gguf"),
                total_bytes: 1599668768u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
                total_bytes: 2019377696u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Llama-3.2-3B-Instruct-Q5_K_M.gguf"),
                total_bytes: 2322154016u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Llama-3.2-3B-Instruct-Q6_K.gguf"),
                total_bytes: 2643853856u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Llama-3.2-3B-Instruct-Q8_0.gguf"),
                total_bytes: 3421899296u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Llama323BInstruct,
    };
    pub const LLAMA_3_1_8B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::META,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Llama-3.1-8B-Instruct-GGUF"),
            friendly_name: Cow::Borrowed("Llama 3.1 8B Instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("meta-llama/Llama-3.1-8B-Instruct-GGUF"),
        gguf_repo_id: Cow::Borrowed("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
        number_of_parameters: 8f64,
        tokenizer_file_name: Some(Cow::Borrowed("meta.llama3.1.8b.instruct.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 4096u64,
            feed_forward_length: Some(14336u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 128256u64,
            architecture: Cow::Borrowed("llama"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Meta-Llama-3.1-8B-Instruct-Q2_K.gguf"),
                total_bytes: 3179136416u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf"),
                total_bytes: 4018922912u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
                total_bytes: 4920739232u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"),
                total_bytes: 5732992416u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"),
                total_bytes: 6596011424u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
                total_bytes: 8540775840u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Llama318BInstructGGUF,
    };
    pub const MIXTRAL_8X7B_INSTRUCT_V0_1: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1"),
            friendly_name: Cow::Borrowed("Mixtral 8x7B Instruct v0.1"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("nvidia/Mixtral-8x7B-Instruct-v0.1"),
        gguf_repo_id: Cow::Borrowed("MaziyarPanahi/Mixtral-8x7B-Instruct-v0.1-GGUF"),
        number_of_parameters: 56f64,
        tokenizer_file_name: Some(Cow::Borrowed(
            "mistral.mixtral8x7b.instruct.v0.1.tokenizer.json",
        )),
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 4096u64,
            feed_forward_length: Some(14336u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32768u64,
            architecture: Cow::Borrowed("mistral"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1.Q2_K.gguf"),
                total_bytes: 17309173632u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1.Q3_K_M.gguf"),
                total_bytes: 22544394112u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"),
                total_bytes: 28446410624u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf"),
                total_bytes: 33227523968u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1.Q6_K.gguf"),
                total_bytes: 38378760064u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf"),
                total_bytes: 49624262528u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Mixtral8x7BInstructV01,
    };
    pub const MISTRAL_7B_INSTRUCT_V0_3: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Mistral-7B-Instruct-v0.3"),
            friendly_name: Cow::Borrowed("Mistral 7B Instruct v0.3"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("mistral/Mistral-7B-Instruct-v0.3"),
        gguf_repo_id: Cow::Borrowed("MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"),
        number_of_parameters: 7f64,
        tokenizer_file_name: Some(Cow::Borrowed(
            "mistral.mistral7b.instruct.v0.3.tokenizer.json",
        )),
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 4096u64,
            feed_forward_length: Some(14336u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32768u64,
            architecture: Cow::Borrowed("mistral"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 1u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.IQ1_M.gguf"),
                total_bytes: 1757663392u64,
            },
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.Q2_K.gguf"),
                total_bytes: 2722877600u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.Q3_K_M.gguf"),
                total_bytes: 3522941088u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"),
                total_bytes: 4372811936u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.Q5_K_M.gguf"),
                total_bytes: 5136175264u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.Q6_K.gguf"),
                total_bytes: 5947248800u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
                total_bytes: 7702565024u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Mistral7BInstructV03,
    };
    pub const MISTRAL_NEMO_INSTRUCT_2407: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Mistral-Nemo-Instruct-2407"),
            friendly_name: Cow::Borrowed("Mistral Nemo Instruct 2407"),
            model_ctx_size: 1024000u64,
            inference_ctx_size: 1024000u64,
        },
        model_repo_id: Cow::Borrowed("mistral/Mistral-Nemo-Instruct-2407"),
        gguf_repo_id: Cow::Borrowed("bartowski/Mistral-Nemo-Instruct-2407-GGUF"),
        number_of_parameters: 12f64,
        tokenizer_file_name: Some(Cow::Borrowed(
            "mistral.mistral.nemo.instruct.2407.tokenizer.json",
        )),
        config: GgufPresetConfig {
            context_length: 1024000u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(14336u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 131072u64,
            architecture: Cow::Borrowed("mistral"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Mistral-Nemo-Instruct-2407-Q2_K.gguf"),
                total_bytes: 4791051392u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Mistral-Nemo-Instruct-2407-Q3_K_M.gguf"),
                total_bytes: 6083093632u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"),
                total_bytes: 7477208192u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"),
                total_bytes: 8727635072u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Mistral-Nemo-Instruct-2407-Q6_K.gguf"),
                total_bytes: 10056213632u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Mistral-Nemo-Instruct-2407-Q8_0.gguf"),
                total_bytes: 13022372992u64,
            },
        ]),
        preset_llm_id: GgufPresetId::MistralNemoInstruct2407,
    };
    pub const MISTRAL_SMALL_24B_INSTRUCT_2501: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Mistral-Small-24B-Instruct-2501"),
            friendly_name: Cow::Borrowed("Mistral Small 24B Instruct 2501"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("mistral/Mistral-Small-24B-Instruct-2501"),
        gguf_repo_id: Cow::Borrowed("bartowski/Mistral-Small-24B-Instruct-2501-GGUF"),
        number_of_parameters: 24f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(32768u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 131072u64,
            architecture: Cow::Borrowed("mistral"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Mistral-Small-24B-Instruct-2501-Q2_K.gguf"),
                total_bytes: 8890324672u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Mistral-Small-24B-Instruct-2501-Q3_K_M.gguf"),
                total_bytes: 11474081472u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"),
                total_bytes: 14333908672u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Mistral-Small-24B-Instruct-2501-Q5_K_M.gguf"),
                total_bytes: 16763983552u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Mistral-Small-24B-Instruct-2501-Q6_K.gguf"),
                total_bytes: 19345938112u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Mistral-Small-24B-Instruct-2501-Q8_0.gguf"),
                total_bytes: 25054779072u64,
            },
        ]),
        preset_llm_id: GgufPresetId::MistralSmall24BInstruct2501,
    };
    pub const MISTRAL_SMALL_INSTRUCT_2409: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Mistral-Small-Instruct-2409"),
            friendly_name: Cow::Borrowed("Mistral Small Instruct 2409"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("mistral/Mistral-Small-Instruct-2409"),
        gguf_repo_id: Cow::Borrowed("bartowski/Mistral-Small-Instruct-2409-GGUF"),
        number_of_parameters: 12f64,
        tokenizer_file_name: Some(Cow::Borrowed(
            "mistral.mistral.small.instruct.2409.tokenizer.json",
        )),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 6144u64,
            feed_forward_length: Some(16384u64),
            head_count: 48u64,
            head_count_kv: Some(8u64),
            block_count: 56u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32768u64,
            architecture: Cow::Borrowed("mistral"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Mistral-Small-Instruct-2409-Q2_K.gguf"),
                total_bytes: 8272098304u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Mistral-Small-Instruct-2409-Q3_K_M.gguf"),
                total_bytes: 10756830208u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Mistral-Small-Instruct-2409-Q4_K_M.gguf"),
                total_bytes: 13341242368u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Mistral-Small-Instruct-2409-Q5_K_M.gguf"),
                total_bytes: 15722558464u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Mistral-Small-Instruct-2409-Q6_K.gguf"),
                total_bytes: 18252706816u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Mistral-Small-Instruct-2409-Q8_0.gguf"),
                total_bytes: 23640552448u64,
            },
        ]),
        preset_llm_id: GgufPresetId::MistralSmallInstruct2409,
    };
    pub const STABLE_LM_2_12B_CHAT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::STABILITY_AI,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("stablelm-2-12b-chat"),
            friendly_name: Cow::Borrowed("Stable LM 2 12B Chat"),
            model_ctx_size: 4096u64,
            inference_ctx_size: 4096u64,
        },
        model_repo_id: Cow::Borrowed("stabilityai/stablelm-2-12b-chat"),
        gguf_repo_id: Cow::Borrowed("second-state/stablelm-2-12b-chat-GGUF"),
        number_of_parameters: 12f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 4096u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(13824u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 100352u64,
            architecture: Cow::Borrowed("stablelm"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("stablelm-2-12b-chat-Q2_K.gguf"),
                total_bytes: 4698894176u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("stablelm-2-12b-chat-Q3_K_M.gguf"),
                total_bytes: 5993885536u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("stablelm-2-12b-chat-Q4_K_M.gguf"),
                total_bytes: 7367642976u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("stablelm-2-12b-chat-Q5_K_M.gguf"),
                total_bytes: 8627900256u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("stablelm-2-12b-chat-Q6_K.gguf"),
                total_bytes: 9966923616u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("stablelm-2-12b-chat-Q8_0.gguf"),
                total_bytes: 12907687776u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Stablelm212bChat,
    };
    pub const QWEN2_5_7B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Qwen2.5-7B-Instruct"),
            friendly_name: Cow::Borrowed("Qwen2.5 7B Instruct"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("Qwen/Qwen2.5-7B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Qwen2.5-7B-Instruct-GGUF"),
        number_of_parameters: 7f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 3584u64,
            feed_forward_length: Some(18944u64),
            head_count: 28u64,
            head_count_kv: Some(4u64),
            block_count: 28u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 152064u64,
            architecture: Cow::Borrowed("qwen2"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Qwen2.5-7B-Instruct-Q2_K.gguf"),
                total_bytes: 3015940800u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Qwen2.5-7B-Instruct-Q3_K_M.gguf"),
                total_bytes: 3808391872u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
                total_bytes: 4683074240u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Qwen2.5-7B-Instruct-Q5_K_M.gguf"),
                total_bytes: 5444831936u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Qwen2.5-7B-Instruct-Q6_K.gguf"),
                total_bytes: 6254199488u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Qwen2.5-7B-Instruct-Q8_0.gguf"),
                total_bytes: 8098525888u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Qwen257BInstruct,
    };
    pub const QWEN2_5_32B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Qwen2.5-32B-Instruct"),
            friendly_name: Cow::Borrowed("Qwen2.5 32B Instruct"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("Qwen/Qwen2.5-32B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Qwen2.5-32B-Instruct-GGUF"),
        number_of_parameters: 32f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(27648u64),
            head_count: 40u64,
            head_count_kv: Some(8u64),
            block_count: 64u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 152064u64,
            architecture: Cow::Borrowed("qwen2"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Qwen2.5-32B-Instruct-Q2_K.gguf"),
                total_bytes: 12313099136u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Qwen2.5-32B-Instruct-Q3_K_M.gguf"),
                total_bytes: 15935048576u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Qwen2.5-32B-Instruct-Q4_K_M.gguf"),
                total_bytes: 19851336576u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Qwen2.5-32B-Instruct-Q5_K_M.gguf"),
                total_bytes: 23262157696u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Qwen2.5-32B-Instruct-Q6_K.gguf"),
                total_bytes: 26886155136u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Qwen2.5-32B-Instruct-Q8_0.gguf"),
                total_bytes: 34820885376u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Qwen2532BInstruct,
    };
    pub const QWEN2_5_14B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Qwen2.5-14B-Instruct"),
            friendly_name: Cow::Borrowed("Qwen2.5 14B Instruct"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("Qwen/Qwen2.5-14B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Qwen2.5-14B-Instruct-GGUF"),
        number_of_parameters: 14f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(13824u64),
            head_count: 40u64,
            head_count_kv: Some(8u64),
            block_count: 48u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 152064u64,
            architecture: Cow::Borrowed("qwen2"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Qwen2.5-14B-Instruct-Q2_K.gguf"),
                total_bytes: 5770498176u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Qwen2.5-14B-Instruct-Q3_K_M.gguf"),
                total_bytes: 7339204736u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
                total_bytes: 8988110976u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Qwen2.5-14B-Instruct-Q5_K_M.gguf"),
                total_bytes: 10508873856u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Qwen2.5-14B-Instruct-Q6_K.gguf"),
                total_bytes: 12124684416u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Qwen2.5-14B-Instruct-Q8_0.gguf"),
                total_bytes: 15701598336u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Qwen2514BInstruct,
    };
    pub const QWEN2_5_3B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Qwen2.5-3B-Instruct"),
            friendly_name: Cow::Borrowed("Qwen2.5 3B Instruct"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 32768u64,
        },
        model_repo_id: Cow::Borrowed("Qwen/Qwen2.5-3B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Qwen2.5-3B-Instruct-GGUF"),
        number_of_parameters: 3f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 32768u64,
            embedding_length: 2048u64,
            feed_forward_length: Some(11008u64),
            head_count: 16u64,
            head_count_kv: Some(2u64),
            block_count: 36u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 151936u64,
            architecture: Cow::Borrowed("qwen2"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Qwen2.5-3B-Instruct-Q2_K.gguf"),
                total_bytes: 1274756256u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Qwen2.5-3B-Instruct-Q3_K_M.gguf"),
                total_bytes: 1590475936u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
                total_bytes: 1929903264u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Qwen2.5-3B-Instruct-Q5_K_M.gguf"),
                total_bytes: 2224815264u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Qwen2.5-3B-Instruct-Q6_K.gguf"),
                total_bytes: 2538159264u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Qwen2.5-3B-Instruct-Q8_0.gguf"),
                total_bytes: 3285476512u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Qwen253BInstruct,
    };
    pub const GRANITE_3_0_8B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::IBM,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("granite-3.0-8b-instruct"),
            friendly_name: Cow::Borrowed("Granite 3.0 8b instruct"),
            model_ctx_size: 4096u64,
            inference_ctx_size: 4096u64,
        },
        model_repo_id: Cow::Borrowed("ibm/granite-3.0-8b-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/granite-3.0-8b-instruct-GGUF"),
        number_of_parameters: 8f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 4096u64,
            embedding_length: 4096u64,
            feed_forward_length: Some(12800u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 49155u64,
            architecture: Cow::Borrowed("granite"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("granite-3.0-8b-instruct-Q2_K.gguf"),
                total_bytes: 3103588576u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("granite-3.0-8b-instruct-Q3_K_L.gguf"),
                total_bytes: 4349427936u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("granite-3.0-8b-instruct-Q4_K_M.gguf"),
                total_bytes: 4942856416u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("granite-3.0-8b-instruct-Q5_K_M.gguf"),
                total_bytes: 5797445856u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("granite-3.0-8b-instruct-Q6_K.gguf"),
                total_bytes: 6705447136u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("granite-3.0-8b-instruct-Q8_0.gguf"),
                total_bytes: 8684244096u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Granite308bInstruct,
    };
    pub const GRANITE_3_0_2B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::IBM,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("granite-3.0-2b-instruct"),
            friendly_name: Cow::Borrowed("Granite 3.0 2b instruct"),
            model_ctx_size: 4096u64,
            inference_ctx_size: 4096u64,
        },
        model_repo_id: Cow::Borrowed("ibm/granite-3.0-2b-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/granite-3.0-2b-instruct-GGUF"),
        number_of_parameters: 2f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 4096u64,
            embedding_length: 2048u64,
            feed_forward_length: Some(8192u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 49155u64,
            architecture: Cow::Borrowed("granite"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("granite-3.0-2b-instruct-Q2_K.gguf"),
                total_bytes: 1011275040u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("granite-3.0-2b-instruct-Q3_K_L.gguf"),
                total_bytes: 1400625056u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("granite-3.0-2b-instruct-Q4_K_M.gguf"),
                total_bytes: 1601919680u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("granite-3.0-2b-instruct-Q5_K_M.gguf"),
                total_bytes: 1874025920u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("granite-3.0-2b-instruct-Q6_K.gguf"),
                total_bytes: 2163138816u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("granite-3.0-2b-instruct-Q8_0.gguf"),
                total_bytes: 2801069184u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Granite302bInstruct,
    };
    pub const SUPERNOVA_MEDIUS: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ARCEE_AI,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("SuperNova-Medius"),
            friendly_name: Cow::Borrowed("SuperNova Medius"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("arcee-ai/arcee-ai/SuperNova-Medius"),
        gguf_repo_id: Cow::Borrowed("arcee-ai/SuperNova-Medius-GGUF"),
        number_of_parameters: 13f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(13824u64),
            head_count: 40u64,
            head_count_kv: Some(8u64),
            block_count: 48u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 152064u64,
            architecture: Cow::Borrowed("qwen2"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("SuperNova-Medius-Q2_K.gguf"),
                total_bytes: 5770498592u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("SuperNova-Medius-Q3_K_M.gguf"),
                total_bytes: 7339205152u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("SuperNova-Medius-Q4_K_M.gguf"),
                total_bytes: 8988111392u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("SuperNova-Medius-Q5_K_M.gguf"),
                total_bytes: 10508874272u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("SuperNova-Medius-Q6_K.gguf"),
                total_bytes: 12124684832u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("SuperNova-Medius-Q8_0.gguf"),
                total_bytes: 15701598752u64,
            },
        ]),
        preset_llm_id: GgufPresetId::SuperNovaMedius,
    };
    pub const LLAMA_3_1_NEMOTRON_70B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::NVIDIA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Llama-3.1-Nemotron-70B-Instruct"),
            friendly_name: Cow::Borrowed("Llama 3.1 Nemotron 70B Instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"),
        gguf_repo_id: Cow::Borrowed("bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF"),
        number_of_parameters: 70f64,
        tokenizer_file_name: Some(Cow::Borrowed("meta.llama3.1.8b.instruct.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 8192u64,
            feed_forward_length: Some(28672u64),
            head_count: 64u64,
            head_count_kv: Some(8u64),
            block_count: 80u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 128256u64,
            architecture: Cow::Borrowed("llama"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Llama-3.1-Nemotron-70B-Instruct-HF-Q2_K.gguf"),
                total_bytes: 26375113632u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Llama-3.1-Nemotron-70B-Instruct-HF-Q3_K_M.gguf"),
                total_bytes: 34267499424u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"),
                total_bytes: 42520398752u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_S.gguf"),
                total_bytes: 48657451936u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Llama31Nemotron70BInstruct,
    };
    pub const LLAMA_3_1_NEMOTRON_51B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::NVIDIA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Llama-3_1-Nemotron-51B-Instruct"),
            friendly_name: Cow::Borrowed("Llama 3.1 Nemotron 51B Instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("nvidia/Llama-3_1-Nemotron-51B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Llama-3_1-Nemotron-51B-Instruct-GGUF"),
        number_of_parameters: 52f64,
        tokenizer_file_name: Some(Cow::Borrowed("meta.llama3.1.8b.instruct.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 8192u64,
            feed_forward_length: None,
            head_count: 64u64,
            head_count_kv: Some(8u64),
            block_count: 80u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 128256u64,
            architecture: Cow::Borrowed("nemotron-nas"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Llama-3_1-Nemotron-51B-Instruct-Q2_K.gguf"),
                total_bytes: 19418642688u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Llama-3_1-Nemotron-51B-Instruct-Q3_K_M.gguf"),
                total_bytes: 25182345472u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Llama-3_1-Nemotron-51B-Instruct-Q4_K_M.gguf"),
                total_bytes: 31037307136u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Llama-3_1-Nemotron-51B-Instruct-Q5_K_M.gguf"),
                total_bytes: 36465391872u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Llama-3_1-Nemotron-51B-Instruct-Q6_K.gguf"),
                total_bytes: 42258774272u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Llama31Nemotron51BInstruct,
    };
    pub const MISTRAL_NEMO_MINITRON_8B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::NVIDIA,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct"),
            friendly_name: Cow::Borrowed("Mistral NeMo Minitron 8B Instruct"),
            model_ctx_size: 8192u64,
            inference_ctx_size: 8192u64,
        },
        model_repo_id: Cow::Borrowed("nvidia/Mistral-NeMo-Minitron-8B-Instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Mistral-NeMo-Minitron-8B-Instruct-GGUF"),
        number_of_parameters: 8f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 8192u64,
            embedding_length: 4096u64,
            feed_forward_length: Some(11520u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 131072u64,
            architecture: Cow::Borrowed("mistral"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct-Q2_K.gguf"),
                total_bytes: 3333392064u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct-Q3_K_M.gguf"),
                total_bytes: 4209149632u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct-Q4_K_M.gguf"),
                total_bytes: 5145298624u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct-Q5_K_M.gguf"),
                total_bytes: 6001460928u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct-Q6_K.gguf"),
                total_bytes: 6911133376u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Mistral-NeMo-Minitron-8B-Instruct-Q8_0.gguf"),
                total_bytes: 8948844224u64,
            },
        ]),
        preset_llm_id: GgufPresetId::MistralNeMoMinitron8BInstruct,
    };
    pub const PHI_3_5_MINI_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Phi-3.5-mini-instruct"),
            friendly_name: Cow::Borrowed("Phi 3.5 mini instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("microsoft/Phi-3.5-mini-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Phi-3.5-mini-instruct-GGUF"),
        number_of_parameters: 4f64,
        tokenizer_file_name: Some(Cow::Borrowed("microsoft.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 3072u64,
            feed_forward_length: Some(8192u64),
            head_count: 32u64,
            head_count_kv: Some(32u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32064u64,
            architecture: Cow::Borrowed("phi3"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Phi-3.5-mini-instruct-Q2_K.gguf"),
                total_bytes: 1416204576u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Phi-3.5-mini-instruct-Q3_K_M.gguf"),
                total_bytes: 1955477280u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Phi-3.5-mini-instruct-Q4_K_M.gguf"),
                total_bytes: 2393232672u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Phi-3.5-mini-instruct-Q5_K_M.gguf"),
                total_bytes: 2815276320u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Phi-3.5-mini-instruct-Q6_K.gguf"),
                total_bytes: 3135853344u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Phi-3.5-mini-instruct-Q8_0.gguf"),
                total_bytes: 4061222688u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Phi35MiniInstruct,
    };
    pub const PHI_3_MEDIUM_4K_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Phi-3-medium-4k-instruct"),
            friendly_name: Cow::Borrowed("Phi 3 medium 4k instruct"),
            model_ctx_size: 4096u64,
            inference_ctx_size: 4096u64,
        },
        model_repo_id: Cow::Borrowed("microsoft/Phi-3-medium-4k-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Phi-3-medium-4k-instruct-GGUF"),
        number_of_parameters: 14f64,
        tokenizer_file_name: Some(Cow::Borrowed("microsoft.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 4096u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(17920u64),
            head_count: 40u64,
            head_count_kv: Some(10u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32064u64,
            architecture: Cow::Borrowed("phi3"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Phi-3-medium-4k-instruct-Q2_K.gguf"),
                total_bytes: 5143000448u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Phi-3-medium-4k-instruct-Q3_K_M.gguf"),
                total_bytes: 6923411328u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Phi-3-medium-4k-instruct-Q4_K_M.gguf"),
                total_bytes: 8566821248u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Phi-3-medium-4k-instruct-Q5_K_M.gguf"),
                total_bytes: 10074190208u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Phi-3-medium-4k-instruct-Q6_K.gguf"),
                total_bytes: 11453817728u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Phi-3-medium-4k-instruct-Q8_0.gguf"),
                total_bytes: 14834712448u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Phi3Medium4kInstruct,
    };
    pub const PHI_4_MINI_INSTRUCT_: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("phi-4-mini-instruct "),
            friendly_name: Cow::Borrowed("Phi-4-mini-instruct "),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("microsoft/Phi-4-mini-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/microsoft_Phi-4-mini-instruct-GGUF"),
        number_of_parameters: 3.84f64,
        tokenizer_file_name: None,
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 3072u64,
            feed_forward_length: Some(8192u64),
            head_count: 24u64,
            head_count_kv: Some(8u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 200064u64,
            architecture: Cow::Borrowed("phi3"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("microsoft_Phi-4-mini-instruct-Q2_K.gguf"),
                total_bytes: 1682636160u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("microsoft_Phi-4-mini-instruct-Q3_K_M.gguf"),
                total_bytes: 2117533056u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"),
                total_bytes: 2491874688u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("microsoft_Phi-4-mini-instruct-Q5_K_M.gguf"),
                total_bytes: 2848128384u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("microsoft_Phi-4-mini-instruct-Q6_K.gguf"),
                total_bytes: 3155623296u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("microsoft_Phi-4-mini-instruct-Q8_0.gguf"),
                total_bytes: 4084611456u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Phi4MiniInstruct,
    };
    pub const PHI_3_MINI_4K_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Phi-3-mini-4k-instruct"),
            friendly_name: Cow::Borrowed("Phi 3 mini 4k instruct"),
            model_ctx_size: 4096u64,
            inference_ctx_size: 4096u64,
        },
        model_repo_id: Cow::Borrowed("microsoft/Phi-3-mini-4k-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Phi-3-mini-4k-instruct-GGUF"),
        number_of_parameters: 4f64,
        tokenizer_file_name: Some(Cow::Borrowed("microsoft.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 4096u64,
            embedding_length: 3072u64,
            feed_forward_length: Some(8192u64),
            head_count: 32u64,
            head_count_kv: Some(32u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32064u64,
            architecture: Cow::Borrowed("phi3"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 1u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-IQ1_M.gguf"),
                total_bytes: 917106176u64,
            },
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-Q2_K.gguf"),
                total_bytes: 1416203264u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-Q3_K_M.gguf"),
                total_bytes: 1955475968u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-Q4_K_M.gguf"),
                total_bytes: 2393231360u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-Q5_K_M.gguf"),
                total_bytes: 2815275008u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-Q6_K.gguf"),
                total_bytes: 3135852032u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Phi-3-mini-4k-instruct-Q8_0.gguf"),
                total_bytes: 4061221376u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Phi3Mini4kInstruct,
    };
    pub const PHI_4: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("phi-4"),
            friendly_name: Cow::Borrowed("Phi-4"),
            model_ctx_size: 16384u64,
            inference_ctx_size: 16384u64,
        },
        model_repo_id: Cow::Borrowed("microsoft/phi-4"),
        gguf_repo_id: Cow::Borrowed("bartowski/phi-4-GGUF"),
        number_of_parameters: 14f64,
        tokenizer_file_name: Some(Cow::Borrowed("microsoft.phi4.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 16384u64,
            embedding_length: 5120u64,
            feed_forward_length: Some(17920u64),
            head_count: 40u64,
            head_count_kv: Some(10u64),
            block_count: 40u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 100352u64,
            architecture: Cow::Borrowed("phi3"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("phi-4-Q2_K.gguf"),
                total_bytes: 5547348416u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("phi-4-Q3_K_M.gguf"),
                total_bytes: 7363269056u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("phi-4-Q4_K_M.gguf"),
                total_bytes: 9053114816u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("phi-4-Q5_K_M.gguf"),
                total_bytes: 10604188096u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("phi-4-Q6_K.gguf"),
                total_bytes: 12030251456u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("phi-4-Q8_0.gguf"),
                total_bytes: 15580500416u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Phi4,
    };
    pub const PHI_3_5_MOE_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("Phi-3.5-MoE-instruct"),
            friendly_name: Cow::Borrowed("Phi 3.5 MoE instruct"),
            model_ctx_size: 131072u64,
            inference_ctx_size: 131072u64,
        },
        model_repo_id: Cow::Borrowed("microsoft/Phi-3.5-MoE-instruct"),
        gguf_repo_id: Cow::Borrowed("bartowski/Phi-3.5-MoE-instruct-GGUF"),
        number_of_parameters: 7f64,
        tokenizer_file_name: Some(Cow::Borrowed("microsoft.tokenizer.json")),
        config: GgufPresetConfig {
            context_length: 131072u64,
            embedding_length: 4096u64,
            feed_forward_length: Some(6400u64),
            head_count: 32u64,
            head_count_kv: Some(8u64),
            block_count: 32u64,
            torch_dtype: Cow::Borrowed("bfloat16"),
            vocab_size: 32064u64,
            architecture: Cow::Borrowed("phimoe"),
            model_size_bytes: None,
        },
        quants: Cow::Borrowed(&[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: Cow::Borrowed("Phi-3.5-MoE-instruct-Q2_K.gguf"),
                total_bytes: 15265136480u64,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: Cow::Borrowed("Phi-3.5-MoE-instruct-Q3_K_M.gguf"),
                total_bytes: 20032718688u64,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: Cow::Borrowed("Phi-3.5-MoE-instruct-Q4_K_M.gguf"),
                total_bytes: 25345994592u64,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: Cow::Borrowed("Phi-3.5-MoE-instruct-Q5_K_M.gguf"),
                total_bytes: 29716098912u64,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: Cow::Borrowed("Phi-3.5-MoE-instruct-Q6_K.gguf"),
                total_bytes: 34359334752u64,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: Cow::Borrowed("Phi-3.5-MoE-instruct-Q8_0.gguf"),
                total_bytes: 44499765088u64,
            },
        ]),
        preset_llm_id: GgufPresetId::Phi35MoEInstruct,
    };
}
pub trait GgufPresetTrait {
    fn preset(&mut self) -> &mut GgufPreset;
    fn preset_from_str(mut self, selected_model_id: &str) -> crate::Result<Self>
    where
        Self: Sized,
    {
        let preset = GgufPreset::ALL_MODELS
            .into_iter()
            .find(|preset| preset.model_base.model_id == selected_model_id)
            .ok_or_else(|| crate::anyhow!("Invalid selected_model_id: {}", selected_model_id))?;
        *self.preset() = preset;
        Ok(self)
    }
    fn llama_3_2_1b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::LLAMA_3_2_1B_INSTRUCT;
        self
    }
    fn llama_3_2_3b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::LLAMA_3_2_3B_INSTRUCT;
        self
    }
    fn llama_3_1_8b_instruct_gguf(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::LLAMA_3_1_8B_INSTRUCT;
        self
    }
    fn mixtral_8x7b_instruct_v0_1(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::MIXTRAL_8X7B_INSTRUCT_V0_1;
        self
    }
    fn mistral_7b_instruct_v0_3(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::MISTRAL_7B_INSTRUCT_V0_3;
        self
    }
    fn mistral_nemo_instruct_2407(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::MISTRAL_NEMO_INSTRUCT_2407;
        self
    }
    fn mistral_small_24b_instruct_2501(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::MISTRAL_SMALL_24B_INSTRUCT_2501;
        self
    }
    fn mistral_small_instruct_2409(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::MISTRAL_SMALL_INSTRUCT_2409;
        self
    }
    fn stablelm_2_12b_chat(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::STABLE_LM_2_12B_CHAT;
        self
    }
    fn qwen2_5_7b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::QWEN2_5_7B_INSTRUCT;
        self
    }
    fn qwen2_5_32b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::QWEN2_5_32B_INSTRUCT;
        self
    }
    fn qwen2_5_14b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::QWEN2_5_14B_INSTRUCT;
        self
    }
    fn qwen2_5_3b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::QWEN2_5_3B_INSTRUCT;
        self
    }
    fn granite_3_0_8b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::GRANITE_3_0_8B_INSTRUCT;
        self
    }
    fn granite_3_0_2b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::GRANITE_3_0_2B_INSTRUCT;
        self
    }
    fn supernova_medius(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::SUPERNOVA_MEDIUS;
        self
    }
    fn llama_3_1_nemotron_70b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::LLAMA_3_1_NEMOTRON_70B_INSTRUCT;
        self
    }
    fn llama_3_1_nemotron_51b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::LLAMA_3_1_NEMOTRON_51B_INSTRUCT;
        self
    }
    fn mistral_nemo_minitron_8b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::MISTRAL_NEMO_MINITRON_8B_INSTRUCT;
        self
    }
    fn phi_3_5_mini_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::PHI_3_5_MINI_INSTRUCT;
        self
    }
    fn phi_3_medium_4k_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::PHI_3_MEDIUM_4K_INSTRUCT;
        self
    }
    fn phi_4_mini_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::PHI_4_MINI_INSTRUCT_;
        self
    }
    fn phi_3_mini_4k_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::PHI_3_MINI_4K_INSTRUCT;
        self
    }
    fn phi_4(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::PHI_4;
        self
    }
    fn phi_3_5_moe_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        *self.preset() = GgufPreset::PHI_3_5_MOE_INSTRUCT;
        self
    }
}
