use super::*;
impl GgufPreset {
    pub fn all_models() -> Vec<Self> {
        vec![
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
            Self::PHI_3_MINI_4K_INSTRUCT,
            Self::PHI_4,
            Self::PHI_3_5_MOE_INSTRUCT,
        ]
    }
    pub const LLAMA_3_2_1B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::META,
        model_id: "Llama-3.2-1B-Instruct",
        friendly_name: "Llama 3.2 1B Instruct",
        model_repo_id: "meta-llama/Llama-3.2-1B-Instruct",
        gguf_repo_id: "bartowski/Llama-3.2-1B-Instruct-GGUF",
        number_of_parameters: 3f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("meta.llama3.1.8b.instruct.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 2048usize,
            feed_forward_length: Some(8192u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 16usize,
            torch_dtype: "bfloat16",
            vocab_size: 128256usize,
            architecture: "llama",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Llama-3.2-1B-Instruct-IQ3_M.gguf",
                total_bytes: 657289344usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                total_bytes: 807694464usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Llama-3.2-1B-Instruct-Q5_K_M.gguf",
                total_bytes: 911503488usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Llama-3.2-1B-Instruct-Q6_K.gguf",
                total_bytes: 1021800576usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Llama-3.2-1B-Instruct-Q8_0.gguf",
                total_bytes: 1321083008usize,
            },
        ],
    };
    pub const LLAMA_3_2_3B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::META,
        model_id: "Llama-3.2-3B-Instruct",
        friendly_name: "Llama 3.2 3B Instruct",
        model_repo_id: "meta-llama/Llama-3.2-3B-Instruct",
        gguf_repo_id: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        number_of_parameters: 3f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("meta.llama3.1.8b.instruct.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 3072usize,
            feed_forward_length: Some(8192u64 as usize),
            head_count: 24usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 28usize,
            torch_dtype: "bfloat16",
            vocab_size: 128256usize,
            architecture: "llama",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Llama-3.2-3B-Instruct-IQ3_M.gguf",
                total_bytes: 1599668768usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                total_bytes: 2019377696usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
                total_bytes: 2322154016usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Llama-3.2-3B-Instruct-Q6_K.gguf",
                total_bytes: 2643853856usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Llama-3.2-3B-Instruct-Q8_0.gguf",
                total_bytes: 3421899296usize,
            },
        ],
    };
    pub const LLAMA_3_1_8B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::META,
        model_id: "Llama-3.1-8B-Instruct-GGUF",
        friendly_name: "Llama 3.1 8B Instruct",
        model_repo_id: "meta-llama/Llama-3.1-8B-Instruct-GGUF",
        gguf_repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        number_of_parameters: 8f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("meta.llama3.1.8b.instruct.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 4096usize,
            feed_forward_length: Some(14336u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 32usize,
            torch_dtype: "bfloat16",
            vocab_size: 128256usize,
            architecture: "llama",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf",
                total_bytes: 3179136416usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf",
                total_bytes: 4018922912usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                total_bytes: 4920739232usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                total_bytes: 5732992416usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
                total_bytes: 6596011424usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                total_bytes: 8540775840usize,
            },
        ],
    };
    pub const MIXTRAL_8X7B_INSTRUCT_V0_1: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_id: "Mixtral-8x7B-Instruct-v0.1",
        friendly_name: "Mixtral 8x7B Instruct v0.1",
        model_repo_id: "nvidia/Mixtral-8x7B-Instruct-v0.1",
        gguf_repo_id: "MaziyarPanahi/Mixtral-8x7B-Instruct-v0.1-GGUF",
        number_of_parameters: 56f64,
        model_ctx_size: 32768usize,
        tokenizer_path: Some("mistral.mixtral8x7b.instruct.v0.1.tokenizer.json"),
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 4096usize,
            feed_forward_length: Some(14336u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 32usize,
            torch_dtype: "bfloat16",
            vocab_size: 32768usize,
            architecture: "mistral",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Mixtral-8x7B-Instruct-v0.1.Q2_K.gguf",
                total_bytes: 17309173632usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Mixtral-8x7B-Instruct-v0.1.Q3_K_M.gguf",
                total_bytes: 22544394112usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf",
                total_bytes: 28446410624usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf",
                total_bytes: 33227523968usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Mixtral-8x7B-Instruct-v0.1.Q6_K.gguf",
                total_bytes: 38378760064usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf",
                total_bytes: 49624262528usize,
            },
        ],
    };
    pub const MISTRAL_7B_INSTRUCT_V0_3: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_id: "Mistral-7B-Instruct-v0.3",
        friendly_name: "Mistral 7B Instruct v0.3",
        model_repo_id: "mistral/Mistral-7B-Instruct-v0.3",
        gguf_repo_id: "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
        number_of_parameters: 7f64,
        model_ctx_size: 32768usize,
        tokenizer_path: Some("mistral.mistral7b.instruct.v0.3.tokenizer.json"),
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 4096usize,
            feed_forward_length: Some(14336u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 32usize,
            torch_dtype: "bfloat16",
            vocab_size: 32768usize,
            architecture: "mistral",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 1u8,
                fname: "Mistral-7B-Instruct-v0.3.IQ1_M.gguf",
                total_bytes: 1757663392usize,
            },
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Mistral-7B-Instruct-v0.3.Q2_K.gguf",
                total_bytes: 2722877600usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Mistral-7B-Instruct-v0.3.Q3_K_M.gguf",
                total_bytes: 3522941088usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
                total_bytes: 4372811936usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
                total_bytes: 5136175264usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Mistral-7B-Instruct-v0.3.Q6_K.gguf",
                total_bytes: 5947248800usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Mistral-7B-Instruct-v0.3.Q8_0.gguf",
                total_bytes: 7702565024usize,
            },
        ],
    };
    pub const MISTRAL_NEMO_INSTRUCT_2407: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_id: "Mistral-Nemo-Instruct-2407",
        friendly_name: "Mistral Nemo Instruct 2407",
        model_repo_id: "mistral/Mistral-Nemo-Instruct-2407",
        gguf_repo_id: "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        number_of_parameters: 12f64,
        model_ctx_size: 1024000usize,
        tokenizer_path: Some("mistral.mistral.nemo.instruct.2407.tokenizer.json"),
        config: ConfigJson {
            context_length: 1024000usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(14336u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 131072usize,
            architecture: "mistral",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Mistral-Nemo-Instruct-2407-Q2_K.gguf",
                total_bytes: 4791051392usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Mistral-Nemo-Instruct-2407-Q3_K_M.gguf",
                total_bytes: 6083093632usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
                total_bytes: 7477208192usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
                total_bytes: 8727635072usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Mistral-Nemo-Instruct-2407-Q6_K.gguf",
                total_bytes: 10056213632usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Mistral-Nemo-Instruct-2407-Q8_0.gguf",
                total_bytes: 13022372992usize,
            },
        ],
    };
    pub const MISTRAL_SMALL_24B_INSTRUCT_2501: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_id: "Mistral-Small-24B-Instruct-2501",
        friendly_name: "Mistral Small 24B Instruct 2501",
        model_repo_id: "mistral/Mistral-Small-24B-Instruct-2501",
        gguf_repo_id: "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        number_of_parameters: 24f64,
        model_ctx_size: 32768usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(32768u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 131072usize,
            architecture: "mistral",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Mistral-Small-24B-Instruct-2501-Q2_K.gguf",
                total_bytes: 8890324672usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Mistral-Small-24B-Instruct-2501-Q3_K_M.gguf",
                total_bytes: 11474081472usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
                total_bytes: 14333908672usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Mistral-Small-24B-Instruct-2501-Q5_K_M.gguf",
                total_bytes: 16763983552usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Mistral-Small-24B-Instruct-2501-Q6_K.gguf",
                total_bytes: 19345938112usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Mistral-Small-24B-Instruct-2501-Q8_0.gguf",
                total_bytes: 25054779072usize,
            },
        ],
    };
    pub const MISTRAL_SMALL_INSTRUCT_2409: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MISTRAL,
        model_id: "Mistral-Small-Instruct-2409",
        friendly_name: "Mistral Small Instruct 2409",
        model_repo_id: "mistral/Mistral-Small-Instruct-2409",
        gguf_repo_id: "bartowski/Mistral-Small-Instruct-2409-GGUF",
        number_of_parameters: 12f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("mistral.mistral.small.instruct.2409.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 6144usize,
            feed_forward_length: Some(16384u64 as usize),
            head_count: 48usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 56usize,
            torch_dtype: "bfloat16",
            vocab_size: 32768usize,
            architecture: "mistral",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Mistral-Small-Instruct-2409-Q2_K.gguf",
                total_bytes: 8272098304usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Mistral-Small-Instruct-2409-Q3_K_M.gguf",
                total_bytes: 10756830208usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Mistral-Small-Instruct-2409-Q4_K_M.gguf",
                total_bytes: 13341242368usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Mistral-Small-Instruct-2409-Q5_K_M.gguf",
                total_bytes: 15722558464usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Mistral-Small-Instruct-2409-Q6_K.gguf",
                total_bytes: 18252706816usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Mistral-Small-Instruct-2409-Q8_0.gguf",
                total_bytes: 23640552448usize,
            },
        ],
    };
    pub const STABLE_LM_2_12B_CHAT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::STABILITY_AI,
        model_id: "stablelm-2-12b-chat",
        friendly_name: "Stable LM 2 12B Chat",
        model_repo_id: "stabilityai/stablelm-2-12b-chat",
        gguf_repo_id: "second-state/stablelm-2-12b-chat-GGUF",
        number_of_parameters: 12f64,
        model_ctx_size: 4096usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 4096usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(13824u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 100352usize,
            architecture: "stablelm",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "stablelm-2-12b-chat-Q2_K.gguf",
                total_bytes: 4698894176usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "stablelm-2-12b-chat-Q3_K_M.gguf",
                total_bytes: 5993885536usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "stablelm-2-12b-chat-Q4_K_M.gguf",
                total_bytes: 7367642976usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "stablelm-2-12b-chat-Q5_K_M.gguf",
                total_bytes: 8627900256usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "stablelm-2-12b-chat-Q6_K.gguf",
                total_bytes: 9966923616usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "stablelm-2-12b-chat-Q8_0.gguf",
                total_bytes: 12907687776usize,
            },
        ],
    };
    pub const QWEN2_5_7B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_id: "Qwen2.5-7B-Instruct",
        friendly_name: "Qwen2.5 7B Instruct",
        model_repo_id: "Qwen/Qwen2.5-7B-Instruct",
        gguf_repo_id: "bartowski/Qwen2.5-7B-Instruct-GGUF",
        number_of_parameters: 7f64,
        model_ctx_size: 32768usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 3584usize,
            feed_forward_length: Some(18944u64 as usize),
            head_count: 28usize,
            head_count_kv: Some(4u64 as usize),
            block_count: 28usize,
            torch_dtype: "bfloat16",
            vocab_size: 152064usize,
            architecture: "qwen2",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Qwen2.5-7B-Instruct-Q2_K.gguf",
                total_bytes: 3015940800usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Qwen2.5-7B-Instruct-Q3_K_M.gguf",
                total_bytes: 3808391872usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                total_bytes: 4683074240usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
                total_bytes: 5444831936usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Qwen2.5-7B-Instruct-Q6_K.gguf",
                total_bytes: 6254199488usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Qwen2.5-7B-Instruct-Q8_0.gguf",
                total_bytes: 8098525888usize,
            },
        ],
    };
    pub const QWEN2_5_32B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_id: "Qwen2.5-32B-Instruct",
        friendly_name: "Qwen2.5 32B Instruct",
        model_repo_id: "Qwen/Qwen2.5-32B-Instruct",
        gguf_repo_id: "bartowski/Qwen2.5-32B-Instruct-GGUF",
        number_of_parameters: 32f64,
        model_ctx_size: 32768usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(27648u64 as usize),
            head_count: 40usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 64usize,
            torch_dtype: "bfloat16",
            vocab_size: 152064usize,
            architecture: "qwen2",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Qwen2.5-32B-Instruct-Q2_K.gguf",
                total_bytes: 12313099136usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Qwen2.5-32B-Instruct-Q3_K_M.gguf",
                total_bytes: 15935048576usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
                total_bytes: 19851336576usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Qwen2.5-32B-Instruct-Q5_K_M.gguf",
                total_bytes: 23262157696usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Qwen2.5-32B-Instruct-Q6_K.gguf",
                total_bytes: 26886155136usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Qwen2.5-32B-Instruct-Q8_0.gguf",
                total_bytes: 34820885376usize,
            },
        ],
    };
    pub const QWEN2_5_14B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_id: "Qwen2.5-14B-Instruct",
        friendly_name: "Qwen2.5 14B Instruct",
        model_repo_id: "Qwen/Qwen2.5-14B-Instruct",
        gguf_repo_id: "bartowski/Qwen2.5-14B-Instruct-GGUF",
        number_of_parameters: 14f64,
        model_ctx_size: 32768usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(13824u64 as usize),
            head_count: 40usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 48usize,
            torch_dtype: "bfloat16",
            vocab_size: 152064usize,
            architecture: "qwen2",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Qwen2.5-14B-Instruct-Q2_K.gguf",
                total_bytes: 5770498176usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Qwen2.5-14B-Instruct-Q3_K_M.gguf",
                total_bytes: 7339204736usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                total_bytes: 8988110976usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Qwen2.5-14B-Instruct-Q5_K_M.gguf",
                total_bytes: 10508873856usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Qwen2.5-14B-Instruct-Q6_K.gguf",
                total_bytes: 12124684416usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Qwen2.5-14B-Instruct-Q8_0.gguf",
                total_bytes: 15701598336usize,
            },
        ],
    };
    pub const QWEN2_5_3B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ALIBABA,
        model_id: "Qwen2.5-3B-Instruct",
        friendly_name: "Qwen2.5 3B Instruct",
        model_repo_id: "Qwen/Qwen2.5-3B-Instruct",
        gguf_repo_id: "bartowski/Qwen2.5-3B-Instruct-GGUF",
        number_of_parameters: 3f64,
        model_ctx_size: 32768usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 32768usize,
            embedding_length: 2048usize,
            feed_forward_length: Some(11008u64 as usize),
            head_count: 16usize,
            head_count_kv: Some(2u64 as usize),
            block_count: 36usize,
            torch_dtype: "bfloat16",
            vocab_size: 151936usize,
            architecture: "qwen2",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Qwen2.5-3B-Instruct-Q2_K.gguf",
                total_bytes: 1274756256usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Qwen2.5-3B-Instruct-Q3_K_M.gguf",
                total_bytes: 1590475936usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                total_bytes: 1929903264usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Qwen2.5-3B-Instruct-Q5_K_M.gguf",
                total_bytes: 2224815264usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Qwen2.5-3B-Instruct-Q6_K.gguf",
                total_bytes: 2538159264usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Qwen2.5-3B-Instruct-Q8_0.gguf",
                total_bytes: 3285476512usize,
            },
        ],
    };
    pub const GRANITE_3_0_8B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::IBM,
        model_id: "granite-3.0-8b-instruct",
        friendly_name: "Granite 3.0 8b instruct",
        model_repo_id: "ibm/granite-3.0-8b-instruct",
        gguf_repo_id: "bartowski/granite-3.0-8b-instruct-GGUF",
        number_of_parameters: 8f64,
        model_ctx_size: 4096usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 4096usize,
            embedding_length: 4096usize,
            feed_forward_length: Some(12800u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 49155usize,
            architecture: "granite",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "granite-3.0-8b-instruct-Q2_K.gguf",
                total_bytes: 3103588576usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "granite-3.0-8b-instruct-Q3_K_L.gguf",
                total_bytes: 4349427936usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "granite-3.0-8b-instruct-Q4_K_M.gguf",
                total_bytes: 4942856416usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "granite-3.0-8b-instruct-Q5_K_M.gguf",
                total_bytes: 5797445856usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "granite-3.0-8b-instruct-Q6_K.gguf",
                total_bytes: 6705447136usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "granite-3.0-8b-instruct-Q8_0.gguf",
                total_bytes: 8684244096usize,
            },
        ],
    };
    pub const GRANITE_3_0_2B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::IBM,
        model_id: "granite-3.0-2b-instruct",
        friendly_name: "Granite 3.0 2b instruct",
        model_repo_id: "ibm/granite-3.0-2b-instruct",
        gguf_repo_id: "bartowski/granite-3.0-2b-instruct-GGUF",
        number_of_parameters: 2f64,
        model_ctx_size: 4096usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 4096usize,
            embedding_length: 2048usize,
            feed_forward_length: Some(8192u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 49155usize,
            architecture: "granite",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "granite-3.0-2b-instruct-Q2_K.gguf",
                total_bytes: 1011275040usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "granite-3.0-2b-instruct-Q3_K_L.gguf",
                total_bytes: 1400625056usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "granite-3.0-2b-instruct-Q4_K_M.gguf",
                total_bytes: 1601919680usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "granite-3.0-2b-instruct-Q5_K_M.gguf",
                total_bytes: 1874025920usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "granite-3.0-2b-instruct-Q6_K.gguf",
                total_bytes: 2163138816usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "granite-3.0-2b-instruct-Q8_0.gguf",
                total_bytes: 2801069184usize,
            },
        ],
    };
    pub const SUPERNOVA_MEDIUS: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::ARCEE_AI,
        model_id: "SuperNova-Medius",
        friendly_name: "SuperNova Medius",
        model_repo_id: "arcee-ai/arcee-ai/SuperNova-Medius",
        gguf_repo_id: "arcee-ai/SuperNova-Medius-GGUF",
        number_of_parameters: 13f64,
        model_ctx_size: 131072usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(13824u64 as usize),
            head_count: 40usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 48usize,
            torch_dtype: "bfloat16",
            vocab_size: 152064usize,
            architecture: "qwen2",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "SuperNova-Medius-Q2_K.gguf",
                total_bytes: 5770498592usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "SuperNova-Medius-Q3_K_M.gguf",
                total_bytes: 7339205152usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "SuperNova-Medius-Q4_K_M.gguf",
                total_bytes: 8988111392usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "SuperNova-Medius-Q5_K_M.gguf",
                total_bytes: 10508874272usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "SuperNova-Medius-Q6_K.gguf",
                total_bytes: 12124684832usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "SuperNova-Medius-Q8_0.gguf",
                total_bytes: 15701598752usize,
            },
        ],
    };
    pub const LLAMA_3_1_NEMOTRON_70B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::NVIDIA,
        model_id: "Llama-3.1-Nemotron-70B-Instruct",
        friendly_name: "Llama 3.1 Nemotron 70B Instruct",
        model_repo_id: "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        gguf_repo_id: "bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF",
        number_of_parameters: 70f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("meta.llama3.1.8b.instruct.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 8192usize,
            feed_forward_length: Some(28672u64 as usize),
            head_count: 64usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 80usize,
            torch_dtype: "bfloat16",
            vocab_size: 128256usize,
            architecture: "llama",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Llama-3.1-Nemotron-70B-Instruct-HF-Q2_K.gguf",
                total_bytes: 26375113632usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Llama-3.1-Nemotron-70B-Instruct-HF-Q3_K_M.gguf",
                total_bytes: 34267499424usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf",
                total_bytes: 42520398752usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_S.gguf",
                total_bytes: 48657451936usize,
            },
        ],
    };
    pub const LLAMA_3_1_NEMOTRON_51B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::NVIDIA,
        model_id: "Llama-3_1-Nemotron-51B-Instruct",
        friendly_name: "Llama 3.1 Nemotron 51B Instruct",
        model_repo_id: "nvidia/Llama-3_1-Nemotron-51B-Instruct",
        gguf_repo_id: "bartowski/Llama-3_1-Nemotron-51B-Instruct-GGUF",
        number_of_parameters: 52f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("meta.llama3.1.8b.instruct.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 8192usize,
            feed_forward_length: None,
            head_count: 64usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 80usize,
            torch_dtype: "bfloat16",
            vocab_size: 128256usize,
            architecture: "nemotron-nas",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Llama-3_1-Nemotron-51B-Instruct-Q2_K.gguf",
                total_bytes: 19418642688usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Llama-3_1-Nemotron-51B-Instruct-Q3_K_M.gguf",
                total_bytes: 25182345472usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Llama-3_1-Nemotron-51B-Instruct-Q4_K_M.gguf",
                total_bytes: 31037307136usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Llama-3_1-Nemotron-51B-Instruct-Q5_K_M.gguf",
                total_bytes: 36465391872usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Llama-3_1-Nemotron-51B-Instruct-Q6_K.gguf",
                total_bytes: 42258774272usize,
            },
        ],
    };
    pub const MISTRAL_NEMO_MINITRON_8B_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::NVIDIA,
        model_id: "Mistral-NeMo-Minitron-8B-Instruct",
        friendly_name: "Mistral NeMo Minitron 8B Instruct",
        model_repo_id: "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
        gguf_repo_id: "bartowski/Mistral-NeMo-Minitron-8B-Instruct-GGUF",
        number_of_parameters: 8f64,
        model_ctx_size: 8192usize,
        tokenizer_path: None,
        config: ConfigJson {
            context_length: 8192usize,
            embedding_length: 4096usize,
            feed_forward_length: Some(11520u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 131072usize,
            architecture: "mistral",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Mistral-NeMo-Minitron-8B-Instruct-Q2_K.gguf",
                total_bytes: 3333392064usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Mistral-NeMo-Minitron-8B-Instruct-Q3_K_M.gguf",
                total_bytes: 4209149632usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Mistral-NeMo-Minitron-8B-Instruct-Q4_K_M.gguf",
                total_bytes: 5145298624usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Mistral-NeMo-Minitron-8B-Instruct-Q5_K_M.gguf",
                total_bytes: 6001460928usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Mistral-NeMo-Minitron-8B-Instruct-Q6_K.gguf",
                total_bytes: 6911133376usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Mistral-NeMo-Minitron-8B-Instruct-Q8_0.gguf",
                total_bytes: 8948844224usize,
            },
        ],
    };
    pub const PHI_3_5_MINI_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_id: "Phi-3.5-mini-instruct",
        friendly_name: "Phi 3.5 mini instruct",
        model_repo_id: "microsoft/Phi-3.5-mini-instruct",
        gguf_repo_id: "bartowski/Phi-3.5-mini-instruct-GGUF",
        number_of_parameters: 4f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("microsoft.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 3072usize,
            feed_forward_length: Some(8192u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(32u64 as usize),
            block_count: 32usize,
            torch_dtype: "bfloat16",
            vocab_size: 32064usize,
            architecture: "phi3",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Phi-3.5-mini-instruct-Q2_K.gguf",
                total_bytes: 1416204576usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Phi-3.5-mini-instruct-Q3_K_M.gguf",
                total_bytes: 1955477280usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                total_bytes: 2393232672usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Phi-3.5-mini-instruct-Q5_K_M.gguf",
                total_bytes: 2815276320usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Phi-3.5-mini-instruct-Q6_K.gguf",
                total_bytes: 3135853344usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Phi-3.5-mini-instruct-Q8_0.gguf",
                total_bytes: 4061222688usize,
            },
        ],
    };
    pub const PHI_3_MEDIUM_4K_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_id: "Phi-3-medium-4k-instruct",
        friendly_name: "Phi 3 medium 4k instruct",
        model_repo_id: "microsoft/Phi-3-medium-4k-instruct",
        gguf_repo_id: "bartowski/Phi-3-medium-4k-instruct-GGUF",
        number_of_parameters: 14f64,
        model_ctx_size: 4096usize,
        tokenizer_path: Some("microsoft.tokenizer.json"),
        config: ConfigJson {
            context_length: 4096usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(17920u64 as usize),
            head_count: 40usize,
            head_count_kv: Some(10u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 32064usize,
            architecture: "phi3",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Phi-3-medium-4k-instruct-Q2_K.gguf",
                total_bytes: 5143000448usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Phi-3-medium-4k-instruct-Q3_K_M.gguf",
                total_bytes: 6923411328usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Phi-3-medium-4k-instruct-Q4_K_M.gguf",
                total_bytes: 8566821248usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Phi-3-medium-4k-instruct-Q5_K_M.gguf",
                total_bytes: 10074190208usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Phi-3-medium-4k-instruct-Q6_K.gguf",
                total_bytes: 11453817728usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Phi-3-medium-4k-instruct-Q8_0.gguf",
                total_bytes: 14834712448usize,
            },
        ],
    };
    pub const PHI_3_MINI_4K_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_id: "Phi-3-mini-4k-instruct",
        friendly_name: "Phi 3 mini 4k instruct",
        model_repo_id: "microsoft/Phi-3-mini-4k-instruct",
        gguf_repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
        number_of_parameters: 4f64,
        model_ctx_size: 4096usize,
        tokenizer_path: Some("microsoft.tokenizer.json"),
        config: ConfigJson {
            context_length: 4096usize,
            embedding_length: 3072usize,
            feed_forward_length: Some(8192u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(32u64 as usize),
            block_count: 32usize,
            torch_dtype: "bfloat16",
            vocab_size: 32064usize,
            architecture: "phi3",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 1u8,
                fname: "Phi-3-mini-4k-instruct-IQ1_M.gguf",
                total_bytes: 917106176usize,
            },
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Phi-3-mini-4k-instruct-Q2_K.gguf",
                total_bytes: 1416203264usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Phi-3-mini-4k-instruct-Q3_K_M.gguf",
                total_bytes: 1955475968usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
                total_bytes: 2393231360usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Phi-3-mini-4k-instruct-Q5_K_M.gguf",
                total_bytes: 2815275008usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Phi-3-mini-4k-instruct-Q6_K.gguf",
                total_bytes: 3135852032usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Phi-3-mini-4k-instruct-Q8_0.gguf",
                total_bytes: 4061221376usize,
            },
        ],
    };
    pub const PHI_4: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_id: "phi-4",
        friendly_name: "Phi-4",
        model_repo_id: "microsoft/phi-4",
        gguf_repo_id: "bartowski/phi-4-GGUF",
        number_of_parameters: 14f64,
        model_ctx_size: 16384usize,
        tokenizer_path: Some("microsoft.phi4.tokenizer.json"),
        config: ConfigJson {
            context_length: 16384usize,
            embedding_length: 5120usize,
            feed_forward_length: Some(17920u64 as usize),
            head_count: 40usize,
            head_count_kv: Some(10u64 as usize),
            block_count: 40usize,
            torch_dtype: "bfloat16",
            vocab_size: 100352usize,
            architecture: "phi3",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "phi-4-Q2_K.gguf",
                total_bytes: 5547348416usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "phi-4-Q3_K_M.gguf",
                total_bytes: 7363269056usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "phi-4-Q4_K_M.gguf",
                total_bytes: 9053114816usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "phi-4-Q5_K_M.gguf",
                total_bytes: 10604188096usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "phi-4-Q6_K.gguf",
                total_bytes: 12030251456usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "phi-4-Q8_0.gguf",
                total_bytes: 15580500416usize,
            },
        ],
    };
    pub const PHI_3_5_MOE_INSTRUCT: GgufPreset = GgufPreset {
        organization: LocalLlmOrganization::MICROSOFT,
        model_id: "Phi-3.5-MoE-instruct",
        friendly_name: "Phi 3.5 MoE instruct",
        model_repo_id: "microsoft/Phi-3.5-MoE-instruct",
        gguf_repo_id: "bartowski/Phi-3.5-MoE-instruct-GGUF",
        number_of_parameters: 7f64,
        model_ctx_size: 131072usize,
        tokenizer_path: Some("microsoft.tokenizer.json"),
        config: ConfigJson {
            context_length: 131072usize,
            embedding_length: 4096usize,
            feed_forward_length: Some(6400u64 as usize),
            head_count: 32usize,
            head_count_kv: Some(8u64 as usize),
            block_count: 32usize,
            torch_dtype: "bfloat16",
            vocab_size: 32064usize,
            architecture: "phimoe",
            model_size_bytes: None,
        },
        quants: &[
            GgufPresetQuant {
                q_lvl: 2u8,
                fname: "Phi-3.5-MoE-instruct-Q2_K.gguf",
                total_bytes: 15265136480usize,
            },
            GgufPresetQuant {
                q_lvl: 3u8,
                fname: "Phi-3.5-MoE-instruct-Q3_K_M.gguf",
                total_bytes: 20032718688usize,
            },
            GgufPresetQuant {
                q_lvl: 4u8,
                fname: "Phi-3.5-MoE-instruct-Q4_K_M.gguf",
                total_bytes: 25345994592usize,
            },
            GgufPresetQuant {
                q_lvl: 5u8,
                fname: "Phi-3.5-MoE-instruct-Q5_K_M.gguf",
                total_bytes: 29716098912usize,
            },
            GgufPresetQuant {
                q_lvl: 6u8,
                fname: "Phi-3.5-MoE-instruct-Q6_K.gguf",
                total_bytes: 34359334752usize,
            },
            GgufPresetQuant {
                q_lvl: 8u8,
                fname: "Phi-3.5-MoE-instruct-Q8_0.gguf",
                total_bytes: 44499765088usize,
            },
        ],
    };
}
pub trait GgufPresetTrait {
    fn preset_loader(&mut self) -> &mut GgufPresetLoader;
    fn preset_from_str(mut self, selected_model_id: &str) -> Result<Self, crate::Error>
    where
        Self: Sized,
    {
        let preset = GgufPreset::all_models()
            .into_iter()
            .find(|preset| preset.model_id == selected_model_id)
            .ok_or_else(|| crate::anyhow!("Invalid selected_model_id: {}", selected_model_id))?;
        self.preset_loader().llm_preset = preset;
        Ok(self)
    }
    fn preset_with_memory_gb(mut self, preset_with_memory_gb: usize) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().preset_with_memory_gb = Some(preset_with_memory_gb);
        self
    }
    fn preset_with_quantization_level(mut self, level: u8) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().preset_with_quantization_level = Some(level);
        self
    }
    fn llama_3_2_1b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::LLAMA_3_2_1B_INSTRUCT;
        self
    }
    fn llama_3_2_3b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::LLAMA_3_2_3B_INSTRUCT;
        self
    }
    fn llama_3_1_8b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::LLAMA_3_1_8B_INSTRUCT;
        self
    }
    fn mixtral_8x7b_instruct_v0_1(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::MIXTRAL_8X7B_INSTRUCT_V0_1;
        self
    }
    fn mistral_7b_instruct_v0_3(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::MISTRAL_7B_INSTRUCT_V0_3;
        self
    }
    fn mistral_nemo_instruct_2407(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::MISTRAL_NEMO_INSTRUCT_2407;
        self
    }
    fn mistral_small_24b_instruct_2501(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::MISTRAL_SMALL_24B_INSTRUCT_2501;
        self
    }
    fn mistral_small_instruct_2409(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::MISTRAL_SMALL_INSTRUCT_2409;
        self
    }
    fn stable_lm_2_12b_chat(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::STABLE_LM_2_12B_CHAT;
        self
    }
    fn qwen2_5_7b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::QWEN2_5_7B_INSTRUCT;
        self
    }
    fn qwen2_5_32b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::QWEN2_5_32B_INSTRUCT;
        self
    }
    fn qwen2_5_14b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::QWEN2_5_14B_INSTRUCT;
        self
    }
    fn qwen2_5_3b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::QWEN2_5_3B_INSTRUCT;
        self
    }
    fn granite_3_0_8b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::GRANITE_3_0_8B_INSTRUCT;
        self
    }
    fn granite_3_0_2b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::GRANITE_3_0_2B_INSTRUCT;
        self
    }
    fn supernova_medius(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::SUPERNOVA_MEDIUS;
        self
    }
    fn llama_3_1_nemotron_70b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::LLAMA_3_1_NEMOTRON_70B_INSTRUCT;
        self
    }
    fn llama_3_1_nemotron_51b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::LLAMA_3_1_NEMOTRON_51B_INSTRUCT;
        self
    }
    fn mistral_nemo_minitron_8b_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::MISTRAL_NEMO_MINITRON_8B_INSTRUCT;
        self
    }
    fn phi_3_5_mini_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::PHI_3_5_MINI_INSTRUCT;
        self
    }
    fn phi_3_medium_4k_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::PHI_3_MEDIUM_4K_INSTRUCT;
        self
    }
    fn phi_3_mini_4k_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::PHI_3_MINI_4K_INSTRUCT;
        self
    }
    fn phi_4(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::PHI_4;
        self
    }
    fn phi_3_5_moe_instruct(mut self) -> Self
    where
        Self: Sized,
    {
        self.preset_loader().llm_preset = GgufPreset::PHI_3_5_MOE_INSTRUCT;
        self
    }
}
