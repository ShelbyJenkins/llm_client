use super::*;
impl CloudLlm {
    pub const ALL_MODELS: [CloudLlm; 27usize] = [
        Self::CLAUDE_3_OPUS,
        Self::CLAUDE_3_SONNET,
        Self::CLAUDE_3_HAIKU,
        Self::CLAUDE_3_5_SONNET,
        Self::CLAUDE_3_7_SONNET,
        Self::CLAUDE_3_7_SONNET_THINKING,
        Self::CLAUDE_3_5_HAIKU,
        Self::GPT_4,
        Self::GPT_3_5_TURBO,
        Self::GPT_4_32K,
        Self::GPT_4_TURBO,
        Self::GPT_4O,
        Self::GPT_4O_MINI,
        Self::O1,
        Self::O1_MINI,
        Self::O3_MINI,
        Self::SONAR_REASONING_PRO,
        Self::SONAR_REASONING,
        Self::SONAR_PRO,
        Self::SONAR,
        Self::MISTRAL_LARGE,
        Self::MINISTRAL_3B,
        Self::MINISTRAL_8B,
        Self::MISTRAL_NEMO,
        Self::MISTRAL_SMALL,
        Self::MISTRAL_SABA,
        Self::CODESTRAL,
    ];
    pub const CLAUDE_3_OPUS: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude3OpusLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-opus-latest"),
            friendly_name: Cow::Borrowed("Claude 3 Opus"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 1500u64,
        cost_per_m_out_tokens: 7500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_SONNET: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude3Sonnet20240229),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-sonnet-20240229"),
            friendly_name: Cow::Borrowed("Claude 3 Sonnet"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 300u64,
        cost_per_m_out_tokens: 1500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_HAIKU: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude3Haiku20240307),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-haiku-20240307"),
            friendly_name: Cow::Borrowed("Claude 3 Haiku"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 75u64,
        cost_per_m_out_tokens: 125u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_5_SONNET: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude35SonnetLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-5-sonnet-latest"),
            friendly_name: Cow::Borrowed("Claude 3.5 Sonnet"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 8192u64,
        },
        cost_per_m_in_tokens: 300u64,
        cost_per_m_out_tokens: 1500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_7_SONNET: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude37SonnetLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-7-sonnet-latest"),
            friendly_name: Cow::Borrowed("Claude 3.7 Sonnet"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 8192u64,
        },
        cost_per_m_in_tokens: 300u64,
        cost_per_m_out_tokens: 1500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_7_SONNET_THINKING: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude37SonnetThinkingLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-7-sonnet-thinking-latest"),
            friendly_name: Cow::Borrowed("Claude 3.7 Sonnet Thinking"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 64000u64,
        },
        cost_per_m_in_tokens: 300u64,
        cost_per_m_out_tokens: 1500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const CLAUDE_3_5_HAIKU: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Anthropic(Claude35HaikuLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("claude-3-5-haiku-latest"),
            friendly_name: Cow::Borrowed("Claude 3.5 Haiku"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 8192u64,
        },
        cost_per_m_in_tokens: 80u64,
        cost_per_m_out_tokens: 400u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const GPT_4: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(Gpt4),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("gpt-4"),
            friendly_name: Cow::Borrowed("GPT-4"),
            model_ctx_size: 8192u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 3000u64,
        cost_per_m_out_tokens: 6000u64,
        tokens_per_message: 3u64,
        tokens_per_name: Some(1i64),
    };
    pub const GPT_3_5_TURBO: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(Gpt35Turbo),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("gpt-3.5-turbo"),
            friendly_name: Cow::Borrowed("GPT-3.5 Turbo"),
            model_ctx_size: 16385u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 50u64,
        cost_per_m_out_tokens: 150u64,
        tokens_per_message: 4u64,
        tokens_per_name: Some(-1i64),
    };
    pub const GPT_4_32K: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(Gpt432k),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("gpt-4-32k"),
            friendly_name: Cow::Borrowed("GPT-4 32K"),
            model_ctx_size: 32768u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 6000u64,
        cost_per_m_out_tokens: 12000u64,
        tokens_per_message: 3u64,
        tokens_per_name: Some(1i64),
    };
    pub const GPT_4_TURBO: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(Gpt4Turbo),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("gpt-4-turbo"),
            friendly_name: Cow::Borrowed("GPT-4 Turbo"),
            model_ctx_size: 128000u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 1000u64,
        cost_per_m_out_tokens: 3000u64,
        tokens_per_message: 3u64,
        tokens_per_name: Some(1i64),
    };
    pub const GPT_4O: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(Gpt4o),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("gpt-4o"),
            friendly_name: Cow::Borrowed("GPT-4o"),
            model_ctx_size: 128000u64,
            inference_ctx_size: 4096u64,
        },
        cost_per_m_in_tokens: 250u64,
        cost_per_m_out_tokens: 1000u64,
        tokens_per_message: 3u64,
        tokens_per_name: Some(1i64),
    };
    pub const GPT_4O_MINI: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(Gpt4oMini),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("gpt-4o-mini"),
            friendly_name: Cow::Borrowed("GPT-4o Mini"),
            model_ctx_size: 128000u64,
            inference_ctx_size: 16384u64,
        },
        cost_per_m_in_tokens: 15u64,
        cost_per_m_out_tokens: 60u64,
        tokens_per_message: 3u64,
        tokens_per_name: Some(1i64),
    };
    pub const O1: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(O1),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("o1"),
            friendly_name: Cow::Borrowed("O1"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 100000u64,
        },
        cost_per_m_in_tokens: 1500u64,
        cost_per_m_out_tokens: 6000u64,
        tokens_per_message: 4u64,
        tokens_per_name: Some(-1i64),
    };
    pub const O1_MINI: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(O1Mini),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("o1-mini"),
            friendly_name: Cow::Borrowed("o1 Mini"),
            model_ctx_size: 128000u64,
            inference_ctx_size: 65536u64,
        },
        cost_per_m_in_tokens: 110u64,
        cost_per_m_out_tokens: 440u64,
        tokens_per_message: 4u64,
        tokens_per_name: Some(-1i64),
    };
    pub const O3_MINI: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::OpenAi(O3Mini),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("o3-mini"),
            friendly_name: Cow::Borrowed("o3 Mini"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 100000u64,
        },
        cost_per_m_in_tokens: 110u64,
        cost_per_m_out_tokens: 440u64,
        tokens_per_message: 4u64,
        tokens_per_name: Some(-1i64),
    };
    pub const SONAR_REASONING_PRO: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Perplexity(SonarReasoningPro),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("sonar-reasoning-pro"),
            friendly_name: Cow::Borrowed("Sonar Reasoning Pro"),
            model_ctx_size: 127000u64,
            inference_ctx_size: 8000u64,
        },
        cost_per_m_in_tokens: 200u64,
        cost_per_m_out_tokens: 8000u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const SONAR_REASONING: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Perplexity(SonarReasoning),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("sonar-reasoning"),
            friendly_name: Cow::Borrowed("Sonar Reasoning"),
            model_ctx_size: 127000u64,
            inference_ctx_size: 8000u64,
        },
        cost_per_m_in_tokens: 100u64,
        cost_per_m_out_tokens: 500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const SONAR_PRO: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Perplexity(SonarPro),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("sonar-pro"),
            friendly_name: Cow::Borrowed("Sonar Pro"),
            model_ctx_size: 200000u64,
            inference_ctx_size: 8000u64,
        },
        cost_per_m_in_tokens: 300u64,
        cost_per_m_out_tokens: 1500u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const SONAR: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::Perplexity(Sonar),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("sonar"),
            friendly_name: Cow::Borrowed("Sonar"),
            model_ctx_size: 127000u64,
            inference_ctx_size: 8000u64,
        },
        cost_per_m_in_tokens: 100u64,
        cost_per_m_out_tokens: 100u64,
        tokens_per_message: 3u64,
        tokens_per_name: None,
    };
    pub const MISTRAL_LARGE: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(MistralLargeLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("mistral-large-latest"),
            friendly_name: Cow::Borrowed("Mistral Large"),
            model_ctx_size: 131000u64,
            inference_ctx_size: 131000u64,
        },
        cost_per_m_in_tokens: 200u64,
        cost_per_m_out_tokens: 6000u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
    pub const MINISTRAL_3B: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(Ministral3bLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("ministral-3b-latest"),
            friendly_name: Cow::Borrowed("Ministral 3B"),
            model_ctx_size: 131000u64,
            inference_ctx_size: 131000u64,
        },
        cost_per_m_in_tokens: 4u64,
        cost_per_m_out_tokens: 4u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
    pub const MINISTRAL_8B: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(Ministral8bLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("ministral-8b-latest"),
            friendly_name: Cow::Borrowed("Ministral 8B"),
            model_ctx_size: 131000u64,
            inference_ctx_size: 131000u64,
        },
        cost_per_m_in_tokens: 10u64,
        cost_per_m_out_tokens: 10u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
    pub const MISTRAL_NEMO: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(OpenMistralNemo),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("open-mistral-nemo"),
            friendly_name: Cow::Borrowed("Mistral NeMo"),
            model_ctx_size: 131000u64,
            inference_ctx_size: 131000u64,
        },
        cost_per_m_in_tokens: 15u64,
        cost_per_m_out_tokens: 15u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
    pub const MISTRAL_SMALL: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(MistralSmallLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("mistral-small-latest"),
            friendly_name: Cow::Borrowed("Mistral Small"),
            model_ctx_size: 32000u64,
            inference_ctx_size: 32000u64,
        },
        cost_per_m_in_tokens: 10u64,
        cost_per_m_out_tokens: 30u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
    pub const MISTRAL_SABA: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(MistralSabaLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("mistral-saba-latest"),
            friendly_name: Cow::Borrowed("Mistral Saba"),
            model_ctx_size: 32000u64,
            inference_ctx_size: 32000u64,
        },
        cost_per_m_in_tokens: 20u64,
        cost_per_m_out_tokens: 60u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
    pub const CODESTRAL: CloudLlm = CloudLlm {
        provider_llm_id: CloudProviderLlmId::MistralAi(CodestralLatest),
        model_base: LlmModelBase {
            model_id: Cow::Borrowed("codestral-latest"),
            friendly_name: Cow::Borrowed("Codestral"),
            model_ctx_size: 256000u64,
            inference_ctx_size: 256000u64,
        },
        cost_per_m_in_tokens: 30u64,
        cost_per_m_out_tokens: 90u64,
        tokens_per_message: 5u64,
        tokens_per_name: Some(1i64),
    };
}
