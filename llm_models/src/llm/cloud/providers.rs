use super::*;
impl CloudLlm {
    pub fn model_from_model_id(model_id: &str) -> crate::Result<CloudLlm> {
        let model_id = model_id.to_lowercase();
        let models = Self::ALL_MODELS;
        for model in &models {
            if model.model_base.model_id.to_lowercase() == model_id {
                return Ok((*model).to_owned());
            }
        }
        for model in models {
            if model_id.contains(&model.model_base.model_id.to_lowercase())
                || model.model_base.model_id.contains(&model_id.to_lowercase())
                || model
                    .model_base
                    .friendly_name
                    .to_lowercase()
                    .contains(&model_id)
            {
                return Ok(model);
            }
        }
        crate::bail!("Model ID '{}' not found", model_id)
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum CloudProviderLlmId {
    Anthropic(AnthropicLlmId),
    OpenAi(OpenAiLlmId),
    Perplexity(PerplexityLlmId),
    MistralAi(MistralAiLlmId),
}
impl CloudProviderLlmId {
    pub fn model(&self) -> CloudLlm {
        match self {
            Self::Anthropic(p) => p.model(),
            Self::OpenAi(p) => p.model(),
            Self::Perplexity(p) => p.model(),
            Self::MistralAi(p) => p.model(),
        }
    }
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Anthropic(p) => p.model_id(),
            Self::OpenAi(p) => p.model_id(),
            Self::Perplexity(p) => p.model_id(),
            Self::MistralAi(p) => p.model_id(),
        }
    }
    pub fn provider_friendly_name(&self) -> &'static str {
        match self {
            Self::Anthropic(p) => p.provider_friendly_name(),
            Self::OpenAi(p) => p.provider_friendly_name(),
            Self::Perplexity(p) => p.provider_friendly_name(),
            Self::MistralAi(p) => p.provider_friendly_name(),
        }
    }
    pub fn all_models() -> Vec<CloudLlm> {
        [
            AnthropicLlmId::all_provider_models(),
            OpenAiLlmId::all_provider_models(),
            PerplexityLlmId::all_provider_models(),
            MistralAiLlmId::all_provider_models(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
    pub fn all_provider_models(&self) -> Vec<CloudLlm> {
        match self {
            Self::Anthropic(_) => AnthropicLlmId::all_provider_models(),
            Self::OpenAi(_) => OpenAiLlmId::all_provider_models(),
            Self::Perplexity(_) => PerplexityLlmId::all_provider_models(),
            Self::MistralAi(_) => MistralAiLlmId::all_provider_models(),
        }
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum AnthropicLlmId {
    Claude3OpusLatest,
    Claude3Sonnet20240229,
    Claude3Haiku20240307,
    Claude35SonnetLatest,
    Claude37SonnetLatest,
    Claude37SonnetThinkingLatest,
    Claude35HaikuLatest,
}
impl AnthropicLlmId {
    pub fn model(&self) -> CloudLlm {
        match self {
            Self::Claude3OpusLatest => CloudLlm::CLAUDE_3_OPUS,
            Self::Claude3Sonnet20240229 => CloudLlm::CLAUDE_3_SONNET,
            Self::Claude3Haiku20240307 => CloudLlm::CLAUDE_3_HAIKU,
            Self::Claude35SonnetLatest => CloudLlm::CLAUDE_3_5_SONNET,
            Self::Claude37SonnetLatest => CloudLlm::CLAUDE_3_7_SONNET,
            Self::Claude37SonnetThinkingLatest => CloudLlm::CLAUDE_3_7_SONNET_THINKING,
            Self::Claude35HaikuLatest => CloudLlm::CLAUDE_3_5_HAIKU,
        }
    }
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Claude3OpusLatest => "claude-3-opus-latest",
            Self::Claude3Sonnet20240229 => "claude-3-sonnet-20240229",
            Self::Claude3Haiku20240307 => "claude-3-haiku-20240307",
            Self::Claude35SonnetLatest => "claude-3-5-sonnet-latest",
            Self::Claude37SonnetLatest => "claude-3-7-sonnet-latest",
            Self::Claude37SonnetThinkingLatest => "claude-3-7-sonnet-thinking-latest",
            Self::Claude35HaikuLatest => "claude-3-5-haiku-latest",
        }
    }
    pub fn all_provider_models() -> Vec<CloudLlm> {
        vec![
            CloudLlm::CLAUDE_3_OPUS,
            CloudLlm::CLAUDE_3_SONNET,
            CloudLlm::CLAUDE_3_HAIKU,
            CloudLlm::CLAUDE_3_5_SONNET,
            CloudLlm::CLAUDE_3_7_SONNET,
            CloudLlm::CLAUDE_3_7_SONNET_THINKING,
            CloudLlm::CLAUDE_3_5_HAIKU,
        ]
    }
    pub fn provider_friendly_name(&self) -> &'static str {
        "Anthropic"
    }
    pub fn default_model() -> CloudLlm {
        CloudLlm::CLAUDE_3_7_SONNET
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum OpenAiLlmId {
    Gpt4,
    Gpt35Turbo,
    Gpt432k,
    Gpt4Turbo,
    Gpt4o,
    Gpt4oMini,
    O1,
    O1Mini,
    O3Mini,
}
impl OpenAiLlmId {
    pub fn model(&self) -> CloudLlm {
        match self {
            Self::Gpt4 => CloudLlm::GPT_4,
            Self::Gpt35Turbo => CloudLlm::GPT_3_5_TURBO,
            Self::Gpt432k => CloudLlm::GPT_4_32K,
            Self::Gpt4Turbo => CloudLlm::GPT_4_TURBO,
            Self::Gpt4o => CloudLlm::GPT_4O,
            Self::Gpt4oMini => CloudLlm::GPT_4O_MINI,
            Self::O1 => CloudLlm::O1,
            Self::O1Mini => CloudLlm::O1_MINI,
            Self::O3Mini => CloudLlm::O3_MINI,
        }
    }
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Gpt4 => "gpt-4",
            Self::Gpt35Turbo => "gpt-3.5-turbo",
            Self::Gpt432k => "gpt-4-32k",
            Self::Gpt4Turbo => "gpt-4-turbo",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::O1 => "o1",
            Self::O1Mini => "o1-mini",
            Self::O3Mini => "o3-mini",
        }
    }
    pub fn all_provider_models() -> Vec<CloudLlm> {
        vec![
            CloudLlm::GPT_4,
            CloudLlm::GPT_3_5_TURBO,
            CloudLlm::GPT_4_32K,
            CloudLlm::GPT_4_TURBO,
            CloudLlm::GPT_4O,
            CloudLlm::GPT_4O_MINI,
            CloudLlm::O1,
            CloudLlm::O1_MINI,
            CloudLlm::O3_MINI,
        ]
    }
    pub fn provider_friendly_name(&self) -> &'static str {
        "OpenAI"
    }
    pub fn default_model() -> CloudLlm {
        CloudLlm::GPT_4O_MINI
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum PerplexityLlmId {
    SonarReasoningPro,
    SonarReasoning,
    SonarPro,
    Sonar,
}
impl PerplexityLlmId {
    pub fn model(&self) -> CloudLlm {
        match self {
            Self::SonarReasoningPro => CloudLlm::SONAR_REASONING_PRO,
            Self::SonarReasoning => CloudLlm::SONAR_REASONING,
            Self::SonarPro => CloudLlm::SONAR_PRO,
            Self::Sonar => CloudLlm::SONAR,
        }
    }
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::SonarReasoningPro => "sonar-reasoning-pro",
            Self::SonarReasoning => "sonar-reasoning",
            Self::SonarPro => "sonar-pro",
            Self::Sonar => "sonar",
        }
    }
    pub fn all_provider_models() -> Vec<CloudLlm> {
        vec![
            CloudLlm::SONAR_REASONING_PRO,
            CloudLlm::SONAR_REASONING,
            CloudLlm::SONAR_PRO,
            CloudLlm::SONAR,
        ]
    }
    pub fn provider_friendly_name(&self) -> &'static str {
        "Perplexity"
    }
    pub fn default_model() -> CloudLlm {
        CloudLlm::SONAR
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum MistralAiLlmId {
    MistralLargeLatest,
    Ministral3bLatest,
    Ministral8bLatest,
    OpenMistralNemo,
    MistralSmallLatest,
    MistralSabaLatest,
    CodestralLatest,
}
impl MistralAiLlmId {
    pub fn model(&self) -> CloudLlm {
        match self {
            Self::MistralLargeLatest => CloudLlm::MISTRAL_LARGE,
            Self::Ministral3bLatest => CloudLlm::MINISTRAL_3B,
            Self::Ministral8bLatest => CloudLlm::MINISTRAL_8B,
            Self::OpenMistralNemo => CloudLlm::MISTRAL_NEMO,
            Self::MistralSmallLatest => CloudLlm::MISTRAL_SMALL,
            Self::MistralSabaLatest => CloudLlm::MISTRAL_SABA,
            Self::CodestralLatest => CloudLlm::CODESTRAL,
        }
    }
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::MistralLargeLatest => "mistral-large-latest",
            Self::Ministral3bLatest => "ministral-3b-latest",
            Self::Ministral8bLatest => "ministral-8b-latest",
            Self::OpenMistralNemo => "open-mistral-nemo",
            Self::MistralSmallLatest => "mistral-small-latest",
            Self::MistralSabaLatest => "mistral-saba-latest",
            Self::CodestralLatest => "codestral-latest",
        }
    }
    pub fn all_provider_models() -> Vec<CloudLlm> {
        vec![
            CloudLlm::MISTRAL_LARGE,
            CloudLlm::MINISTRAL_3B,
            CloudLlm::MINISTRAL_8B,
            CloudLlm::MISTRAL_NEMO,
            CloudLlm::MISTRAL_SMALL,
            CloudLlm::MISTRAL_SABA,
            CloudLlm::CODESTRAL,
        ]
    }
    pub fn provider_friendly_name(&self) -> &'static str {
        "Mistral AI"
    }
    pub fn default_model() -> CloudLlm {
        CloudLlm::MINISTRAL_8B
    }
}
pub trait AnthropicModelTrait {
    fn model(&mut self) -> &mut CloudLlm;
    fn model_id_str(mut self, model_id: &str) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::model_from_model_id(model_id)?;
        Ok(self)
    }
    fn claude_3_opus_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_OPUS;
        Ok(self)
    }
    fn claude_3_sonnet_20240229(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_SONNET;
        Ok(self)
    }
    fn claude_3_haiku_20240307(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_HAIKU;
        Ok(self)
    }
    fn claude_3_5_sonnet_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_5_SONNET;
        Ok(self)
    }
    fn claude_3_7_sonnet_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_7_SONNET;
        Ok(self)
    }
    fn claude_3_7_sonnet_thinking_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_7_SONNET_THINKING;
        Ok(self)
    }
    fn claude_3_5_haiku_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CLAUDE_3_5_HAIKU;
        Ok(self)
    }
}
pub trait OpenAiModelTrait {
    fn model(&mut self) -> &mut CloudLlm;
    fn model_id_str(mut self, model_id: &str) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::model_from_model_id(model_id)?;
        Ok(self)
    }
    fn gpt_4(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::GPT_4;
        Ok(self)
    }
    fn gpt_3_5_turbo(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::GPT_3_5_TURBO;
        Ok(self)
    }
    fn gpt_4_32k(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::GPT_4_32K;
        Ok(self)
    }
    fn gpt_4_turbo(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::GPT_4_TURBO;
        Ok(self)
    }
    fn gpt_4o(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::GPT_4O;
        Ok(self)
    }
    fn gpt_4o_mini(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::GPT_4O_MINI;
        Ok(self)
    }
    fn o1(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::O1;
        Ok(self)
    }
    fn o1_mini(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::O1_MINI;
        Ok(self)
    }
    fn o3_mini(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::O3_MINI;
        Ok(self)
    }
}
pub trait PerplexityModelTrait {
    fn model(&mut self) -> &mut CloudLlm;
    fn model_id_str(mut self, model_id: &str) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::model_from_model_id(model_id)?;
        Ok(self)
    }
    fn sonar_reasoning_pro(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::SONAR_REASONING_PRO;
        Ok(self)
    }
    fn sonar_reasoning(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::SONAR_REASONING;
        Ok(self)
    }
    fn sonar_pro(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::SONAR_PRO;
        Ok(self)
    }
    fn sonar(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::SONAR;
        Ok(self)
    }
}
pub trait MistralAiModelTrait {
    fn model(&mut self) -> &mut CloudLlm;
    fn model_id_str(mut self, model_id: &str) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::model_from_model_id(model_id)?;
        Ok(self)
    }
    fn mistral_large_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::MISTRAL_LARGE;
        Ok(self)
    }
    fn ministral_3b_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::MINISTRAL_3B;
        Ok(self)
    }
    fn ministral_8b_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::MINISTRAL_8B;
        Ok(self)
    }
    fn open_mistral_nemo(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::MISTRAL_NEMO;
        Ok(self)
    }
    fn mistral_small_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::MISTRAL_SMALL;
        Ok(self)
    }
    fn mistral_saba_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::MISTRAL_SABA;
        Ok(self)
    }
    fn codestral_latest(mut self) -> crate::Result<Self>
    where
        Self: Sized,
    {
        *self.model() = CloudLlm::CODESTRAL;
        Ok(self)
    }
}
