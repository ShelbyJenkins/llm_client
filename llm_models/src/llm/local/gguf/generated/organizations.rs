use super::*;
impl LocalLlmOrganization {
    pub fn all_organizations() -> Vec<Self> {
        vec![
            Self::META,
            Self::MISTRAL,
            Self::STABILITY_AI,
            Self::ALIBABA,
            Self::IBM,
            Self::ARCEE_AI,
            Self::NVIDIA,
            Self::MICROSOFT,
        ]
    }
    pub const META: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Meta"),
        hf_account: Cow::Borrowed("meta-llama"),
    };
    pub const MISTRAL: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Mistral"),
        hf_account: Cow::Borrowed("mistralai"),
    };
    pub const STABILITY_AI: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Stability AI"),
        hf_account: Cow::Borrowed("stabilityai"),
    };
    pub const ALIBABA: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Alibaba"),
        hf_account: Cow::Borrowed("qwen"),
    };
    pub const IBM: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("IBM"),
        hf_account: Cow::Borrowed("ibm-granite"),
    };
    pub const ARCEE_AI: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Arcee AI"),
        hf_account: Cow::Borrowed("arcee-ai"),
    };
    pub const NVIDIA: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Nvidia"),
        hf_account: Cow::Borrowed("nvidia"),
    };
    pub const MICROSOFT: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: Cow::Borrowed("Microsoft"),
        hf_account: Cow::Borrowed("microsoft"),
    };
}
