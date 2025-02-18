#[derive(Debug, Clone, PartialEq)]
pub struct LocalLlmOrganization {
    pub friendly_name: &'static str,
    pub hf_account: &'static str,
}
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
        friendly_name: "Meta",
        hf_account: "meta-llama",
    };
    pub const MISTRAL: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "Mistral",
        hf_account: "mistralai",
    };
    pub const STABILITY_AI: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "Stability AI",
        hf_account: "stabilityai",
    };
    pub const ALIBABA: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "Alibaba",
        hf_account: "qwen",
    };
    pub const IBM: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "IBM",
        hf_account: "ibm-granite",
    };
    pub const ARCEE_AI: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "Arcee AI",
        hf_account: "arcee-ai",
    };
    pub const NVIDIA: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "Nvidia",
        hf_account: "nvidia",
    };
    pub const MICROSOFT: LocalLlmOrganization = LocalLlmOrganization {
        friendly_name: "Microsoft",
        hf_account: "microsoft",
    };
}
