use std::borrow::Cow;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LocalLlmOrganization {
    pub friendly_name: Cow<'static, str>,
    pub hf_account: Cow<'static, str>,
}

impl LocalLlmOrganization {
    pub fn new(friendly_name: &str, hf_account: Option<&str>) -> Self {
        let friendly_name: Cow<'_, str> = friendly_name.to_string().into();
        let hf_account: Cow<'_, str> = if let Some(hf_account) = hf_account {
            hf_account.to_string().into()
        } else {
            "".into()
        };
        Self {
            friendly_name,
            hf_account,
        }
    }
}
