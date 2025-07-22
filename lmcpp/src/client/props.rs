use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    LmcppServer, client::types::generation_settings::GenerationSettings, error::LmcppResult,
    server::ipc::ServerClientExt,
};

impl LmcppServer {
    pub fn props(&self) -> LmcppResult<PropsResponse> {
        self.client.get("/props").map_err(Into::into)
    }
}

/// Properties returned by the llama.cpp `/props` endpoint.
///
/// *All fields are optional* so that clients never fail to deserialize
/// when running against older or newer server builds that add / remove fields.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct PropsResponse {
    // ── Model / prompt metadata ──────────────────────────────────────────────
    /// Beginning-of-stream (BOS) token extracted from GGUF metadata.
    /// Useful when constructing raw prompts manually.  
    /// Absent on very old model files that don’t declare a BOS token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bos_token: Option<String>,

    /// Assistant role label injected by a global system prompt (e.g. `"Assistant:"`).
    /// Empty / None if the server was launched without a system-prompt file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assistant_name: Option<String>,

    /// User role label (anti-prompt) used for stopping generation (e.g. `"User:"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>,

    /// Raw chat-prompt template string (Jinja-2 style) shipped with the model,
    /// or provided via `--chat-template`.  
    /// Empty / None for models that are *not* chat-tuned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    // ── Server / build information ───────────────────────────────────────────
    /// Build identifier of the running llama.cpp binary
    /// (usually `b<build-nr>-<git-sha>` or `"unknown"` for unofficial builds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_info: Option<String>,

    /// Total parallel generation slots compiled into the server
    /// (`--parallel`, defaults to 1).  `None` on very old builds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_slots: Option<u32>,

    // ── Model location & modality info ───────────────────────────────────────
    /// Absolute filesystem path to the loaded GGUF / GGML model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,

    /// Flags indicating which input modalities (beyond plain text) the model supports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Modalities>,

    // ── Default generation parameters ───────────────────────────────────────
    /// Struct holding the server’s **default** generation settings—
    /// exactly the object you’d find echoed under `generation_settings`
    /// in a `/completion` response when *no overrides* are supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_generation_settings: Option<GenerationSettings>,
}

/// Sub-object of `/props` enumerating non-text capabilities.
///
/// All flags are optional so that new modalities can be added
/// without breaking older clients.
#[derive(Debug, Serialize, Deserialize)]
pub struct Modalities {
    /// `true` if the loaded model accepts image embeddings (e.g. LLaVA variants).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,

    /// `true` if the model supports audio tokens (as of multimodal audio models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<bool>,
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::server::builder::LmcppServerLauncher;

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_props() -> LmcppResult<()> {
        let client = LmcppServerLauncher::default().load()?;

        let props = client.props()?;
        println!("Props response: {:#?}", props);
        Ok(())
    }
}
