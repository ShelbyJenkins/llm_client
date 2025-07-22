use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{
    LmcppServer,
    client::types::generation_settings::{ImageData, Pooling},
    error::{LmcppError, LmcppResult},
    server::ipc::{ClientError, ServerClientExt},
};

impl LmcppServer {
    /// Single-input embedding call
    pub fn embedding<A: EmbeddingsRequestProvider>(
        &self,
        request: A,
    ) -> LmcppResult<EmbeddingResponse> {
        request.with_request(|req| {
            // `self.post` → Result<EmbeddingsResponse, ClientError>
            let embds: EmbeddingsResponse = self.client.post("/embedding", req)?;

            let [item]: [EmbeddingResponse; 1] =
                embds.try_into().map_err(|v: Vec<_>| ClientError::Remote {
                    code: 500,
                    message: format!(
                        "Expected exactly one embedding in response, got {}",
                        v.len()
                    ),
                })?;

            Ok(item)
        })
    }

    /// Batch embedding call
    pub fn embeddings<A: EmbeddingsRequestProvider>(
        &self,
        request: A,
    ) -> LmcppResult<EmbeddingsResponse> {
        request.with_request(|req| {
            let embds: EmbeddingsResponse = self.client.post("/embeddings", req)?;
            if embds.is_empty() {
                return Err(LmcppError::Internal(
                    "Expected at least one embedding in response, got zero".into(),
                ));
            }
            Ok(embds)
        })
    }
}

/// Request body for **`POST /embedding`** (single) *and*
/// **`POST /embeddings`** (batch).#[derive(Debug, Serialize, Deserialize, Builder)]
#[derive(Builder, Debug, Clone, Serialize, Deserialize)]
#[builder(derive(Debug, Clone), finish_fn(vis = "", name = build_internal))]
pub struct EmbeddingsRequest {
    /// A single string *or* an array of strings to embed.
    #[builder(into)]
    pub input: EmbeddingInput,

    /// Optional images for multimodal embedding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Vec<ImageData>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pooling: Option<Pooling>,
}

impl<S: embeddings_request_builder::IsComplete> EmbeddingsRequestBuilder<S> {
    pub fn build(self) -> LmcppResult<EmbeddingsRequest> {
        // Delegate to `build_internal()` to get the instance of user.
        let req = self.build_internal();

        match req.input {
            EmbeddingInput::Single(ref s) if s.is_empty() => {
                return Err(LmcppError::InvalidConfig {
                    field: "input: EmbeddingInput::Single",
                    reason: "`input` must contain a non-empty string or a non-empty array of non-empty strings".into(),
                });
            }
            EmbeddingInput::Batch(ref v) => {
                if v.is_empty() || v.iter().any(|s| s.is_empty()) {
                    return Err(LmcppError::InvalidConfig {
                    field: "input: EmbeddingInput::Batch",
                    reason: "`input` must contain a non-empty string or a non-empty array of non-empty strings".into(),
                });
                }
            }
            _ => {}
        }

        Ok(req)
    }
}

/// A single string *or* a list of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

// --- single conversions ---
impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        Self::Single(s)
    }
}
impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        Self::Single(s.into())
    }
}
impl From<&String> for EmbeddingInput {
    fn from(s: &String) -> Self {
        Self::Single(s.clone())
    }
}

// --- batch conversions ---
impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        Self::Batch(v)
    }
}
impl From<Vec<&str>> for EmbeddingInput {
    fn from(v: Vec<&str>) -> Self {
        Self::Batch(v.into_iter().map(str::to_owned).collect())
    }
}
impl From<&[&str]> for EmbeddingInput {
    fn from(v: &[&str]) -> Self {
        Self::Batch(v.iter().map(|s| (*s).to_owned()).collect())
    }
}
impl From<&[String]> for EmbeddingInput {
    fn from(v: &[String]) -> Self {
        Self::Batch(v.to_vec())
    }
}

/// Anything that can present a `&EmbeddingsRequest` for one synchronous call.
pub trait EmbeddingsRequestProvider {
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>;
}

/* ─────────── plain EmbeddingsRequest references ─────────── */

impl<'a> EmbeddingsRequestProvider for &'a EmbeddingsRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>,
    {
        f(self) // already a `Result`
    }
}

impl<'a> EmbeddingsRequestProvider for &'a mut EmbeddingsRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>,
    {
        f(self)
    }
}

/* ─────────── owned EmbeddingsRequest ─────────── */

impl EmbeddingsRequestProvider for EmbeddingsRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>,
    {
        f(&self)
    }
}

/* ─────────── EmbeddingsRequestBuilder variants ─────────── */

impl<S> EmbeddingsRequestProvider for EmbeddingsRequestBuilder<S>
where
    S: embeddings_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>,
    {
        let req = self.build()?; // ← may return a `crate::Error`
        f(&req)
    }
}

impl<'a, S> EmbeddingsRequestProvider for &'a EmbeddingsRequestBuilder<S>
where
    S: embeddings_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>,
    {
        let req = self.clone().build()?;
        f(&req)
    }
}

impl<'a, S> EmbeddingsRequestProvider for &'a mut EmbeddingsRequestBuilder<S>
where
    S: embeddings_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> LmcppResult<R>
    where
        F: FnOnce(&EmbeddingsRequest) -> LmcppResult<R>,
    {
        let req = self.clone().build()?;
        f(&req)
    }
}

/// Response body for **`POST /embeddings`**.
pub type EmbeddingsResponse = Vec<EmbeddingResponse>;

/// Single embedding item.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Position in the batch (always `0` for `/embedding`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,

    /// Embedding vectors.
    ///
    /// *If pooling ≠ `none`*: a **single** vector of length `dim`  
    /// *If pooling = `none`*: one vector **per input token**  
    /// (shape `[n_tokens][dim]`), matching llama.cpp output.
    pub embedding: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        LmcppServer,
        server::{builder::LmcppServerLauncher, types::start_args::ServerArgs},
    };

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_embedding() -> LmcppResult<()> {
        let client = LmcppServerLauncher::builder()
            .server_args(
                ServerArgs::builder()
                    .embeddings_only(true)
                    .default_model()?
                    .build(),
            )
            .load()?;

        let request = EmbeddingsRequest::builder()
            .input("LLMs are awesome.")
            .build()?;
        let embedding = client.embedding(&request)?;
        println!("Embedding index: {:?}", embedding.index);
        Ok(())
    }

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_embeddings() -> LmcppResult<()> {
        let client = LmcppServerLauncher::builder()
            .server_args(
                ServerArgs::builder()
                    .embeddings_only(true)
                    .default_model()?
                    .build(),
            )
            .load()?;

        let request = EmbeddingsRequest::builder()
            .input(vec!["foo", "bar", "baz"])
            .build()?;
        let embeddings = client.embeddings(&request)?;
        println!("Embeddings count: {:?}", embeddings.len());
        Ok(())
    }

    #[test]
    #[ignore]
    #[allow(unused_mut)]
    fn test_lmcpp_server_embeddings_variants() -> LmcppResult<()> {
        let client = LmcppServer::dummy();

        // ─────────────────────────  EmbeddingsRequest ──────────────────

        // ─────────────────────────  Owned - Immutable ──────────────────
        let req_owned = EmbeddingsRequest::builder()
            .input(vec!["foo", "bar", "baz"])
            .build()?;
        let _ = client.embeddings(req_owned);

        // ─────────────────────────  Owned - Mutable ───────────────────
        let mut req_owned = EmbeddingsRequest::builder()
            .input(vec!["foo", "bar", "baz"])
            .build()?;
        let _ = client.embeddings(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = EmbeddingsRequest::builder()
            .input(vec!["foo", "bar", "baz"])
            .build()?;
        let _ = client.embeddings(&req_owned);

        // ─────────────────────────  Ref - Mutable ─────────────────────
        let mut req_owned = EmbeddingsRequest::builder()
            .input(vec!["foo", "bar", "baz"])
            .build()?;
        let _ = client.embeddings(&mut req_owned);

        // ───────────────────  EmbeddingsRequestBuilder ────────────────

        // ─────────────────────────  Owned - Immutable ─────────────────
        let req_owned = EmbeddingsRequest::builder().input(vec!["foo", "bar", "baz"]);
        let _ = client.embeddings(req_owned);

        // ─────────────────────────  Owned - Mutable ───────────────────
        let mut req_owned = EmbeddingsRequest::builder().input(vec!["foo", "bar", "baz"]);
        let _ = client.embeddings(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = EmbeddingsRequest::builder().input(vec!["foo", "bar", "baz"]);
        let _ = client.embeddings(&req_owned);

        // ─────────────────────────  Ref - Mutable ─────────────────────
        let mut req_owned = EmbeddingsRequest::builder().input(vec!["foo", "bar", "baz"]);
        let _ = client.embeddings(&mut req_owned);

        Ok(())
    }
}
