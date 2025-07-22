use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{
    LmcppServer, client::types::generation_settings::TokenIds, error::LmcppResult,
    server::ipc::ServerClientExt,
};

impl LmcppServer {
    pub fn detokenize<A: DetokenizeRequestProvider>(
        &self,
        request: A,
    ) -> LmcppResult<DetokenizeResponse> {
        request.with_request(|req| self.client.post("/detokenize", req).map_err(Into::into))
    }
}

/// Request payload for the `/detokenize` endpoint.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
#[builder(derive(Debug, Clone))]
pub struct DetokenizeRequest {
    /// The sequence of token IDs to convert back into text (required).
    ///
    /// These should be valid token IDs as produced by the model’s tokenizer. They will be concatenated and decoded into a string.
    #[builder(into)]
    pub tokens: TokenIds,
}

/// Anything that can present a `&DetokenizeRequest` for one synchronous call.
pub trait DetokenizeRequestProvider {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R;
}

/* ─────────── plain DetokenizeRequest references ─────────── */

impl<'a> DetokenizeRequestProvider for &'a DetokenizeRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R,
    {
        f(self)
    }
}

impl<'a> DetokenizeRequestProvider for &'a mut DetokenizeRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R,
    {
        f(self)
    }
}

/* ─────────── owned DetokenizeRequest ─────────── */

impl DetokenizeRequestProvider for DetokenizeRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R,
    {
        f(&self)
    }
}

/* ─────────── DetokenizeRequestBuilder variants ─────────── */

// owned builder (no clone, zero-copy)
impl<S> DetokenizeRequestProvider for DetokenizeRequestBuilder<S>
where
    S: detokenize_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R,
    {
        let req = self.build(); // safe: S guarantees completeness
        f(&req)
    }
}

// immutable ref (clone once, then build)
impl<'a, S> DetokenizeRequestProvider for &'a DetokenizeRequestBuilder<S>
where
    S: detokenize_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R,
    {
        let req = self.clone().build();
        f(&req)
    }
}

// mutable ref (same as above)
impl<'a, S> DetokenizeRequestProvider for &'a mut DetokenizeRequestBuilder<S>
where
    S: detokenize_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&DetokenizeRequest) -> R,
    {
        let req = self.clone().build();
        f(&req)
    }
}

/// Response payload for the `/detokenize` endpoint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetokenizeResponse {
    /// The resulting text after detokenizing the provided tokens.
    ///
    /// This string is the decoded text corresponding to the sequence of input token IDs. If the token list was empty, this may be an empty string.
    pub content: String,
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{LmcppServer, error::LmcppResult, server::builder::LmcppServerLauncher};

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_detokenize() -> LmcppResult<()> {
        let client = LmcppServerLauncher::default().load()?;

        let detok = client.detokenize(DetokenizeRequest::builder().tokens(vec![0u64, 1, 2, 3]))?;
        println!("Detokenize response: {:?}", detok);
        Ok(())
    }

    #[test]
    #[ignore]
    #[allow(unused_mut)]
    fn test_lmcpp_server_detokenize_variants() -> LmcppResult<()> {
        let client = LmcppServer::dummy();

        // ─────────────────────────  DetokenizeRequest ──────────────────

        // ─────────────────────────  Owned - Immutable ──────────────────
        let req_owned = DetokenizeRequest::builder()
            .tokens(vec![0u64, 1, 2])
            .build();
        let _ = client.detokenize(req_owned);

        // ─────────────────────────  Owned - Mutable ───────────────────
        let mut req_owned = DetokenizeRequest::builder()
            .tokens(vec![0u64, 1, 2])
            .build();
        let _ = client.detokenize(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = DetokenizeRequest::builder()
            .tokens(vec![0u64, 1, 2])
            .build();
        let _ = client.detokenize(&req_owned);

        // ─────────────────────────  Ref - Mutable ─────────────────────
        let mut req_owned = DetokenizeRequest::builder()
            .tokens(vec![0u64, 1, 2])
            .build();
        let _ = client.detokenize(&mut req_owned);

        // ───────────────────  DetokenizeRequestBuilder ────────────────

        // ─────────────────────────  Owned - Immutable ─────────────────
        let req_owned = DetokenizeRequest::builder().tokens(vec![0u64, 1, 2]);
        let _ = client.detokenize(req_owned);

        // ─────────────────────────  Owned - Mutable ───────────────────
        let mut req_owned = DetokenizeRequest::builder().tokens(vec![0u64, 1, 2]);
        let _ = client.detokenize(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = DetokenizeRequest::builder().tokens(vec![0u64, 1, 2]);
        let _ = client.detokenize(&req_owned);

        // ─────────────────────────  Ref - Mutable ─────────────────────
        let mut req_owned = DetokenizeRequest::builder().tokens(vec![0u64, 1, 2]);
        let _ = client.detokenize(&mut req_owned);

        Ok(())
    }
}
