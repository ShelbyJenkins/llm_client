use std::ops::Not;

use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{LmcppServer, error::LmcppResult, server::ipc::ServerClientExt};

impl LmcppServer {
    pub fn tokenize<A: TokenizeRequestProvider>(
        &self,
        request: A,
    ) -> LmcppResult<TokenizeResponse> {
        request.with_request(|req| self.client.post("/tokenize", req).map_err(Into::into))
    }
}

/// Request payload for the `/tokenize` endpoint.
#[derive(Default, Serialize, Deserialize, Debug, Clone, Builder)]
#[builder(derive(Debug, Clone))]
pub struct TokenizeRequest {
    /// The text content to tokenize (required).
    ///
    /// This is the input string that will be converted into tokens by the model's tokenizer.
    /// If not provided or empty, no tokens will be produced.
    #[builder(into)]
    pub content: String,

    /// Whether to insert special tokens (e.g. the BOS token) during tokenization.
    ///
    /// If set to `true`, special tokens (like the beginning-of-stream token) will be included in the token sequence.
    /// Default: `false` (no special tokens are added unless the model’s tokenizer inherently does so).
    #[serde(skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub add_special: bool,

    /// Whether to include token text pieces along with token IDs in the output.
    ///
    /// If `true`, the tokenization result will include each token’s text (or byte sequence) in addition to its ID.
    /// If `false`, the result will only be a list of token ID numbers. Default: `false`.
    #[serde(skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub with_pieces: bool,
}

/// Anything that can present a `&TokenizeRequest` for one synchronous call.
pub trait TokenizeRequestProvider {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R;
}

/* ─────────── plain TokenizeRequest references ─────────── */

impl<'a> TokenizeRequestProvider for &'a TokenizeRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R,
    {
        f(self)
    }
}

impl<'a> TokenizeRequestProvider for &'a mut TokenizeRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R,
    {
        f(self)
    }
}

/* ─────────── owned TokenizeRequest ─────────── */

impl TokenizeRequestProvider for TokenizeRequest {
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R,
    {
        f(&self)
    }
}

/* ─────────── TokenizeRequestBuilder variants ─────────── */

// owned builder (no clone, zero-copy)
impl<S> TokenizeRequestProvider for TokenizeRequestBuilder<S>
where
    S: tokenize_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R,
    {
        let req = self.build(); // safe: S guarantees completeness
        f(&req)
    }
}

// immutable ref (clone once, then build)
impl<'a, S> TokenizeRequestProvider for &'a TokenizeRequestBuilder<S>
where
    S: tokenize_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R,
    {
        let req = self.clone().build();
        f(&req)
    }
}

// mutable ref (same as above)
impl<'a, S> TokenizeRequestProvider for &'a mut TokenizeRequestBuilder<S>
where
    S: tokenize_request_builder::IsComplete,
{
    #[inline]
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&TokenizeRequest) -> R,
    {
        let req = self.clone().build();
        f(&req)
    }
}

/// Response payload for the `/tokenize` endpoint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenizeResponse {
    /// Tokenization result as an array of tokens.
    ///
    /// If `with_pieces` was false in the request, this will be a list of token IDs (integers).  
    /// If `with_pieces` was true, this will be a list of objects each containing an `id` and a `piece` for the token.
    /// (The piece is a string if the token text is valid Unicode, or a list of byte values if not.)
    pub tokens: Vec<TokenOutput>,
}

/// Enum for a token in the `/tokenize` response, which may be just an ID or an object with id and piece.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum TokenOutput {
    /// Variant for a token ID when `with_pieces` is false.
    Id(u32),
    /// Variant for a token with its piece of text/bytes when `with_pieces` is true.
    IdWithPiece {
        /// The token ID.
        id: u32,
        /// The token's text piece (or byte sequence) content.
        piece: TokenPiece,
    },
}

/// Enum representing a token's textual content, used when returning token pieces.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum TokenPiece {
    /// A Unicode text segment for the token.
    Text(String),
    /// Raw bytes for tokens that aren't valid Unicode (each number represents a byte).
    Bytes(Vec<u8>),
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{LmcppServer, server::builder::LmcppServerLauncher};

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_tokenize() -> LmcppResult<()> {
        let client = LmcppServerLauncher::default().load()?;

        let tokens = client.tokenize(TokenizeRequest::builder().content("Hello world!"))?;
        println!("Tokenize response: {:?}", tokens);
        Ok(())
    }

    #[test]
    #[ignore]
    #[allow(unused_mut)]
    fn test_lmcpp_server_tokenize_variants() -> LmcppResult<()> {
        let client = LmcppServer::dummy();

        // ─────────────────────────  TokenizeRequest ──────────────────

        // ─────────────────────────  Owned - Immutable ──────────────────
        let req_owned = TokenizeRequest::builder().content("hi mom!").build();
        let _ = client.tokenize(req_owned);

        // ─────────────────────────  Owned - Mutable ───────────────────
        let mut req_owned = TokenizeRequest::builder().content("hi mom!").build();
        let _ = client.tokenize(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = TokenizeRequest::builder().content("hi mom!").build();
        let _ = client.tokenize(&req_owned);

        // ─────────────────────────  Ref - Mutable ─────────────────────
        let mut req_owned = TokenizeRequest::builder().content("hi mom!").build();
        let _ = client.tokenize(&mut req_owned);

        // ───────────────────  TokenizeRequestBuilder ────────────────

        // ─────────────────────────  Owned - Immutable ─────────────────
        let req_owned = TokenizeRequest::builder().content("hi mom!");
        let _ = client.tokenize(req_owned);

        // ─────────────────────────  Owned - Mutable ───────────────────
        let mut req_owned = TokenizeRequest::builder().content("hi mom!");
        let _ = client.tokenize(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = TokenizeRequest::builder().content("hi mom!");
        let _ = client.tokenize(&req_owned);

        // ─────────────────────────  Ref - Mutable ─────────────────────
        let mut req_owned = TokenizeRequest::builder().content("hi mom!");
        let _ = client.tokenize(&mut req_owned);

        Ok(())
    }
}
