use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{
    LmcppServer,
    client::types::completion::{CompletionRequest, CompletionResponse},
    server::ipc::{ClientError, ServerClientExt},
};

impl LmcppServer {
    pub fn infill<A: InfillRequestProvider>(
        &self,
        request: A,
    ) -> Result<CompletionResponse, ClientError> {
        request.with_request(|req| self.client.post("/infill", req))
    }
}
/// JSON payload for `POST /infill`, llama.cpp’s **fill-in-the-middle (FIM)**
/// endpoint.
///
/// # How the server uses these fields
/// * The server wraps your values with the model’s special tokens:
///   ```text
///   <FIM_PRE>{input_prefix}<FIM_SUF>{input_suffix}<FIM_MID>{prompt}
///   ```
///   and, when the model exposes repo-level tokens, it prepends each
///   `input_extra` file inside a `<FIM_REP>` / `<FIM_FILE_SEP>` block.
/// * Sampling knobs such as `temperature`, `top_k`, `n_predict`, etc. are
///   accepted alongside this struct because the `/infill` route inherits
///   _all_ `/completion` parameters. They are **omitted here** to keep the
///   type focused on fields unique to infill.
///
/// # Output
/// The endpoint streams back the **generated infill** token-by-token.  Each
/// chunk is a small JSON object with a `content` string and a `stop` flag.
/// Concatenate all `content` pieces until `stop == true` to obtain the full
/// insertion text.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
#[builder(on(String, into))]
#[builder(derive(Debug, Clone))]
pub struct InfillRequest {
    /// Everything that appears *before* the cursor.
    /// Usually raw source-code text, but you can also pass a
    /// token-ID string if your frontend encodes the prefix that way.
    /// Required.
    pub input_prefix: String,

    /// Everything that appears *after* the cursor (the code that must
    /// remain untouched after the generated infill).
    /// Required.
    pub input_suffix: String,

    /// Extra context supplied **before** the prefix, typically other
    /// files in the same project.
    /// Every element provides a `filename` and its full `text` so the
    /// server can wrap them in `<FIM_REPO>/<FIM_FILE_SEP>` tokens when
    /// the model supports repo-level FIM.
    /// Optional – omit or set to `None` when no extra files are needed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_extra: Option<Vec<InputExtra>>,

    /// All options available in the `CompletionRequest` struct.
    ///
    /// The prompt text is inserted immediately after the
    /// `<FIM_MID>` marker (i.e. after the gap to be filled).
    /// Use this to nudge style or give instructions; leave `None` for
    /// a plain “fill the gap” request.
    /// Optional.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionRequest>,
}

/// Helper object for `input_extra`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InputExtra {
    /// Path or logical name of the extra file.
    pub filename: String,
    /// Full text contents of that file.
    pub text: String,
}

/// Anything that can present a `&InfillRequest` for the duration
/// of one synchronous operation.
pub trait InfillRequestProvider {
    /// Run `f` with a reference to the (possibly freshly-built) request.
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R;
}

// 1. Owned InfillRequest
impl InfillRequestProvider for InfillRequest {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R,
    {
        // We have the owned request, so just call the closure with a reference to it.
        f(&self)
        // `self` (the InfillRequest) will be dropped after f returns, which is fine
        // because f cannot persist the reference beyond its scope.
    }
}

// 2. Immutable reference to InfillRequest
impl<'a> InfillRequestProvider for &'a InfillRequest {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R,
    {
        // We already have a reference, just pass it along.
        f(self)
    }
}

// 3. Mutable reference to InfillRequest
impl<'a> InfillRequestProvider for &'a mut InfillRequest {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R,
    {
        // Reborrow the mutable reference as an immutable reference for the call.
        f(&*self)
        // This does not clone the request; it simply allows an immutable borrow
        // while we hold a mutable reference.
    }
}

// 4. Owned InfillRequestBuilder (with complete state)
impl<S> InfillRequestProvider for InfillRequestBuilder<S>
where
    S: infill_request_builder::IsComplete, // Only implement for complete builder state
{
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R,
    {
        // We can consume the builder and build the request without cloning.
        let req: InfillRequest = self.build();
        f(&req)
        // `req` is a new InfillRequest built from the builder.
    }
}

// 5. Immutable reference to InfillRequestBuilder (requires Clone)
impl<'a, S> InfillRequestProvider for &'a InfillRequestBuilder<S>
where
    // ONLY the “is complete” bound is needed
    S: infill_request_builder::IsComplete,
{
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R,
    {
        // the builder itself is Clone; S doesn’t have to be
        let req = self.clone().build();
        f(&req)
    }
}

// 6. Mutable reference to InfillRequestBuilder (requires Clone)
impl<'a, S> InfillRequestProvider for &'a mut InfillRequestBuilder<S>
where
    S: infill_request_builder::IsComplete,
{
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&InfillRequest) -> R,
    {
        let req = self.clone().build();
        f(&req)
    }
}

#[cfg(test)]
mod tests {

    use serial_test::serial;

    use super::*;
    use crate::{
        LmcppServer,
        error::LmcppResult,
        server::{
            builder::LmcppServerLauncher, toolchain::builder::LmcppToolChain,
            types::start_args::ServerArgs,
        },
    };

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_infill() -> LmcppResult<()> {
        // We need an infill model to run this test.
        let client = LmcppServerLauncher::builder()
            .toolchain(LmcppToolChain::builder().install_only().build()?)
            .server_args(
                ServerArgs::builder()
                    .hf_repo("bartowski/codegemma-2b-GGUF")?
                    .build(),
            )
            .load()?;

        let response = client.infill(
            InfillRequest::builder()
                .input_prefix("Hello, ")
                .input_suffix(" world!")
                .completion(CompletionRequest::builder().prompt("Hello, world!").build())
                .build(),
        )?;
        println!("Infill response: {:#?}", response);
        Ok(())
    }

    #[test]
    #[ignore]
    #[allow(unused_mut)]
    fn test_lmcpp_server_infill_variants() -> LmcppResult<()> {
        let client = LmcppServer::dummy();

        // ─────────────────────────  InfillRequest ───────────────────

        // ─────────────────────────  Owned - Immutable ───────────────────
        let req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build())
            .build();
        let _ = client.infill(req_owned);

        // ─────────────────────────  Owned - Mutable ────────────────────

        let mut req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build())
            .build();
        let _ = client.infill(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build())
            .build();
        let _ = client.infill(&req_owned);

        // ─────────────────────────  Ref - Mutable ────────────────────
        let mut req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build())
            .build();
        let _ = client.infill(&mut req_owned);

        // ─────────────────────────  InfillRequestBuilder ───────────────────

        // ─────────────────────────  Owned - Immutable ───────────────────
        let req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build());
        let _ = client.infill(req_owned);

        // ─────────────────────────  Owned - Mutable ────────────────────
        let mut req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build());
        let _ = client.infill(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build());
        let _ = client.infill(&req_owned);

        // ─────────────────────────  Ref - Mutable ────────────────────
        let mut req_owned = InfillRequest::builder()
            .input_prefix("hi")
            .input_suffix("ho")
            .completion(CompletionRequest::builder().prompt("Test request").build());
        let _ = client.infill(&mut req_owned);

        Ok(())
    }
}
