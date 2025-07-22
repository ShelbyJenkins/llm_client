use crate::{
    LmcppServer,
    client::types::completion::{
        CompletionRequest, CompletionRequestBuilder, CompletionResponse, completion_request_builder,
    },
    error::LmcppResult,
    server::ipc::ServerClientExt,
};

impl LmcppServer {
    pub fn completion<A: CompletionRequestProvider>(
        &self,
        request: A,
    ) -> LmcppResult<CompletionResponse> {
        request.with_request(|req| self.client.post("/completion", req).map_err(Into::into))
    }
}

/// Anything that can present a `&CompletionRequest` for the duration
/// of one synchronous operation.
pub trait CompletionRequestProvider {
    /// Run `f` with a reference to the (possibly freshly-built) request.
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R;
}

// 1. Owned CompletionRequest
impl CompletionRequestProvider for CompletionRequest {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R,
    {
        // We have the owned request, so just call the closure with a reference to it.
        f(&self)
        // `self` (the CompletionRequest) will be dropped after f returns, which is fine
        // because f cannot persist the reference beyond its scope.
    }
}

// 2. Immutable reference to CompletionRequest
impl<'a> CompletionRequestProvider for &'a CompletionRequest {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R,
    {
        // We already have a reference, just pass it along.
        f(self)
    }
}

// 3. Mutable reference to CompletionRequest
impl<'a> CompletionRequestProvider for &'a mut CompletionRequest {
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R,
    {
        // Reborrow the mutable reference as an immutable reference for the call.
        f(&*self)
        // This does not clone the request; it simply allows an immutable borrow
        // while we hold a mutable reference.
    }
}

// 4. Owned CompletionRequestBuilder (with complete state)
impl<S> CompletionRequestProvider for CompletionRequestBuilder<S>
where
    S: completion_request_builder::IsComplete, // Only implement for complete builder state
{
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R,
    {
        // We can consume the builder and build the request without cloning.
        let req: CompletionRequest = self.build();
        f(&req)
        // `req` is a new CompletionRequest built from the builder.
    }
}

// 5. Immutable reference to CompletionRequestBuilder (requires Clone)
impl<'a, S> CompletionRequestProvider for &'a CompletionRequestBuilder<S>
where
    // ONLY the “is complete” bound is needed
    S: completion_request_builder::IsComplete,
{
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R,
    {
        // the builder itself is Clone; S doesn’t have to be
        let req = self.clone().build();
        f(&req)
    }
}

// 6. Mutable reference to CompletionRequestBuilder (requires Clone)
impl<'a, S> CompletionRequestProvider for &'a mut CompletionRequestBuilder<S>
where
    S: completion_request_builder::IsComplete,
{
    fn with_request<F, R>(self, f: F) -> R
    where
        F: FnOnce(&CompletionRequest) -> R,
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
        LmcppServer, client::types::generation_settings::SamplingParams,
        server::builder::LmcppServerLauncher,
    };

    #[test]
    #[ignore]
    #[serial]
    fn test_lmcpp_server_completion() -> LmcppResult<()> {
        let client = LmcppServerLauncher::default().load()?;

        let response = client.completion(
            CompletionRequest::builder()
                .prompt("Hello, world!")
                .n_predict(100),
        )?;
        println!("Completion response: {:#?}", response);
        Ok(())
    }

    #[test]
    #[ignore]
    #[allow(unused_mut)]
    fn test_lmcpp_server_completion_variants() -> LmcppResult<()> {
        let client = LmcppServer::dummy();

        // ─────────────────────────  CompletionRequest ───────────────────

        // ─────────────────────────  Owned - Immutable ───────────────────
        let req_owned = CompletionRequest::builder()
            .prompt("Test request")
            .sampling(SamplingParams::builder().temperature(0.7).build())
            .build();
        let _ = client.completion(req_owned);

        // ─────────────────────────  Owned - Mutable ────────────────────

        let mut req_owned = CompletionRequest::builder().prompt("Test request").build();
        let _ = client.completion(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = CompletionRequest::builder().prompt("Test request").build();
        let _ = client.completion(&req_owned);

        // ─────────────────────────  Ref - Mutable ────────────────────
        let mut req_owned = CompletionRequest::builder().prompt("Test request").build();
        let _ = client.completion(&mut req_owned);

        // ─────────────────────────  CompletionRequestBuilder ───────────────────

        // ─────────────────────────  Owned - Immutable ───────────────────
        let req_owned = CompletionRequest::builder().prompt("Test request");
        let _ = client.completion(req_owned);

        // ─────────────────────────  Owned - Mutable ────────────────────
        let mut req_owned = CompletionRequest::builder().prompt("Test request");
        let _ = client.completion(req_owned);

        // ─────────────────────────  Ref - Immutable ───────────────────
        let req_owned = CompletionRequest::builder().prompt("Test request");
        let _ = client.completion(&req_owned);

        // ─────────────────────────  Ref - Mutable ────────────────────
        let mut req_owned = CompletionRequest::builder().prompt("Test request");
        let _ = client.completion(&mut req_owned);

        Ok(())
    }
}
