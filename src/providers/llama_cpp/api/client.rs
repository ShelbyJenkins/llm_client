use bytes::Bytes;

use serde::{de::DeserializeOwned, Serialize};

use super::config::{Config, LlamaConfig};
use super::error::{map_deserialization_error, LlamaApiError, WrappedError};
use super::Completions;
use super::Detokenize;
use super::Embedding;
use super::Tokenize;

#[derive(Debug, Clone)]
/// Client is a container for config, backoff and http_client
/// used to make API calls.
pub struct Client<C: Config> {
    http_client: reqwest::Client,
    config: C,
    backoff: backoff::ExponentialBackoff,
}

impl Default for Client<LlamaConfig> {
    fn default() -> Self {
        Self::new()
    }
}

impl Client<LlamaConfig> {
    /// Client with default [LlamaConfig]

    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            config: LlamaConfig::default(),
            backoff: Default::default(),
        }
    }
}

impl<C: Config> Client<C> {
    /// Create client with [LlamaConfig] or [crate::config::AzureConfig]
    pub fn with_config(config: C) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            config,
            backoff: Default::default(),
        }
    }

    /// Provide your own [client] to make HTTP requests with.
    ///
    /// [client]: reqwest::Client
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Exponential backoff for retrying [rate limited](https://platform.openai.com/docs/guides/rate-limits) requests.
    pub fn with_backoff(mut self, backoff: backoff::ExponentialBackoff) -> Self {
        self.backoff = backoff;
        self
    }

    // API groups

    /// To call [Completions] group related APIs using this client.
    pub fn completions(&self) -> Completions<C> {
        Completions::new(self)
    }

    /// To call [Tokenize] group related APIs using this client.
    pub fn tokenize(&self) -> Tokenize<C> {
        Tokenize::new(self)
    }

    /// To call [Detokenize] group related APIs using this client.
    pub fn detokenize(&self) -> Detokenize<C> {
        Detokenize::new(self)
    }
    /// To call [Embedding] group related APIs using this client.
    pub fn embedding(&self) -> Embedding<C> {
        Embedding::new(self)
    }

    pub fn config(&self) -> &C {
        &self.config
    }

    /// Make a POST request to {path} and deserialize the response body
    pub(crate) async fn post<I, O>(&self, path: &str, request: I) -> Result<O, LlamaApiError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .post(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .json(&request)
                .build()?)
        };

        self.execute(request_maker).await
    }

    /// Execute a HTTP request and retry on rate limit
    ///
    /// request_maker serves one purpose: to be able to create request again
    /// to retry API call after getting rate limited. request_maker is async because
    /// reqwest::multipart::Form is created by async calls to read files for uploads.
    async fn execute_raw<M, Fut>(&self, request_maker: M) -> Result<Bytes, LlamaApiError>
    where
        M: Fn() -> Fut,
        Fut: core::future::Future<Output = Result<reqwest::Request, LlamaApiError>>,
    {
        let client = self.http_client.clone();

        backoff::future::retry(self.backoff.clone(), || async {
            let request = request_maker().await.map_err(backoff::Error::Permanent)?;
            let response = client
                .execute(request)
                .await
                .map_err(LlamaApiError::Reqwest)
                .map_err(backoff::Error::Permanent)?;

            let status = response.status();
            let bytes = response
                .bytes()
                .await
                .map_err(LlamaApiError::Reqwest)
                .map_err(backoff::Error::Permanent)?;

            // Deserialize response body from either error object or actual response object
            if !status.is_success() {
                let wrapped_error: WrappedError = serde_json::from_slice(bytes.as_ref())
                    .map_err(|e| map_deserialization_error(e, bytes.as_ref()))
                    .map_err(backoff::Error::Permanent)?;

                if status.as_u16() == 429
                    // API returns 429 also when:
                    // "You exceeded your current quota, please check your plan and billing details."
                    && wrapped_error.error.r#type != Some("insufficient_quota".to_string())
                {
                    // Rate limited retry...
                    tracing::warn!("Rate limited: {}", wrapped_error.error.message);
                    return Err(backoff::Error::Transient {
                        err: LlamaApiError::ApiError(wrapped_error.error),
                        retry_after: None,
                    });
                } else {
                    return Err(backoff::Error::Permanent(LlamaApiError::ApiError(
                        wrapped_error.error,
                    )));
                }
            }

            Ok(bytes)
        })
        .await
    }

    /// Execute a HTTP request and retry on rate limit
    ///
    /// request_maker serves one purpose: to be able to create request again
    /// to retry API call after getting rate limited. request_maker is async because
    /// reqwest::multipart::Form is created by async calls to read files for uploads.
    async fn execute<O, M, Fut>(&self, request_maker: M) -> Result<O, LlamaApiError>
    where
        O: DeserializeOwned,
        M: Fn() -> Fut,
        Fut: core::future::Future<Output = Result<reqwest::Request, LlamaApiError>>,
    {
        let bytes = self.execute_raw(request_maker).await?;

        let response: O = serde_json::from_slice(bytes.as_ref())
            .map_err(|e| map_deserialization_error(e, bytes.as_ref()))?;

        Ok(response)
    }
}
