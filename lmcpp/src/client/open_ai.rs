//! OpenAI‑compatible endpoints for **`LmcppServer`**
//!
//! This module exposes **thin, untyped** helpers that forward directly to
//! llama.cpp’s OpenAI‑style HTTP surface (`/v1/*`).  Each helper takes and
//! returns a `serde_json::Value`, giving callers full freedom to build requests
//! with `serde_json::json!`, `serde_json::from_str`, or any other JSON API.
//! The transport layer itself still works with raw bytes; JSON
//! (de)serialisation happens only at the edge.
//!
//! ## Public API
//!
//! ```text
//!   LmcppServer::open_ai_v1_models()             ─▶ GET /v1/models
//!   LmcppServer::open_ai_v1_completions(body)    ─▶ POST /v1/completions
//!   LmcppServer::open_ai_v1_chat_completions(..) ─▶ POST /v1/chat/completions
//!   LmcppServer::open_ai_v1_embeddings(..)       ─▶ POST /v1/embeddings
//! ```
//!
//! Internally, everything funnels through the private
//! `LmcppServer::openai_json` helper.  That function selects `GET` or `POST`,
//! attaches the JSON body (if any), and delegates to the lower‑level
//! `ServerClientExt` shims.  When we eventually migrate to fully‑typed
//! request/response structs, only `openai_json` will need to change—no
//! transport code will be touched.
//!
//! ### Call Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  application (serde_json::Value)            │
//! └───────────────┬─────────────────────────────┘
//!                 │
//!                 ▼
//!   LmcppServer::<endpoint>()  (serde_json)      │
//!                 │                             │ JSON
//!                 ▼                             │
//!       LmcppServer::openai_json()              │
//!                 │                             │ bytes
//!                 ▼                             │
//!   ServerClientExt::{get, post} ──► llama.cpp HTTP server
//! ```
//!
//! ### Caveats
//!
//! * The wrapper performs **no schema validation**; malformed requests are
//!   forwarded as‑is and the server’s error response is returned verbatim.

use serde_json::Value;

use crate::{LmcppServer, error::LmcppResult, server::ipc::ServerClientExt};

impl LmcppServer {
    /// Sends a JSON request to `path` and returns the raw JSON response.
    ///
    /// * `method` – "GET" or "POST" (others are unreachable).
    /// * `path`   – Endpoint starting with a leading slash, e.g. "/v1/models".
    /// * `body`   – `None` for GET or empty‑body POST, otherwise the JSON payload.
    fn openai_json(
        &self,
        method: &'static str,
        path: &str,
        body: Option<&Value>,
    ) -> LmcppResult<Value> {
        match (method, body) {
            ("GET", _) => self.client.get(path),
            ("POST", Some(json)) => self.client.post(path, json),
            ("POST", None) => self.client.post(path, &Value::Null),
            _ => unreachable!("unsupported HTTP verb"),
        }
        .map_err(Into::into)
    }

    /// GET `/v1/models` – returns model metadata list
    pub fn open_ai_v1_models(&self) -> LmcppResult<Value> {
        self.openai_json("GET", "/v1/models", None)
    }

    /// POST `/v1/completions` – classic completion endpoint
    pub fn open_ai_v1_completions(&self, body: &Value) -> LmcppResult<Value> {
        self.openai_json("POST", "/v1/completions", Some(body))
    }

    /// POST `/v1/chat/completions` – ChatML / tool‑call capable endpoint
    pub fn open_ai_v1_chat_completions(&self, body: &Value) -> LmcppResult<Value> {
        self.openai_json("POST", "/v1/chat/completions", Some(body))
    }

    /// POST `/v1/embeddings` – embedding vectors for one or many inputs
    pub fn open_ai_v1_embeddings(&self, body: &Value) -> LmcppResult<Value> {
        self.openai_json("POST", "/v1/embeddings", Some(body))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use serial_test::serial;

    use super::*;
    use crate::*;

    #[test]
    #[ignore]
    #[serial]
    fn test_v1_models() -> LmcppResult<()> {
        let server = LmcppServerLauncher::default().load()?;
        let resp = server.open_ai_v1_models()?;
        println!("Models: {:#?}", resp);
        assert_eq!(resp["object"], "list");
        let first_id = &resp["data"][0]["id"];
        println!("First model id: {}", first_id);
        assert!(first_id.is_string());
        Ok(())
    }

    #[test]
    #[ignore]
    #[serial]
    fn test_v1_completions() -> LmcppResult<()> {
        let server = LmcppServerLauncher::default().load()?;
        let req = json!({
            "prompt": "Rust is…",
            "max_tokens": 8
        });
        let resp = server.open_ai_v1_completions(&req)?;
        println!("Completion: {:#?}", resp);
        let first = &resp["choices"][0]["text"];
        assert!(first.is_string());
        println!("Assistant said: {}", first);
        Ok(())
    }

    #[test]
    #[ignore]
    #[serial]
    fn test_v1_chat_completions() -> LmcppResult<()> {
        let server = LmcppServerLauncher::default().load()?;
        let req = json!({
            "messages": [
                {"role": "user", "content": "Say hello in French"}
            ]
        });
        let resp = server.open_ai_v1_chat_completions(&req)?;
        println!("Chat completion: {:#?}", resp);
        let first = &resp["choices"][0]["message"]["content"];
        assert!(first.is_string());
        println!("Assistant said: {}", first);

        Ok(())
    }

    #[test]
    #[ignore]
    #[serial]
    fn test_v1_embeddings() -> LmcppResult<()> {
        // Launch in embeddings‑only mode to speed things up.
        let server = LmcppServerLauncher::builder()
            .server_args(
                ServerArgs::builder()
                    .pooling(Pooling::Cls) // Requires pooling to be set when loading?
                    .embeddings_only(true)
                    .default_model()?
                    .build(),
            )
            .load()?;

        let req = json!({
            "input": "LLMs are awesome.",
            "encoding_format": "float",
        });
        let resp = server.open_ai_v1_embeddings(&req)?;
        println!("Embedding usage: {:#?}", resp["usage"]);

        let data = resp["data"].as_array().unwrap();
        println!("Embeddings returned: {:#?}", data.len());

        let vec_len = data
            .get(0)
            .and_then(|e| e["embedding"].as_array())
            .map_or(0, |v| v.len());
        println!("Length of first embedding vector: {:#?}", vec_len);

        Ok(())
    }
}
