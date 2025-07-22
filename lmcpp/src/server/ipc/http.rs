//! Server IPC – HTTP Client
//! ====================================
//!
//! Thin wrapper around [`ureq`] that provides an internal
//! [`HttpClient`] implementing the crate‑local [`ServerClient`] trait.
//!
//! The client is used mainly by integration tests and inter‑process
//! communication (IPC) helpers.  Its goals are **zero configuration**,
//! **one connection‑pool per client**, and **predictable failure
//! semantics** (all transport errors are mapped to `ClientError`).
//!
//! ## Design Highlights
//! * **Ephemeral port discovery** – When no port is supplied, the
//!   constructor binds to port `0`, asks the OS for a free port, then
//!   immediately releases the listener so the forthcoming server can
//!   reuse the same port.
//! * **Global timeout** – A hard, three‑minute limit (see
//!   [`TIMEOUT`]) applies to the *entire* request: connect + read +
//!   write.
//! * **Cheap clones** – Cloning an `HttpClient` only bumps a reference
//!   count; the underlying connection‑pool lives in a single
//!   [`ureq::Agent`].
//! * **Raw‑byte interface** – Higher‑level JSON helpers are provided by
//!   `server::ipc::ServerClientExt`; this module stays transport‑only.

use std::{
    io,
    io::Read,
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener},
    time::Duration,
};

use ureq::Agent;

pub use super::ServerClient;
use super::error::*;

/// Default request timeout (connect + read + write).
const TIMEOUT: Duration = Duration::from_secs(180);

/// Default host when none is provided to [`HttpClient::new`].
const HOST: &str = "127.0.0.1";

/// Lightweight HTTP/1.1 client backed by [`ureq`].
///
/// The client offers two *transport‑level* primitives—[`get_raw`]
/// and [`post_raw`]—required by the [`ServerClient`] trait.  Higher‑level
/// helpers that (de)serialise JSON live elsewhere so this type remains
/// dependency‑free except for *ureq*.
///
/// Cloning the struct shares the same connection‑pool; therefore it is
/// cheap and thread‑safe.
#[derive(Debug)]
pub struct HttpClient {
    /// Underlying *ureq* connection‑pool and HTTP state‑machine.
    agent: Agent,
    /// Identifier for the server derived from the executable name, schema, host, and port.
    pid_id: String,
    /// Prefix shared by every request, e.g. `http://127.0.0.1:8080`.
    base_url: String,
    /// Hostname portion of `base_url`.
    pub host: String,
    /// TCP port portion of `base_url`.
    pub port: u16,
}

impl HttpClient {
    /// Creates a client whose `base_url` is `http://127.0.0.1:<free‑port>`.
    pub fn new(executable_name: &str, host: Option<&str>, port: Option<u16>) -> Result<Self> {
        let host = host.unwrap_or_else(|| HOST).to_string();
        let port = if let Some(port) = port {
            port
        } else {
            // Bind to an ephemeral port.
            let listener: TcpListener =
                TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).map_err(
                    |e| ClientError::Setup {
                        reason: format!("failed to obtain an ephemeral port: {e}"),
                    },
                )?;

            let port = listener
                .local_addr()
                .map_err(|e| ClientError::Setup {
                    reason: format!("could not read local address: {e}"),
                })?
                .port();
            drop(listener); // close the listener, we only need the port
            port
        };

        // One agent → one connection‑pool (threads can clone HttpClient cheaply).
        let agent = Agent::new_with_config(
            Agent::config_builder()
                .timeout_global(Some(TIMEOUT)) // applies to connect + read + write
                .build(),
        );

        let pid_id = sanitize_filename::sanitize(
            format!("{executable_name}_http_{host}_{port}").to_ascii_lowercase(),
        );
        if pid_id.len() > 240 {
            return Err(ClientError::Setup {
                reason: format!("pid_id \"{pid_id}\" exceeds 240 characters"),
            });
        }
        let client = Self {
            base_url: format!("http://{host}:{port}"),
            host,
            agent,
            port,
            pid_id,
        };
        crate::trace!("Client created: {client}");
        Ok(client)
    }

    /// Internal helper that performs the request and maps *ureq* errors
    /// to our unified [`ClientError`] enum.
    ///
    /// * `verb` – `"GET"` or `"POST"`.
    /// * `path` – Must start with `/`.
    /// * `body` – `None` for GET, `Some` for POST (empty slice permitted).
    fn send(&self, verb: &'static str, path: &str, body: Option<&[u8]>) -> Result<Vec<u8>> {
        debug_assert!(path.starts_with('/'));
        let url = format!("{}{}", self.base_url, path);

        // ── build & execute the request, getting the *response* ───────────────
        let response = match (verb, body) {
            ("GET", _) => self.agent.get(&url).call(),

            // -------- POST with body ----------------------------------------
            ("POST", Some(b)) if !b.is_empty() => self
                .agent
                .post(&url)
                .content_type("application/json")
                .send(b),

            ("POST", _) => self
                .agent
                .post(&url)
                .content_type("application/json")
                .send_empty(),

            _ => unreachable!("unsupported verb"),
        };

        match response {
            // ───────────── successful transport ─────────────
            Ok(resp) if (200..300).contains(&resp.status().as_u16()) => {
                let mut body = Vec::new();
                resp.into_body().into_reader().read_to_end(&mut body)?;
                Ok(body)
            }

            Ok(resp) => Err(ClientError::Remote {
                code: resp.status().as_u16(),
                message: resp
                    .status()
                    .canonical_reason()
                    .unwrap_or("unknown error")
                    .to_string(),
            }),
            Err(ureq::Error::StatusCode(code)) => Err(ClientError::Remote {
                code,
                message: format!("HTTP {code}"),
            }),

            Err(ureq::Error::Timeout(_)) => Err(ClientError::Timeout(TIMEOUT)),

            Err(ureq::Error::Io(e)) => Err(ClientError::Io(e)),

            Err(ureq::Error::Protocol(p)) => Err(ClientError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("protocol error: {p}"),
            ))),

            Err(ureq::Error::BadUri(u)) => Err(ClientError::Setup {
                reason: format!("bad URI: {u}"),
            }),

            Err(other) => Err(ClientError::Io(io::Error::new(
                io::ErrorKind::Other,
                format!("ureq error: {other}"),
            ))),
        }
    }

    #[cfg(test)]
    pub fn dummy() -> Self {
        HttpClient {
            base_url: format!("{HOST}:0"),
            agent: Agent::new_with_config(
                Agent::config_builder()
                    .timeout_global(Some(TIMEOUT)) // applies to connect + read + write
                    .build(),
            ),
            port: 0,
            pid_id: "dummy_http_client.pid".to_string(),
            host: HOST.to_string(),
        }
    }
}

impl std::fmt::Display for HttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HttpClient({:#?})", self.base_url)
    }
}

/* ───────────────────────── ServerClient impl ───────────────────────── */

impl ServerClient for HttpClient {
    fn get_raw(&self, path: &str) -> Result<Vec<u8>> {
        self.send("GET", path, None) // Just pass through!
    }

    fn post_raw(&self, path: &str, body: &[u8]) -> Result<Vec<u8>> {
        self.send("POST", path, Some(body)) // Just pass through!
    }

    fn stop(&self) -> Result<()> {
        // Nothing to tear down (no socket file); drop() closes pooled conns.
        Ok(())
    }

    fn host(&self) -> String {
        self.host.to_string()
    }

    fn pid_id(&self) -> String {
        self.pid_id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use serial_test::serial;

    use super::*;
    use crate::server::ipc::ServerClientExt;

    // ────────────────────────────────────────────────────────────────────────
    // Lightweight one‑shot HTTP server reused by every scenario.
    // ────────────────────────────────────────────────────────────────────────
    fn spawn_http_server(
        port: u16,
        response: &'static [u8],
        keep_open_ms: u64,
    ) -> std::thread::JoinHandle<()> {
        use std::{
            io::{Read, Write},
            net::{Shutdown, TcpListener},
        };

        let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind test HTTP socket");

        let reply = response.to_vec();

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Consume request so the client side won't get RST.
                let _ = {
                    let mut buf = [0u8; 512];
                    stream.read(&mut buf).ok()
                };

                stream.write_all(&reply).unwrap();
                stream.flush().unwrap();

                if keep_open_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(keep_open_ms));
                }
                // Close write‑side unless we advertised keep‑alive.
                let _ = stream.shutdown(Shutdown::Write);
            }
        })
    }

    /// Builds a minimal but well‑formed HTTP reply.
    fn make_reply(status_line: &str, headers: &[(&str, String)], body: &[u8]) -> String {
        let mut msg = String::new();

        msg.push_str(status_line);
        msg.push_str("\r\n");

        for (k, v) in headers {
            msg.push_str(k);
            msg.push_str(": ");
            msg.push_str(v);
            msg.push_str("\r\n");
        }

        // Empty line separates headers from body.
        msg.push_str("\r\n");

        // Body bytes.
        msg.push_str(std::str::from_utf8(body).unwrap());

        // No extra CR‑LF after the body because Content‑Length is exact.
        msg
    }

    /* ───────────────────────── response‑matrix ───────────────────────── */

    #[test]
    #[serial]
    fn http_response_scenarios() {
        struct Case {
            name: &'static str,
            reply: String,
            keep_open: u64, // milliseconds
            expect: std::result::Result<serde_json::Value, u16>,
        }

        // Fixed‑length 200 OK
        let fixed_body = br#"{"ok":true}"#;
        let fixed_ok = make_reply(
            "HTTP/1.1 200 OK",
            &[
                ("Content-Length", fixed_body.len().to_string()),
                ("Content-Type", "application/json".into()),
                ("Connection", "close".into()),
            ],
            fixed_body,
        );

        // Chunked 200 OK
        let chunk_body = br#"{"hello":"world"}"#;
        let chunked_ok = {
            let mut m = String::new();
            use std::fmt::Write as _;
            write!(
                &mut m,
                "HTTP/1.1 200 OK\r\n\
                 Transfer-Encoding: chunked\r\n\
                 Connection: close\r\n\r\n\
                 {:x}\r\n{}\r\n0\r\n\r\n",
                chunk_body.len(),
                std::str::from_utf8(chunk_body).unwrap(),
            )
            .unwrap();
            m
        };

        // 500 error
        let err_500 = make_reply(
            "HTTP/1.1 500 Internal Server Error",
            &[
                ("Content-Length", "0".into()),
                ("Connection", "close".into()),
            ],
            b"",
        );

        let cases = vec![
            Case {
                name: "fixed_len_ok",
                reply: fixed_ok,
                keep_open: 0, // we said Connection: close
                expect: Ok(json!({"ok": true})),
            },
            Case {
                name: "chunked_ok",
                reply: chunked_ok,
                keep_open: 0,
                expect: Ok(json!({"hello": "world"})),
            },
            Case {
                name: "err_500",
                reply: err_500,
                keep_open: 0,
                expect: Err(500),
            },
        ];

        for c in cases {
            // Fresh client with a free port.
            let client =
                HttpClient::new("dummy_http_client", None, None).expect("create HttpClient");
            let port = client.port;

            // Leak the reply so it lives for the thread lifetime.
            let reply_bytes: &'static [u8] =
                Box::<[u8]>::leak(c.reply.clone().into_bytes().into_boxed_slice());

            let _srv = spawn_http_server(port, reply_bytes, c.keep_open);

            match c.expect {
                Ok(ref wanted) => {
                    let got: serde_json::Value = client.get("/").expect("request should succeed");
                    assert_eq!(&got, wanted, "case `{}` JSON mismatch", c.name);
                }
                Err(code) => {
                    let err = client
                        .get::<serde_json::Value>("/")
                        .expect_err("expected error");
                    match err {
                        ClientError::Remote { code: c2, .. } => {
                            assert_eq!(c2, code, "case `{}` wrong status code", c.name)
                        }
                        other => panic!("case `{}` expected Remote error, got {other:?}", c.name),
                    }
                }
            }
        }
    }

    /* ───────────────────────── POST body handling ───────────────────────── */

    #[test]
    #[serial]
    fn http_post_sends_body_and_parses_response() {
        let body = br#"{"ack":true}"#;
        let reply = make_reply(
            "HTTP/1.1 200 OK",
            &[
                ("Content-Length", body.len().to_string()),
                ("Content-Type", "application/json".into()),
                ("Connection", "close".into()),
            ],
            body,
        );
        let client = HttpClient::new("dummy_http_client", None, None).expect("create HttpClient");
        let port = client.port;

        let reply_bytes: &'static [u8] = Box::<[u8]>::leak(reply.into_bytes().into_boxed_slice());
        let _srv = spawn_http_server(port, reply_bytes, 0);

        let payload = json!({"msg":"hi"});
        let v: serde_json::Value = client
            .post("/", &payload)
            .expect("POST request should succeed");

        assert_eq!(v, json!({"ack": true}));
    }
}
