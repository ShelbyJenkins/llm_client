//! Server IPC – Unix Domain Socket (UDS) client
//! ============================================
//!
//! A tiny **HTTP-over-UDS** client used by the integration tests of this crate.
//! It transparently handles platform differences (`UnixStream` vs `TcpStream` on
//! Windows named pipes), re-uses a cached stream when possible, enforces read /
//! write time-outs, and understands the subset of HTTP it needs
//! (fixed-length, `chunked`, and `204/304` empty responses).
//!
//! The companion `ServerClient` trait abstracts over transport so the rest of
//! the code base can stay platform-agnostic.

#[cfg(unix)]
use std::os::{
    fd::IntoRawFd,
    unix::{io::FromRawFd, net::UnixStream},
};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    path::PathBuf,
    time::Duration,
};
#[cfg(windows)]
use std::{
    net::TcpStream,
    os::windows::io::{FromRawSocket, IntoRawSocket},
};

use socket2::{Domain, SockAddr, Socket, Type};

pub use super::ServerClient;
use super::error::*;

#[cfg(unix)]
type IpcStream = UnixStream;

#[cfg(windows)]
type IpcStream = TcpStream;

const TIMEOUT: Duration = Duration::from_secs(180);
const CRLF: &str = "\r\n";
const NL: &str = "\n";
const UDS_MAX: usize = 91; //  108 incl. NUL byte,  104 on Solaris, 92 on macOS

/// High-level HTTP client that talks to a server process through a
/// **temporary Unix-domain socket** (or Windows named pipe).
///
/// The client:
/// * lazily establishes a connection on first use and caches it,
/// * attaches 180 s read/write time-outs (configurable per request),
/// * silently recycles the stream on I/O or (de-)serialisation errors, and
/// * limits response bodies to **16 MiB** as a defence-in-depth measure.
#[derive(Debug)]
pub struct UdsClient {
    /// Absolute path of the freshly created, process-local socket.
    base_url: PathBuf,
    /// Cached stream. `None` until the first request succeeds.
    stream: std::sync::RwLock<Option<IpcStream>>,
    /// Default time-out applied to `get_raw` / `post_raw`.
    timeout: Duration,
    /// Identifier for the server derived from the executable name, schema, and UUID.
    pid_id: String,
}

impl UdsClient {
    pub fn new(executable_name: &str) -> Result<Self> {
        debug_assert!(!executable_name.is_empty());
        let (base_url, pid_id) = Self::next_free_id(executable_name, 3)?;
        debug_assert!(base_url.as_os_str().len() <= UDS_MAX);

        let client = UdsClient {
            base_url,
            stream: None.into(),
            timeout: TIMEOUT,
            pid_id,
        };
        crate::trace!("Client created: {client}");
        Ok(client)
    }

    /// Convenience wrapper that normalises error handling:
    ///
    /// * Any low-level error (`Io`, `Timeout`, `Serde`) clears the cached
    ///   stream so the next request starts fresh.
    /// * Responds with the parsed body on success.
    fn send(
        &self,
        verb: &'static str,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<Vec<u8>> {
        match self.send_inner(verb, path, body, timeout) {
            Ok((result, saw_conn_close)) => {
                // If the server indicated it will close the connection, drop the stream.
                if saw_conn_close {
                    self.stream.write().expect("Failed to lock stream").take();
                }
                // Optional debug logging of the entire body.
                #[cfg(debug_assertions)]
                {
                    let pretty = Self::render_body(&result);
                    crate::trace!("{pretty}");
                    println!("{pretty}");
                }
                Ok(result)
            }
            Err(e) => {
                if matches!(
                    e,
                    ClientError::Io(_) | ClientError::Timeout(_) | ClientError::Serde(_)
                ) {
                    self.stream.write().expect("Failed to lock stream").take();
                }
                Err(e)
            }
        }
    }

    /// Core request/response state machine.  
    /// See source for exhaustive inline assertions.
    fn send_inner(
        &self,
        verb: &'static str,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<(Vec<u8>, bool)> {
        // Prevents header-injection vulnerabilities during development.
        debug_assert!(path.starts_with('/') && !path.contains('\r') && !path.contains('\n'));

        // Establish a stream to the server.

        let mut stream_guard = self.acquire_stream(timeout)?;
        let stream = stream_guard.as_mut().unwrap();

        // Use an empty slice when the caller supplies no body.
        let body = body.unwrap_or_default();
        let len = body.len(); // cache for reuse in the header

        // Write the request line and headers in one buffered burst.
        {
            let mut writer = std::io::BufWriter::new(&mut *stream);
            write!(
                writer,
                "{verb} {path} HTTP/1.1{CRLF}\
     Host: localhost{CRLF}\
     Content-Type: application/json{CRLF}\
     Content-Length: {len}{CRLF}\
     Connection: close{CRLF}{CRLF}"
            )?;
            if len > 0 {
                writer.write_all(body)?; // copy payload bytes, if any
            }
            writer.flush()?; // send everything immediately
        } // writer dropped here, releasing its borrow on the stream

        // Set up a buffered reader for the response.
        let mut reader = BufReader::new(&mut *stream);

        // Read the status line (e.g. "HTTP/1.1 200 OK\r\n").
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let status: u16 = line
            .split_whitespace()
            .nth(1) // second token is the status code
            .and_then(|s| s.parse().ok())
            .unwrap_or(500); // default to 500 if parsing fails

        // Surfaces protocol desynchronisation early instead of silently treating the response as a 500.
        debug_assert!((100..=599).contains(&status));

        // Scan headers we care about (Content-Length and Transfer-Encoding).
        let mut saw_conn_close = false;
        let mut content_len: Option<usize> = None;
        let mut chunked = false;
        loop {
            line.clear();
            reader.read_line(&mut line)?;
            if line == CRLF || line == NL {
                // Blank line marks the end of headers.
                break;
            }

            let header = line.to_ascii_lowercase();
            // mark if the peer plans to close
            if header.trim() == "connection: close" {
                saw_conn_close = true;
            }
            if let Some(v) = header.strip_prefix("content-length:") {
                content_len = v.trim().parse::<usize>().ok();
            } else if header.trim() == "transfer-encoding: chunked" {
                chunked = true;
            }
        }

        // Alerts during testing when a server sends an illegal combination we would mishandle.
        debug_assert!(!(chunked && content_len.is_some()));

        // Build a descriptive ClientError for any non-2xx status.
        if status / 100 != 2 {
            let message = {
                #[cfg(debug_assertions)]
                {
                    let body = Self::read_body(&mut reader, chunked, content_len)?;
                    format!("{} — {}", line.trim_end(), Self::render_body(&body).trim())
                }
                #[cfg(not(debug_assertions))]
                {
                    line.trim_end().to_owned()
                }
            };
            return Err(ClientError::Remote {
                code: status,
                message,
            });
        }

        // Fast-return for bodies that must be empty.
        if status == 204 || status == 304 || content_len == Some(0) {
            return Ok((b"null".to_vec(), saw_conn_close));
        }
        // When Content-Length is present, read exactly that many bytes.
        if let Some(len) = content_len {
            let mut buf = Vec::with_capacity(len); //  pre-allocates in one shot
            reader.take(len as u64).read_to_end(&mut buf)?; //  reads exactly len bytes
            return Ok((buf, saw_conn_close)); //  hands raw bytes back
        }

        // Otherwise fall back to the capped, chunk-aware reader.
        let body_buf = Self::read_body(&mut reader, chunked, content_len)?;

        // Parse the JSON into the caller’s requested type.
        Ok((body_buf, saw_conn_close))
    }

    /// Reads an HTTP body, honouring *either* `chunked` transfer encoding
    /// *or* `Content-Length`.  Caps the total at 16 MiB.
    #[inline]
    fn read_body<R: Read>(
        reader: &mut BufReader<R>,
        chunked: bool,
        content_len: Option<usize>,
    ) -> io::Result<Vec<u8>> {
        const MAX_BYTES: u64 = 16 * 1024 * 1024; // upper limit for defence in depth
        let mut body = Vec::new();

        if chunked {
            // Manually decode each chunk, tracking how much room is left under MAX_BYTES.
            let mut remaining = MAX_BYTES as usize;
            loop {
                let mut sz_line = String::new();
                if reader.read_line(&mut sz_line)? == 0 {
                    break; // unexpected end of stream
                }
                let sz = usize::from_str_radix(sz_line.trim(), 16).unwrap_or(0);

                debug_assert!(
                    sz <= remaining,
                    "chunk size {sz} exceeds remaining budget {remaining}"
                );

                if sz == 0 || remaining == 0 {
                    // Zero-length chunk signals end; also exit if cap is reached.
                    let _ = reader.read_line(&mut sz_line); // consume trailing CRLF
                    break;
                }
                let take = sz.min(remaining);
                let mut chunk = vec![0u8; take];
                reader.read_exact(&mut chunk)?;
                body.extend_from_slice(&chunk);
                remaining -= take;
                let _ = reader.read_line(&mut sz_line); // CRLF after chunk
                if remaining == 0 {
                    break;
                }
            }
        } else {
            // Non-chunked: respect Content-Length if present, never exceed MAX_BYTES.
            let limit = content_len
                .map(|n| n as u64)
                .unwrap_or(MAX_BYTES)
                .min(MAX_BYTES);
            reader.take(limit).read_to_end(&mut body)?;
        }

        Ok(body)
    }

    /// Pretty-prints JSON when possible; otherwise falls back to lossy UTF-8.
    #[inline]
    fn render_body(buf: &[u8]) -> String {
        // Try JSON pretty-print first …
        match serde_json::from_slice::<serde_json::Value>(buf) {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            // … fallback: interpret as loss-y UTF-8 for arbitrary text / binary
            Err(_) => String::from_utf8_lossy(buf).into_owned(),
        }
    }

    /// Picks a unique socket path in the system temp directory and *pre-binds*
    /// it to ensure no competitor grabs it first.
    fn next_free_id(executable_name: &str, max_attempts: usize) -> Result<(PathBuf, String)> {
        debug_assert!(!executable_name.is_empty());
        debug_assert!(max_attempts > 0);

        let base_dir = std::env::temp_dir();

        for _ in 0..max_attempts {
            let uid = uuid::Uuid::new_v4().hyphenated().to_string();
            let endsplit = uid
                .rsplit('-')
                .next()
                .expect("UUID should have at least one part");
            // Now get last part of the UID
            // and append it to the executable name to create a unique socket path.
            let pid_id = sanitize_filename::sanitize(
                format!("{executable_name}_unix_{endsplit}").to_ascii_lowercase(),
            );
            if pid_id.len() > 240 {
                return Err(ClientError::Setup {
                    reason: format!("pid_id \"{pid_id}\" exceeds 240 characters"),
                });
            }
            let name = format!("{executable_name}-{endsplit}.sock").to_ascii_lowercase();
            let path = base_dir.join(&name);

            // skip if path would overflow the platform limit
            if path.as_os_str().len() > UDS_MAX {
                return Err(ClientError::Setup {
                    reason: format!(
                        "UDS path length {} exceeds platform limit of {} bytes",
                        path.as_os_str().len(),
                        UDS_MAX
                    ),
                });
            }

            //  probe the path
            let sock = Socket::new(Domain::UNIX, Type::STREAM, None)?;
            match sock.bind(&SockAddr::unix(&path)?) {
                Ok(_) => {
                    drop(sock); // close FD
                    let _ = std::fs::remove_file(&path);
                    return Ok((path, pid_id));
                }
                Err(ref e) if e.kind() == io::ErrorKind::AddrInUse => continue,
                Err(e) => {
                    return Err(ClientError::Setup {
                        reason: format!("Failed to bind to UDS path {path:?}: {e}"),
                    });
                }
            }
        }
        Err(ClientError::Setup {
            reason: format!("could not reserve a free UDS endpoint after {max_attempts} attempts"),
        })
    }

    /// Returns a **mutable borrow** to the cached stream, opening a new one on
    /// demand and installing the requested time-out on both directions.
    fn acquire_stream(
        &self,
        timeout: Duration,
    ) -> Result<std::sync::RwLockWriteGuard<'_, Option<IpcStream>>> {
        // Enforce a positive timeout at development time.
        debug_assert!(timeout > Duration::ZERO);

        // Take ONE mutable borrow for the whole function.
        let mut borrow = self.stream.write().expect("Failed to lock stream");

        if borrow.is_none() {
            let s = self.connect_stream()?;
            // Apply read- and write-side timeouts so a misbehaving peer cannot stall forever.
            s.set_read_timeout(Some(timeout))?;
            s.set_write_timeout(Some(timeout))?;
            *borrow = Some(s);
        }

        Ok(borrow)
    }

    /// Platform-specific connector that turns a raw FD / SOCKET into an
    /// `IpcStream` implementing `Read + Write`.
    #[cfg(unix)]
    fn connect_stream(&self) -> io::Result<IpcStream> {
        let sock = Socket::new(Domain::UNIX, Type::STREAM, None)?;
        sock.connect(&SockAddr::unix(&self.base_url)?)?;

        // Ensure the OS returned a valid (non-negative) file descriptor
        assert!(
            std::os::fd::AsRawFd::as_raw_fd(&sock) >= 0,
            "connect_stream: received an invalid raw fd"
        );

        // SAFETY: we own the fd; into_raw_fd would leak
        Ok(unsafe { UnixStream::from_raw_fd(sock.into_raw_fd()) })
    }

    /// Platform-specific connector that turns a raw FD / SOCKET into an
    /// `IpcStream` implementing `Read + Write`.
    #[cfg(windows)]
    fn connect_stream(&self) -> io::Result<IpcStream> {
        use std::os::windows::io::AsRawSocket;

        let sock = Socket::new(Domain::UNIX, Type::STREAM, None)?;
        sock.connect(&SockAddr::unix(&self.base_url)?)?;

        // winsock uses all-bits-set (usize::MAX) as its INVALID_SOCKET value
        assert!(
            AsRawSocket::as_raw_socket(&sock) != u64::MAX,
            "connect_stream: received an invalid raw socket"
        );

        // SAFETY: we own the SOCKET returned by `sock`; transferring ownership to `TcpStream`
        // gives us a type that already implements `Read`, `Write`, and the timeout setters.
        Ok(unsafe { TcpStream::from_raw_socket(sock.into_raw_socket()) })
    }

    #[cfg(test)]
    pub fn dummy() -> Self {
        let dummy_endpoint = std::env::temp_dir().join("dummy.sock");
        Self {
            base_url: dummy_endpoint,
            stream: None.into(),
            timeout: TIMEOUT,
            pid_id: "dummy_uds_client".to_string(),
        }
    }
}

impl Drop for UdsClient {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            crate::error!("Failed to stop UdsClient during drop: {e}");
        }
    }
}

impl std::fmt::Display for UdsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UdsClient({:#?})", self.base_url)
    }
}

impl ServerClient for UdsClient {
    fn get_raw(&self, path: &str) -> Result<Vec<u8>> {
        self.send("GET", path, None, self.timeout)
    }

    fn post_raw(&self, path: &str, body: &[u8]) -> Result<Vec<u8>> {
        self.send("POST", path, Some(body), self.timeout)
    }

    fn stop(&self) -> Result<()> {
        match std::fs::remove_file(&self.base_url) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), // already gone → fine
            Err(e) => Err(ClientError::Io(e)),                            // propagate
        }
    }

    fn host(&self) -> String {
        self.base_url.to_string_lossy().to_string()
    }

    fn pid_id(&self) -> String {
        self.pid_id.to_string()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::{
        io::{Read, Write},
        path::Path,
        thread,
        time::Duration,
    };

    use super::*;
    #[cfg(unix)]
    use crate::server::ipc::ServerClientExt;

    /// ────────────────────────────────────────────────────────────────────────
    /// Tiny one-shot Unix-domain HTTP server used by several scenarios below.
    /// ────────────────────────────────────────────────────────────────────────
    #[cfg(unix)]
    fn spawn_uds_server(path: &Path, response: &[u8], keep_open: bool) -> thread::JoinHandle<()> {
        use std::os::unix::net::UnixListener;

        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path).expect("bind test Unix socket");
        let resp = response.to_owned();
        let p = path.to_owned();

        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Read the request header so the client can write without SIGPIPE.
                let _ = {
                    let mut buf = [0u8; 512];
                    stream.read(&mut buf).ok()
                };

                stream.write_all(&resp).unwrap();
                stream.flush().unwrap();

                if keep_open {
                    thread::sleep(Duration::from_millis(200));
                }
            }
            let _ = std::fs::remove_file(&p);
        })
    }

    /* ───────────────────────── next_free_endpoint variants ───────────────────────── */

    #[test]
    fn next_free_endpoint_variants() {
        let cases: Vec<(String, bool)> = vec![
            ("unit-core".into(), true),        // happy path
            ("x".repeat(UDS_MAX + 64), false), // guaranteed overflow
        ];

        for (exe, should_succeed) in cases {
            let res = UdsClient::next_free_id(&exe, 3);
            if should_succeed {
                let (path, _) = res.expect("expected Ok");
                assert!(
                    std::fs::metadata(&path).is_err(),
                    "socket file should have been cleaned up"
                );
            } else {
                assert!(res.is_err(), "expected Err for oversize path");
            }
        }
    }

    /* ───────────────────────── unified UDS response scenarios ─────────────────────── */

    #[cfg(unix)]
    #[test]
    fn uds_response_scenarios() {
        use serde_json::json;

        struct Case {
            name: &'static str,
            http: String,
            keep_open: bool,
            expect: std::result::Result<serde_json::Value, u16>,
            expect_stream_cached: bool,
        }

        /* ── canned HTTP responses ── */
        let fixed_body = br#"{"ok":true}"#;
        let fixed_http = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            fixed_body.len(),
            std::str::from_utf8(fixed_body).unwrap()
        );

        let chunked_body = r#"{"hello":"world"}"#;
        let chunked_http = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            chunked_body.len(),
            chunked_body
        );

        let conn_close_http =
            "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned();

        let no_content_http = "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n".to_owned();
        let err_500_http =
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n".to_owned();

        /* ── table of scenarios ── */
        let cases = vec![
            Case {
                name: "fixed_len_ok",
                http: fixed_http,
                keep_open: true,
                expect: Ok(json!({"ok": true})),
                expect_stream_cached: true,
            },
            Case {
                name: "chunked_ok",
                http: chunked_http,
                keep_open: false,
                expect: Ok(json!({"hello": "world"})),
                expect_stream_cached: true, // no “Connection: close” header ⇒ keep
            },
            Case {
                name: "conn_close",
                http: conn_close_http,
                keep_open: false,
                expect: Ok(serde_json::Value::Null),
                expect_stream_cached: false,
            },
            Case {
                name: "no_content_204",
                http: no_content_http,
                keep_open: false,
                expect: Ok(serde_json::Value::Null),
                expect_stream_cached: true,
            },
            Case {
                name: "err_500",
                http: err_500_http,
                keep_open: false,
                expect: Err(500),
                expect_stream_cached: true, // error is `Remote`, stream retained
            },
        ];

        for case in cases {
            let (sock_path, _) =
                UdsClient::next_free_id(&case.name, 3).expect("should get a free path");
            let _srv = spawn_uds_server(&sock_path, case.http.as_bytes(), case.keep_open);

            let mut client = UdsClient::dummy();
            client.base_url = sock_path;

            match case.expect {
                Ok(ref wanted) => {
                    let v: serde_json::Value = client
                        .get::<serde_json::Value>("/")
                        .expect("request should succeed");
                    assert_eq!(&v, wanted, "case `{}` JSON mismatch", case.name);
                }
                Err(code) => {
                    let err = client
                        .get::<serde_json::Value>("/")
                        .expect_err("expected error");
                    match err {
                        ClientError::Remote { code: c, .. } => {
                            assert_eq!(c, code, "case `{}` wrong status code", case.name)
                        }
                        other => panic!("case `{}` wrong error variant: {other:?}", case.name),
                    }
                }
            }

            assert_eq!(
                client
                    .stream
                    .read()
                    .expect("Failed to lock stream")
                    .is_some(),
                case.expect_stream_cached,
                "case `{}` unexpected stream-caching behaviour",
                case.name
            );
        }
    }

    /* ───────────────────────── grouped “error-path” checks ───────────────────────── */

    #[test]
    fn client_error_paths() {
        // invalid address → connect_stream must error
        let dummy = UdsClient::dummy();
        assert!(dummy.connect_stream().is_err(), "invalid path must error");

        // stop() is idempotent
        let c = UdsClient::dummy();
        c.stop().expect("first stop() must succeed");
        c.stop().expect("second stop() must remain a no-op");
    }

    /* ───────────────────────── render_body helper ───────────────────────── */

    #[test]
    fn render_body_pretty_prints_json_and_survives_binary() {
        let pretty = UdsClient::render_body(br#"{"n":1}"#);
        assert!(
            pretty.contains('\n'),
            "JSON should be pretty-printed (contain newline)"
        );

        let bin = b"\xFFrandom\xFE";
        let rendered = UdsClient::render_body(bin);
        assert!(
            rendered.contains("random"),
            "binary body should be lossy-converted to UTF-8"
        );
    }
}
