use std::path::{Path, PathBuf};

use crate::error::{LmcppError, LmcppResult};

pub fn download_and_extract_zip(url: &str, working_dir: &Path, zip_file_name: &str) -> LmcppResult<()> {
    std::fs::create_dir_all(working_dir).map_err(|e| {
        LmcppError::file_system("create working directory", working_dir.to_path_buf(), e)
    })?;

    let zip_path = download_zip(url, working_dir, zip_file_name)?;
    extract_zip(working_dir, &zip_path)?;

    std::fs::remove_file(&zip_path).map_err(|e| {
        LmcppError::file_system("remove zip file after extraction", zip_path.clone(), e)
    })?;
    Ok(())
}

pub fn download_zip(url: &str, working_dir: &Path, zip_file_name: &str) -> LmcppResult<PathBuf> {
    assert!(
        url.starts_with("http"),
        "download_zip expects an http/https URL"
    );

    let zip_path = working_dir.join(format!("{zip_file_name}.zip"));

    // ── fast-path: cached & valid ───────────────────────────────────────────────
    if zip_path.exists() {
        let cached_ok = match std::fs::File::open(&zip_path) {
            Ok(file) => zip::ZipArchive::new(file).is_ok(),
            Err(_) => false,
        };

        if cached_ok {
            crate::trace!("Using cached archive {}", zip_path.display());
            return Ok(zip_path);
        }

        crate::warn!(
            "Cached archive {} is corrupted or incomplete; redownloading",
            zip_path.display()
        );
        let _ = std::fs::remove_file(&zip_path); // best-effort cleanup
    }
    // ── fresh download ──────────────────────────────────────────────────────────
    crate::trace!("Downloading from {} to {}", url, zip_path.display());
    fetch_with_ureq(url, &zip_path, /*tries=*/ 3)?;

    crate::trace!("Download completed successfully");
    Ok(zip_path)
}

pub fn extract_zip(working_dir: &Path, zip_path: &Path) -> LmcppResult<()> {
    let zip_file = std::fs::File::open(zip_path)
        .map_err(|e| LmcppError::file_system("open zip file", zip_path.to_path_buf(), e))?;

    let mut archive = zip::ZipArchive::new(zip_file)
        .map_err(|e| LmcppError::file_system("read zip archive", zip_path.to_path_buf(), e))?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| {
            LmcppError::file_system("read entry from zip archive", zip_path.to_path_buf(), e)
        })?;

        // ── 1. Derive a *sanitised* relative path ──────────────────────────────
        let rel = match entry.enclosed_name() {
            Some(p) => {
                // Drop the top-level folder (GitHub releases usually add one).
                let mut comps = p.components();
                comps.next(); // discard first component
                let tail = comps.as_path(); // still a slice of `p`

                if tail.as_os_str().is_empty() {
                    p.to_owned() // archive was already flat
                } else {
                    tail.to_owned()
                }
            }
            None => continue, // skip entries with invalid / malicious names
        };

        let out = working_dir.join(rel);

        // ── 2. Create directories or copy file contents ───────────────────────
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| {
                LmcppError::file_system("create directory from zip entry", out.clone(), e)
            })?;
            continue;
        }

        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LmcppError::file_system(
                    "create parent directory from zip entry",
                    parent.to_path_buf(),
                    e,
                )
            })?;
        }
        let mut out_file = std::fs::File::create(&out)
            .map_err(|e| LmcppError::file_system("create file from zip entry", out.clone(), e))?;

        std::io::copy(&mut entry, &mut out_file)
            .map_err(|e| LmcppError::file_system("copy contents from zip entry", out.clone(), e))?;

        // ── 3. Restore Unix permissions when present ──────────────────────────
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode)).map_err(|e| {
                LmcppError::file_system("set Unix permissions from zip entry", out.clone(), e)
            })?;
        }
    }

    Ok(())
}

/// Blocking HTTP GET with retries.
///
/// * `url`   – remote file to fetch  
/// * `dest`  – local path that will be **created/overwritten**  
/// * `tries` – 1 = no retry, 3 = initial try + 2 retries, …  
fn fetch_with_ureq(url: &str, dest: &Path, tries: u8) -> LmcppResult<()> {
    debug_assert!(tries >= 1, "fetch_with_ureq: tries must be ≥ 1");

    let mut failures = Vec::new();
    for attempt in 1..=tries {
        match ureq::get(url).call() {
            Ok(resp) if resp.status() == 200 => {
                let mut reader = resp.into_body().into_reader();
                let mut file = std::fs::File::create(dest).map_err(|e| {
                    LmcppError::file_system("create destination file", dest.to_path_buf(), e)
                })?;
                std::io::copy(&mut reader, &mut file).map_err(|e| {
                    LmcppError::file_system(
                        "copy contents to destination file",
                        dest.to_path_buf(),
                        e,
                    )
                })?;
                return Ok(());
            }
            Ok(resp) => {
                let msg = format!("HTTP {} (attempt {}/{})", resp.status(), attempt, tries);
                crate::warn!("{}", msg);
                failures.push(msg);
            }
            Err(e) => {
                let msg = format!("Request error {} (attempt {}/{})", e, attempt, tries);
                crate::warn!("{}", msg);
                failures.push(msg);
            }
        }

        // back-off: 1 s, 2 s, 4 s, capped at 8 s afterwards
        if attempt < tries {
            std::thread::sleep(std::time::Duration::from_secs(1 << attempt.min(3)));
        }
    }

    Err(LmcppError::DownloadFailed(format!(
        "Failed to download {url} after {tries} attempts:\n{}",
        failures.join("\n")
    )))
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
    };

    use tempfile::tempdir;
    use zip::{write::FileOptions, ZipWriter};

    use super::*;

    // ──────────────────────────────────────────────────────────────────────────
    // helpers – kept inline to obey “no extra functions” guideline
    // ──────────────────────────────────────────────────────────────────────────
    fn tiny_zip(path: &std::path::Path) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::write::ZipWriter::new(file);

        // specify the generics:  <name-type, extension-type>
        zip.start_file::<_, ()>("dummy.txt", zip::write::FileOptions::default())
            .unwrap();

        zip.write_all(b"hello").unwrap();
        zip.finish().unwrap();
    }

    // ✔️ Uses cached archive when zip is valid
    #[test]
    fn cached_archive_is_used() {
        let tmp = tempdir().unwrap();
        let work = tmp.path();

        // pre‑seed a valid archive
        tiny_zip(&work.join("cached.zip"));

        // first & second calls must both succeed without touching the network
        let url = "http://example.invalid/never-used.zip";
        for _ in 0..2 {
            let p = download_zip(url, work, "cached").unwrap();
            assert_eq!(p, work.join("cached.zip"));
        }
    }

    // ✔️ Redownloads when cache is corrupted
    #[test]
    fn corrupted_cache_triggers_redownload() {
        let tmp = tempdir().unwrap();
        let work = tmp.path();
        let zip_path = work.join("bad.zip");

        // corrupt placeholder
        fs::write(&zip_path, b"garbage").unwrap();
        let old_size = fs::metadata(&zip_path).unwrap().len();

        // stand‑in HTTP server that provides a *real* zip
        let mut server = mockito::Server::new();
        let endpoint = "/bad.zip";
        let zip_bytes = {
            let mut buf = Vec::<u8>::new();
            {
                let mut z = ZipWriter::new(std::io::Cursor::new(&mut buf));
                z.start_file::<_, ()>("file.txt", FileOptions::default())
                    .unwrap();
                z.write_all(b"hello").unwrap();
                z.finish().unwrap();
            }
            buf
        };
        let _m = server
            .mock("GET", endpoint)
            .with_status(200)
            .with_body(zip_bytes.clone())
            .create();

        let url = format!("{}{}", server.url(), endpoint);
        let p = download_zip(&url, work, "bad").unwrap();
        let new_size = fs::metadata(p).unwrap().len();

        assert!(new_size > old_size, "file should have been replaced");
    }

    // ✔️ extract_zip strips the top‑level folder
    #[test]
    fn extract_strips_top_folder() {
        let tmp = tempdir().unwrap();
        let work = tempdir().unwrap();

        let zip_path = tmp.path().join("outer.zip");
        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = ZipWriter::new(file);
            zip.start_file::<_, ()>("outer/inner/file.txt", FileOptions::default())
                .unwrap();
            zip.write_all(b"content").unwrap();
            zip.finish().unwrap();
        }

        extract_zip(work.path(), &zip_path).unwrap();
        assert!(work.path().join("inner/file.txt").exists());
        assert!(!work.path().join("outer").exists());
    }

    // ✔️ extract_zip restores executable bits on Unix
    #[cfg(unix)]
    #[test]
    fn extract_respects_permissions_unix() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempdir().unwrap();
        let work = tempdir().unwrap();
        let zip_path = tmp.path().join("perm.zip");

        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = ZipWriter::new(file);
            let opts = FileOptions::default().unix_permissions(0o755);
            zip.start_file::<_, ()>("script.sh", opts).unwrap();
            zip.write_all(b"#!/bin/sh\necho hi\n").unwrap();
            zip.finish().unwrap();
        }

        extract_zip(work.path(), &zip_path).unwrap();
        let meta = fs::metadata(work.path().join("script.sh")).unwrap();
        assert!(
            meta.permissions().mode() & 0o111 != 0,
            "executable bit lost"
        );
    }

    // ✔️ Path‑traversal entries are ignored
    #[test]
    fn extract_skips_path_traversal() {
        let tmp = tempdir().unwrap();
        let work = tempdir().unwrap();
        let zip_path = tmp.path().join("evil.zip");

        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = ZipWriter::new(file);
            let opts = FileOptions::default();
            zip.start_file::<_, ()>("../evil.txt", opts).unwrap();
            zip.write_all(b"malice").unwrap();
            zip.finish().unwrap();
        }

        extract_zip(work.path(), &zip_path).unwrap();

        assert!(
            !work.path().join("evil.txt").exists(),
            "entry should have been discarded"
        );
        assert!(
            !work.path().parent().unwrap().join("evil.txt").exists(),
            "must not escape working_dir"
        );
    }
}
