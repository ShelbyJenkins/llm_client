use derive_more::*;
use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HfRepoId {
    repo_id: String,
    name_space: HfNameSpace,
    repo_name: HfRepoName,
    sha: Option<HfRepoSha>,
}

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum HfRepoIdError {
    #[error("Invalid input '{repo_id}': missing namespace")]
    MissingNamespace { repo_id: String },

    #[error(transparent)]
    InvalidNameSpace(#[from] HfNameSpaceError),

    #[error("Invalid repository ID '{repo_id}': too many path segments")]
    TooManySegments { repo_id: String },

    #[error("Invalid input '{repo_id}': missing repository name")]
    MissingRepoName { repo_id: String },

    #[error(transparent)]
    InvalidRepoName(#[from] HfRepoNameError),

    #[error(transparent)]
    InvalidSha(#[from] HfRepoShaError),
}

impl HfRepoId {
    pub fn try_from_repo_id<Id, Rev>(repo_id: Id, sha: Option<Rev>) -> Result<Self, HfRepoIdError>
    where
        Id: AsRef<str>,
        Rev: AsRef<str>,
    {
        let raw = repo_id
            .as_ref()
            .trim()
            .trim_end_matches('/')
            .trim_start_matches('/');

        let mut parts = raw.split('/');

        let ns = parts
            .next()
            .ok_or_else(|| HfRepoIdError::MissingNamespace {
                repo_id: raw.to_owned(),
            })?;

        let rn = parts.next().ok_or_else(|| HfRepoIdError::MissingRepoName {
            repo_id: raw.to_owned(),
        })?;

        if parts.next().is_some() {
            return Err(HfRepoIdError::TooManySegments {
                repo_id: raw.to_owned(),
            });
        }

        Self::try_from_namespace_and_repo_name(ns, rn, sha)
    }

    /// Build a repo from **already‑separated** namespace + repo parts.
    /// Both inputs are validated; errors mirror `from_repo_id`.
    pub fn try_from_namespace_and_repo_name<NS, RN, RV>(
        ns: NS,
        rn: RN,
        sha: Option<RV>,
    ) -> Result<Self, HfRepoIdError>
    where
        NS: AsRef<str>,
        RN: AsRef<str>,
        RV: AsRef<str>,
    {
        let name_space = HfNameSpace::try_new(ns.as_ref())?;
        let repo_name = HfRepoName::try_new(rn.as_ref())?;

        Ok(Self {
            repo_id: format!("{}/{}", name_space.as_ref(), repo_name.as_ref()),
            name_space,
            repo_name,
            sha: sha.map(|r| HfRepoSha::try_new(r.as_ref())).transpose()?,
        })
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn name_space(&self) -> &str {
        self.name_space.as_ref()
    }

    pub fn repo_name(&self) -> &str {
        self.repo_name.as_ref()
    }

    pub fn sha(&self) -> Option<&str> {
        self.sha.as_ref().map(|x| x.as_str())
    }

    pub fn set_sha(&mut self, sha: String) -> Result<(), HfRepoShaError> {
        self.sha = Some(HfRepoSha::try_new(sha)?);
        Ok(())
    }

    pub fn as_url(&self) -> Url {
        Url::parse(&format!(
            "https://huggingface.co/{}/{}",
            self.name_space.as_ref(),
            self.repo_name.as_ref()
        ))
        .expect("static URL cannot fail")
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Deref, Display, Into,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct HfNameSpace(String);

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum HfNameSpaceError {
    #[error(
        "Invalid name space '{name_space}': length must be 2-64 characters (after trim); got {len}"
    )]
    Length { name_space: String, len: usize },

    #[error("Invalid name space '{name_space}': cannot be all digits")]
    AllDigits { name_space: String },

    #[error("Invalid name space '{name_space}': cannot start with a hyphen")]
    LeadingHyphen { name_space: String },

    #[error("Invalid name space '{name_space}': cannot end with a hyphen")]
    TrailingHyphen { name_space: String },

    #[error("Invalid name space '{name_space}': found consecutive hyphens")]
    ConsecutiveHyphens { name_space: String },

    #[error("Invalid name space '{name_space}': contains invalid characters {invalid:?}")]
    InvalidChars {
        name_space: String,
        invalid: Vec<(usize, char)>,
    },
}

impl HfNameSpace {
    pub fn try_new<S: AsRef<str>>(raw: S) -> Result<Self, HfNameSpaceError> {
        let s = raw.as_ref().trim();
        let name_space = s.to_owned();

        let len = s.len();
        if len < 2 || len > 64 {
            return Err(HfNameSpaceError::Length { name_space, len });
        }

        if s.chars().all(|c| c.is_ascii_digit()) {
            return Err(HfNameSpaceError::AllDigits { name_space });
        }

        if s.starts_with('-') {
            return Err(HfNameSpaceError::LeadingHyphen { name_space });
        }
        if s.ends_with('-') {
            return Err(HfNameSpaceError::TrailingHyphen { name_space });
        }

        if s.as_bytes().windows(2).any(|w| w == b"--") {
            return Err(HfNameSpaceError::ConsecutiveHyphens { name_space });
        }

        let invalid: Vec<_> = s
            .char_indices()
            .filter(|(_, ch)| !(ch.is_ascii_alphanumeric() || *ch == '-'))
            .collect();

        if !invalid.is_empty() {
            return Err(HfNameSpaceError::InvalidChars {
                name_space,
                invalid,
            });
        }

        Ok(Self(name_space))
    }

    pub fn as_url(&self) -> Url {
        Url::parse(&format!("https://huggingface.co/{}", self.as_ref()))
            .expect("static URL cannot fail")
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Deref, Display, Into,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct HfRepoName(String);

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum HfRepoNameError {
    #[error(
        "Invalid repository name '{repo_name}': length must be 1-96 characters (after trim); got {len}"
    )]
    Length { repo_name: String, len: usize },

    #[error(
        "Invalid repository name '{repo_name}': first character must be alphanumeric; got '{ch}'"
    )]
    FirstChar { repo_name: String, ch: char },

    #[error(
        "Invalid repository name '{repo_name}': trailing character must not be '.', '_' or '-'"
    )]
    TrailingSeparator { repo_name: String },

    #[error("Invalid repository name '{repo_name}': contains invalid characters {invalid:?}")]
    InvalidChars {
        repo_name: String,
        invalid: Vec<(usize, char)>,
    },
}

impl HfRepoName {
    pub fn try_new<S: AsRef<str>>(raw: S) -> Result<Self, HfRepoNameError> {
        let s = raw.as_ref().trim();
        let repo_name = s.to_owned();

        let len = s.len();
        if len == 0 || len > 96 {
            return Err(HfRepoNameError::Length { repo_name, len });
        }

        let first = s.chars().next().unwrap();
        if !first.is_ascii_alphanumeric() {
            return Err(HfRepoNameError::FirstChar {
                repo_name,
                ch: first,
            });
        }

        let last = s.chars().last().unwrap();
        if matches!(last, '.' | '_' | '-') {
            return Err(HfRepoNameError::TrailingSeparator { repo_name });
        }

        let invalid: Vec<_> = s
            .char_indices()
            .filter(|(_, ch)| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-')))
            .collect();

        if !invalid.is_empty() {
            return Err(HfRepoNameError::InvalidChars { repo_name, invalid });
        }

        Ok(Self(repo_name))
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Deref, Display, Into,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct HfRepoSha(String);

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum HfRepoShaError {
    #[error("SHA is empty")]
    Empty,
    #[error("SHA '{sha}': length must be 7-40 characters; got {len}")]
    Length { sha: String, len: usize },
    #[error("SHA '{sha}': must be lowercase hex")]
    InvalidHex { sha: String },
}

impl HfRepoSha {
    pub fn try_new<S: AsRef<str>>(raw: S) -> Result<Self, HfRepoShaError> {
        let s = raw.as_ref();
        if s.is_empty() {
            return Err(HfRepoShaError::Empty);
        }
        let len = s.len();
        if !(7..=40).contains(&len) {
            return Err(HfRepoShaError::Length { sha: s.into(), len });
        }
        if s.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
            Ok(Self(s.to_owned()))
        } else {
            Err(HfRepoShaError::InvalidHex { sha: s.into() })
        }
    }
}

/// A validated, UTF-8, repo-relative path.
///
/// * Never empty  
/// * No leading slash (must be relative)  
/// * No parent traversal (`..`) or `.` segments  
/// * No double slashes (`//`)  
/// * Each segment ≤ 255 bytes  
/// * Allowed chars: ASCII letters, digits, `_`, `-`, `.`, or any non-ASCII UTF-8 code-point
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Deref, Display, Into,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct HfRFilename(String);

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum HfRFilenameError {
    #[error("path is empty")]
    Empty,
    #[error("Invalid rfilename `{rfilename}`: path must be relative (no leading '/')")]
    Absolute { rfilename: String },
    #[error("Invalid rfilename `{rfilename}`: path contains empty segment (\"//\")")]
    EmptySegment { rfilename: String },
    #[error(
        "Invalid rfilename `{rfilename}`: path contains parent traversal ('..') or current dir ('.')"
    )]
    ParentTraversal { rfilename: String },
    #[error("Invalid rfilename `{rfilename}`: segment '{segment}' exceeds 255 bytes")]
    SegmentTooLong { rfilename: String, segment: String },
    #[error("Invalid rfilename `{rfilename}`: contains invalid characters {invalid:?}")]
    InvalidChars {
        rfilename: String,
        invalid: Vec<(usize, char)>,
    },
}

impl HfRFilename {
    pub fn try_new<S: AsRef<str>>(raw: S) -> Result<Self, HfRFilenameError> {
        let rfilename = raw.as_ref().to_owned();
        if rfilename.is_empty() {
            return Err(HfRFilenameError::Empty);
        }
        if rfilename.starts_with('/') {
            return Err(HfRFilenameError::Absolute { rfilename });
        }
        if rfilename.contains("//") {
            return Err(HfRFilenameError::EmptySegment { rfilename });
        }

        let mut seg_start = 0;
        let mut invalid: Vec<(usize, char)> = Vec::new();
        for (idx, ch) in rfilename.char_indices() {
            if ch == '/' {
                if idx - seg_start > 255 {
                    return Err(HfRFilenameError::SegmentTooLong {
                        segment: rfilename[seg_start..idx].into(),
                        rfilename,
                    });
                }
                seg_start = idx + 1;
                continue;
            }
            let ok = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') || !ch.is_ascii();
            if !ok {
                invalid.push((idx, ch));
            }
        }

        if rfilename.len() - seg_start > 255 {
            return Err(HfRFilenameError::SegmentTooLong {
                segment: rfilename[seg_start..].into(),
                rfilename,
            });
        }

        if !invalid.is_empty() {
            return Err(HfRFilenameError::InvalidChars { rfilename, invalid });
        }

        // Fast traversal check (no "." or "..")
        if rfilename.split('/').any(|seg| seg == "." || seg == "..") {
            return Err(HfRFilenameError::ParentTraversal { rfilename });
        }

        Ok(Self(rfilename))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Into)]
pub struct HfFile {
    repo_id: HfRepoId,
    rfilename: HfRFilename,
    extension: Option<String>,
}

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum HfFileError {
    #[error("Invalid URL '{url}': {e}")]
    InvalidUrl { url: String, e: String },

    #[error("Invalid URL '{url}':  scheme must be http/https")]
    InvalidScheme { url: String },

    #[error("Invalid domain '{url}': domain must be huggingface.co")]
    InvalidDomain { url: String },

    #[error("Invalid URL '{url}': missing path segments")]
    MissingPathSegments { url: String },

    #[error("Invalid URL '{raw}': missing namespace")]
    MissingNamespace { raw: String },

    #[error("Invalid URL '{raw}': missing repository name")]
    MissingRepoName { raw: String },

    #[error("Invalid URL '{raw}': missing repository file name")]
    MissingRFilename { raw: String },

    #[error("Unexpected extension: expected `{expected}`, found `{found}`")]
    UnexpectedExtension { expected: String, found: String },

    #[error(transparent)]
    HfRepoId(#[from] HfRepoIdError),

    #[error(transparent)]
    HfRFilename(#[from] HfRFilenameError),
}

impl HfFile {
    pub fn try_from_url<U: AsRef<str>>(url: U) -> Result<Self, HfFileError> {
        let raw = url.as_ref();
        let url = Url::parse(raw).map_err(|e| HfFileError::InvalidUrl {
            url: raw.into(),
            e: e.to_string(),
        })?;

        match url.scheme() {
            "http" | "https" => {}
            _ => return Err(HfFileError::InvalidScheme { url: raw.into() }),
        }

        if url.domain() != Some("huggingface.co") {
            return Err(HfFileError::InvalidDomain { url: raw.into() });
        }

        let mut segs = url
            .path_segments()
            .ok_or_else(|| HfFileError::MissingPathSegments { url: raw.into() })?
            .filter(|s| !s.is_empty());

        let name_space: &str = segs
            .next()
            .ok_or_else(|| HfFileError::MissingNamespace { raw: raw.into() })?;

        let repo_name = segs.next().ok_or_else(|| HfFileError::MissingRepoName {
            raw: url.as_str().to_owned(),
        })?;

        let mut rest: Vec<&str> = segs.collect();
        let sha = match rest.first().copied() {
            Some("resolve" | "blob" | "raw") => {
                if rest.len() < 3 {
                    return Err(HfFileError::MissingRFilename { raw: raw.into() });
                }
                rest.remove(0); // drop "resolve" (or blob/raw)
                Some(rest.remove(0).to_string()) // take the <sha>
            }
            _ => None,
        };

        let repo_id = HfRepoId::try_from_namespace_and_repo_name::<&str, &str, String>(
            name_space, repo_name, sha,
        )?;

        if rest.is_empty() {
            return Err(HfFileError::MissingRFilename { raw: raw.into() });
        }

        let rfilename = HfRFilename::try_new(rest.join("/"))?;

        let extension = std::path::Path::new(rfilename.as_ref())
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase());

        Ok(Self {
            repo_id,
            rfilename,
            extension,
        })
    }

    pub fn try_from_repo_id_and_file(
        repo_id: &HfRepoId,
        rfilename: HfRFilename,
    ) -> Result<Self, HfFileError> {
        let extension = std::path::Path::new(rfilename.as_ref())
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase());

        Ok(Self {
            repo_id: repo_id.to_owned(),
            rfilename,
            extension,
        })
    }

    pub fn is_extension(&self, ext: &str) -> Result<(), HfFileError> {
        // strip a leading '.' if present and lowercase both sides
        let want = ext.trim_start_matches('.').to_ascii_lowercase();

        match self.extension.as_deref() {
            Some(found) if found.eq_ignore_ascii_case(&want) => Ok(()),
            Some(found) => Err(HfFileError::UnexpectedExtension {
                expected: format!(".{want}"),
                found: found.into(),
            }),
            None => Err(HfFileError::UnexpectedExtension {
                expected: format!(".{want}"),
                found: "none".into(),
            }),
        }
    }
}
