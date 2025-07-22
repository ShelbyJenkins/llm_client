use cmdstruct::Arg;
use serde::{Deserialize, Serialize};

use crate::{
    error::{LmcppError, LmcppResult},
    server::types::file::ValidFile,
};
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct LocalModelPath(pub ValidFile);

impl LocalModelPath {
    pub fn new(path: impl Into<std::path::PathBuf>) -> LmcppResult<Self> {
        Ok(Self(ValidFile::new(path.into())?))
    }
}

impl Arg for LocalModelPath {
    fn append_arg(&self, cmd: &mut std::process::Command) {
        cmd.arg(&self.0 .0);
    }
}

impl core::fmt::Display for LocalModelPath {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

// Wrap the `url::Url` type in a new‑type so we can attach custom trait impls.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ModelUrl(pub url::Url);

impl core::fmt::Display for ModelUrl {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0.as_str())
    }
}

impl Arg for ModelUrl {
    fn append_arg(&self, cmd: &mut std::process::Command) {
        cmd.arg(self.0.as_str());
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct HfRepo(pub String);

impl Arg for HfRepo {
    fn append_arg(&self, cmd: &mut std::process::Command) {
        cmd.arg(&self.0); // builder already supplies `-hf`
    }
}

impl std::fmt::Display for HfRepo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// <user>/<model>[:quant] (quant ≥ 2 chars, ASCII [A-Z a-z 0-9 _ -])
impl TryFrom<&str> for HfRepo {
    type Error = LmcppError;

    fn try_from(value: &str) -> LmcppResult<Self> {
        let (user_model, _quant) = match value.split_once(':') {
            Some((p, q)) => {
                if q.len() < 2 {
                    return Err(LmcppError::InvalidConfig {
                        field: "Hugging Face repo",
                        reason: "quant suffix must be at least 2 characters".into(),
                    });
                }
                (p, Some(q))
            }
            None => (value, None),
        };

        let (user, model) = user_model.split_once('/').ok_or_else(|| {
            return LmcppError::InvalidConfig {
                field: "Hugging Face repo",
                reason: "expected `<user>/<model>`".into(),
            };
        })?;

        let ok = |s: &str| {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        };

        if !ok(user) || !ok(model) {
            return Err(LmcppError::InvalidConfig {
                field: "Hugging Face repo",
                reason: "user and model names must be non-empty and contain only alphanumeric characters, underscores, or hyphens".into(),
            });
        }
        Ok(HfRepo(value.to_owned()))
    }
}

impl TryFrom<String> for HfRepo {
    type Error = LmcppError;
    fn try_from(v: String) -> LmcppResult<Self> {
        HfRepo::try_from(v.as_str()).map(|_| HfRepo(v))
    }
}

/// A concrete file inside a Hugging Face repo (e.g. `MyModel.Q4_K.gguf`).
///
/// Validation rules:
/// * non‑empty
/// * no path separators (‘/’ or ‘\’) – it should be just a file name
/// * must end with the literal `.gguf`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct HfFile(pub String);

impl Arg for HfFile {
    fn append_arg(&self, cmd: &mut std::process::Command) {
        cmd.arg(&self.0);
    }
}

impl std::fmt::Display for HfFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<&str> for HfFile {
    type Error = LmcppError;

    fn try_from(value: &str) -> LmcppResult<Self> {
        if value.is_empty() {
            return Err(LmcppError::InvalidConfig {
                field: "Hugging Face file",
                reason: "file name cannot be empty".into(),
            });
        }
        if !value.ends_with(".gguf") {
            return Err(LmcppError::InvalidConfig {
                field: "Hugging Face file",
                reason: "file name must end with .gguf".into(),
            });
        }
        if value.contains('/') || value.contains('\\') {
            return Err(LmcppError::InvalidConfig {
                field: "Hugging Face file",
                reason: "file name must not contain path separators".into(),
            });
        }

        Ok(HfFile(value.to_owned()))
    }
}

impl TryFrom<String> for HfFile {
    type Error = LmcppError;
    fn try_from(v: String) -> LmcppResult<Self> {
        HfFile::try_from(v.as_str()).map(|_| HfFile(v))
    }
}
