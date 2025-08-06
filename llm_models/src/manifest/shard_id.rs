//! Shard‑tag parsing utilities.
//!
//! Many multi‑part model formats append a fragment like  
//! `"-00002-of-00008"` to each shard’s filename (e.g.  
//! `"model-00002-of-00008.gguf"`).  This module provides:
//! * [`ShardId`] — an efficient 1‑based `(index, total)` pair.
//! * [`ShardId::from_fname`] — infers a `ShardId` from a file name
//!   or returns sensible defaults / precise error information.

use std::num::NonZeroU32;

#[derive(Debug, thiserror::Error)]
pub enum ShardIdError {
    /// Empty string was provided instead of a file name.
    #[error("file name is empty")]
    FnameEmpty,

    /// The regex matched but the `idx` capture was unexpectedly absent.
    #[error("file '{fname}': shard index capture group was absent (bug?)")]
    MissingIdxGroup { fname: String },

    /// The regex matched but the `total` capture was unexpectedly absent.
    #[error("file '{fname}': shard total capture group was absent (bug?)")]
    MissingTotalGroup { fname: String },

    /// The `idx` capture did not parse as a positive non‑zero `u32`.
    #[error("file '{fname}': index part \"{part}\" is not a positive integer")]
    BadIndex {
        fname: String,
        part: String,
        #[source]
        source: std::num::ParseIntError,
    },

    /// The `total` capture did not parse as a positive non‑zero `u32`.
    #[error("file '{fname}': total part \"{part}\" is not a positive integer")]
    BadTotal {
        fname: String,
        part: String,
        #[source]
        source: std::num::ParseIntError,
    },

    /// Parsed `index` exceeded the parsed `total`.
    #[error("file '{fname}': index ({index}) cannot exceed total ({total})")]
    IndexExceedsTotal {
        fname: String,
        index: NonZeroU32,
        total: NonZeroU32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// In‑memory representation of a shard tag.
///
/// The numbers are **1‑based** and satisfy `index ≤ total`.
pub struct ShardId {
    /// Position of this shard (starting at 1).
    index: NonZeroU32,
    /// Total number of shards making up the model.
    total: NonZeroU32,

    string: String, // Added to store the string representation of the shard
}

impl Default for ShardId {
    /// Returns the sentinel “not‑sharded” value `(1 of 1)`.
    fn default() -> Self {
        Self {
            index: NonZeroU32::new(1).expect("default index must be non-zero"),
            total: NonZeroU32::new(1).expect("default total must be non-zero"),
            string: String::from("1-of-1"),
        }
    }
}

impl ShardId {
    /// Extracts a shard tag from `fname`.
    ///
    /// * If the pattern is **absent**, returns [`Ok`]`(ShardId::default())`.
    /// * If the pattern is **malformed**, returns a variant of
    ///   [`ShardIdError`] describing the exact failure.
    pub fn from_fname(fname: &str) -> Result<Self, ShardIdError> {
        if fname.is_empty() {
            return Err(ShardIdError::FnameEmpty);
        }

        static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re = RE.get_or_init(|| {
            regex::Regex::new(
                r#"(?x)                     # Activate free-spacing / comments
        -                           # literal '-' before the shard index
        (?P<idx>                    # ┌─ named capture: idx  (shard number)
            0*                      # │ any leading zeros
            [0-9]                   # │ first non-zero digit (disallows 0)
            [0-9]*                  # │ remaining digits
        )                           # └─ end idx
        -of-                        # literal separator "-of-"
        (?P<total>                  # ┌─ named capture: total (shard count)
            0*                      # │ leading zeros allowed
            [0-9]                   # │ first non-zero digit
            [0-9]*                  # │ remaining digits
        )                           # └─ end total
        (?:                         # ┌─ OPTIONAL extension(s), e.g. ".gguf"
            \.[A-Za-z0-9]+          # │ one ".ext" segment
            (?:\.[A-Za-z0-9]+)*     # │ …followed by zero or more segments
        )?                          # └─ whole extension group is optional
        $                           # anchor: whole string must now end
        "#,
            )
            .expect("static regex must compile")
        });

        let caps = match re.captures(fname) {
            Some(caps) => caps,
            None => return Ok(Self::default()),
        };

        let idx_str = caps
            .name("idx")
            .ok_or_else(|| ShardIdError::MissingIdxGroup {
                fname: fname.into(),
            })?
            .as_str();
        let total_str = caps
            .name("total")
            .ok_or_else(|| ShardIdError::MissingTotalGroup {
                fname: fname.into(),
            })?
            .as_str();

        let index: NonZeroU32 = idx_str.parse().map_err(|e| ShardIdError::BadIndex {
            fname: fname.into(),
            part: idx_str.into(),
            source: e,
        })?;

        let total: NonZeroU32 = total_str.parse().map_err(|e| ShardIdError::BadTotal {
            fname: fname.into(),
            part: total_str.into(),
            source: e,
        })?;

        if index > total {
            return Err(ShardIdError::IndexExceedsTotal {
                fname: fname.into(),
                index,
                total,
            });
        }

        Ok(Self {
            index,
            total,
            // keep the user’s original padding (without the leading dash)
            string: format!("{}-of-{}", idx_str, total_str),
        })
    }

    pub fn is_single(&self) -> bool {
        self.total.get() == 1
    }

    pub fn index(&self) -> NonZeroU32 {
        self.index
    }

    pub fn total(&self) -> NonZeroU32 {
        self.total
    }

    pub fn as_str(&self) -> &str {
        &self.string
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gguf, idx, total
    const GGUF_FILES: &[(&str, u8, u8)] = &[
        ("llama-7b.Q4_K_M-3-of-10", 3, 10),
        ("llama-7b.Q4_K_M-3-of-10.gguf", 3, 10),
        ("codellama-7b-3-of-10", 3, 10),
        ("mistral-7b-instruct-v0.2.Q4_K_M-3-of-10", 3, 10),
        ("SmolLM-1.7B-Instruct-v0.2-Q5_K_M-3-of-10", 3, 10),
        (
            "meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            1,
            2,
        ),
    ];

    #[test]
    fn parses_gguf_shard() {
        for (fname, idx, total) in GGUF_FILES {
            let id = ShardId::from_fname(fname).unwrap();
            assert_eq!(id.index.get(), *idx as u32);
            assert_eq!(id.total.get(), *total as u32);
        }
    }

    #[test]
    fn parses_safetensors_shard() {
        let id = ShardId::from_fname("model-00002-of-00008.safetensors").unwrap();
        assert_eq!(id.index.get(), 2);
        assert_eq!(id.total.get(), 8);
        let id = ShardId::from_fname("model-00002-of-00008").unwrap();
        assert_eq!(id.index.get(), 2);
        assert_eq!(id.total.get(), 8);
    }

    #[test]
    fn parses_multi_extension_path() {
        let id = ShardId::from_fname("some.dir/my.model-42-of-100.ggml.bin").unwrap();
        assert_eq!(id.index.get(), 42);
        assert_eq!(id.total.get(), 100);
    }

    #[test]
    fn returns_default_when_no_shard_tag() {
        let id = ShardId::from_fname("model.gguf").unwrap();
        assert_eq!(id, ShardId::default());
    }

    #[test]
    fn err_empty_filename() {
        let err = ShardId::from_fname("").unwrap_err();
        assert!(matches!(err, ShardIdError::FnameEmpty));
    }

    #[test]
    fn err_bad_index_overflow() {
        // 43_000_000_000 > u32::MAX (4_294_967_295)
        let err = ShardId::from_fname("model-43000000000-of-10.gguf").unwrap_err();
        assert!(matches!(err, ShardIdError::BadIndex { .. }));
    }

    #[test]
    fn err_bad_index_of_zero() {
        let err = ShardId::from_fname("model-0-of-10.gguf").unwrap_err();
        assert!(matches!(err, ShardIdError::BadIndex { .. }));
    }

    #[test]
    fn err_bad_total_overflow() {
        let err = ShardId::from_fname("model-1-of-43000000000.gguf").unwrap_err();
        assert!(matches!(err, ShardIdError::BadTotal { .. }));
    }

    #[test]
    fn err_bad_total_of_zero() {
        let err = ShardId::from_fname("model-1-of-0.gguf").unwrap_err();
        assert!(matches!(err, ShardIdError::BadTotal { .. }));
    }

    #[test]
    fn err_index_exceeds_total() {
        let err = ShardId::from_fname("model-10-of-5.gguf").unwrap_err();
        match err {
            ShardIdError::IndexExceedsTotal { index, total, .. } => {
                assert_eq!(index.get(), 10);
                assert_eq!(total.get(), 5);
            }
            _ => panic!("expected IndexExceedsTotal"),
        }
    }

    // “should‑never‑happen” variants
    #[test]
    fn err_missing_idx_group_display() {
        let err = ShardIdError::MissingIdxGroup {
            fname: "foo".into(),
        };
        assert_eq!(
            format!("{err}"),
            "file 'foo': shard index capture group was absent (bug?)"
        );
    }

    #[test]
    fn err_missing_total_group_display() {
        let err = ShardIdError::MissingTotalGroup {
            fname: "bar".into(),
        };
        assert_eq!(
            format!("{err}"),
            "file 'bar': shard total capture group was absent (bug?)"
        );
    }
}
