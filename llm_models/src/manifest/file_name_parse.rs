use std::{path::Path, result::Result};

use crate::manifest::{
    file_encoding_type::GgmlFileType,
    shard_id::{ShardId, ShardIdError},
};

#[derive(Debug, thiserror::Error)]
pub enum FileNameParseError {
    #[error("file name is empty")]
    FnameEmpty,

    #[error("file name `{fname}` does not have an extension")]
    MissingExtension { fname: String },

    #[error("file name `{fname}`  has an incorrect extension, expected `{wanted}`")]
    IncorrectExtension { fname: String, wanted: &'static str },

    #[error("missing encoding in `{fname}`: {e}")]
    MissingEncoding { fname: String, e: String },

    #[error(transparent)]
    ShardId(#[from] ShardIdError),

    #[error(
        "file name `{fname}` resulted in an invalid model name `{model_name}` or base name `{base_name}`"
    )]
    InvalidNameOutput {
        fname: String,
        model_name: String,
        base_name: String,
    },
}

/// See the regex defined here https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#validating-above-naming-convention
#[derive(Debug)]
pub struct GgufFileNameParse {
    pub base_name: String,
    pub model_name: String,
    pub file_type: GgmlFileType,
    pub shard_id: ShardId,
}

impl GgufFileNameParse {
    pub fn from_fname(fname: &str) -> Result<Self, FileNameParseError> {
        if fname.is_empty() {
            return Err(FileNameParseError::FnameEmpty);
        }

        let path = Path::new(fname);

        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .ok_or(FileNameParseError::MissingExtension {
                fname: fname.to_string(),
            })?;

        if ext != "gguf" {
            return Err(FileNameParseError::IncorrectExtension {
                fname: fname.to_string(),
                wanted: "gguf",
            });
        }

        let stem = path.file_stem().and_then(|s| s.to_str()).ok_or(
            FileNameParseError::MissingExtension {
                fname: fname.to_string(),
            },
        )?;

        let shard_id = ShardId::from_fname(stem)?;
        let shard_tag = format!("-{}", &shard_id.as_str());
        let model_name = stem.strip_suffix(&shard_tag).unwrap_or(stem);

        let file_type =
            GgmlFileType::try_from(stem).map_err(|e| FileNameParseError::MissingEncoding {
                fname: stem.to_string(),
                e: e.to_string(),
            })?;

        let pat = {
            let esc = regex::escape(file_type.as_str());
            format!(r"(?i)([\.-]){}((?:-\d{{5}}-of-\d{{5}})?|-|$)", esc)
        };
        let re = regex::Regex::new(&pat).expect("dynamic regex compiles");
        let base_name = re.replace(&model_name, "").to_string();

        if base_name.is_empty()
            || model_name.is_empty()
            || base_name == model_name
            || !model_name.contains(base_name.as_str())
        {
            return Err(FileNameParseError::InvalidNameOutput {
                fname: stem.to_string(),
                model_name: model_name.to_string(),
                base_name: base_name.to_string(),
            });
        }

        Ok(Self {
            base_name: base_name.to_string(),
            model_name: model_name.to_string(),
            file_type,
            shard_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ──────────────────────────────────────────────────────────────────────────
    // Common fixtures
    // ──────────────────────────────────────────────────────────────────────────
    //
    //  tuple layout:
    //      0  → file name
    //      1  → expected base_name
    //      2  → expected model_name
    //      3  → expected GgmlFileType
    //      4  → expected shard string (omit "" ⇢ defaults to "1-of-1")
    const GGUF_FILES: &[(&str, &str, &str, GgmlFileType, &str)] = &[
        (
            "llama-7b.Q4_K_M.gguf",
            "llama-7b",
            "llama-7b.Q4_K_M",
            GgmlFileType::Q4_K_M,
            "",
        ),
        (
            "mistral-7b-v0.1.Q4_K_M.gguf",
            "mistral-7b-v0.1",
            "mistral-7b-v0.1.Q4_K_M",
            GgmlFileType::Q4_K_M,
            "",
        ),
        (
            "codellama-7b.Q3_K_M.gguf",
            "codellama-7b",
            "codellama-7b.Q3_K_M",
            GgmlFileType::Q3_K_M,
            "",
        ),
        (
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "mistral-7b-instruct-v0.2",
            "mistral-7b-instruct-v0.2.Q4_K_M",
            GgmlFileType::Q4_K_M,
            "",
        ),
        (
            "SmolLM-1.7B-Instruct-v0.2-Q5_K_M.gguf",
            "SmolLM-1.7B-Instruct-v0.2",
            "SmolLM-1.7B-Instruct-v0.2-Q5_K_M",
            GgmlFileType::Q5_K_M,
            "",
        ),
        (
            "Mixtral-8x22B-Instruct-v0.1.Q4_K_M-00002-of-00002.gguf",
            "Mixtral-8x22B-Instruct-v0.1",
            "Mixtral-8x22B-Instruct-v0.1.Q4_K_M",
            GgmlFileType::Q4_K_M,
            "00002-of-00002",
        ),
        (
            "meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            "meta-llama_Llama-4-Scout-17B-16E-Instruct",
            "meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M",
            GgmlFileType::Q4_K_M,
            "00001-of-00002",
        ),
    ];

    // ──────────────────────────────────────────────────────────────────────────
    // Happy-path parsing
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn parses_known_good_gguf_files() {
        for &(file, base, model, ftype, shard) in GGUF_FILES {
            let parsed = GgufFileNameParse::from_fname(file).unwrap();
            assert_eq!(parsed.base_name, base);
            assert_eq!(parsed.model_name, model);
            assert_eq!(parsed.file_type, ftype);
            assert_eq!(
                parsed.shard_id.as_str(),
                if shard.is_empty() { "1-of-1" } else { shard }
            );
            assert_eq!(parsed.shard_id.is_single(), shard.is_empty());
        }
    }

    #[test]
    fn parses_minimal_gguf() {
        let g = GgufFileNameParse::from_fname("llama-7B-Q4_K_M.gguf").unwrap();
        assert_eq!(g.base_name, "llama-7B");
        assert_eq!(g.model_name, "llama-7B-Q4_K_M");
        assert_eq!(g.file_type, GgmlFileType::Q4_K_M);
        assert!(g.shard_id.is_single());
    }

    #[test]
    fn parses_largest_plausible_gguf() {
        let fname = "mistral-7B-chat-v2-f16-00001-of-00008.gguf";
        let g = GgufFileNameParse::from_fname(fname).unwrap();
        assert_eq!(g.base_name, "mistral-7B-chat-v2");
        assert_eq!(g.model_name, "mistral-7B-chat-v2-f16");
        assert_eq!(g.file_type, GgmlFileType::F16);
        assert_eq!(g.shard_id.as_str(), "00001-of-00008");
    }

    #[test]
    fn parses_case_insensitive_extension() {
        let g = GgufFileNameParse::from_fname("MODEL.Q4_K_M.GGUF").unwrap();
        assert_eq!(g.file_type, GgmlFileType::Q4_K_M);
    }

    #[test]
    fn parses_case_insensitive_quant_tag() {
        let g = GgufFileNameParse::from_fname("Llama-7B.q4_k_m.GGUF").unwrap();
        assert_eq!(g.file_type, GgmlFileType::Q4_K_M);
    }

    #[test]
    fn parses_without_shard_tag() {
        let g = GgufFileNameParse::from_fname("mistral-7b-f16.gguf").unwrap();
        assert_eq!(g.base_name, "mistral-7b");
        assert_eq!(g.model_name, "mistral-7b-f16");
        assert_eq!(g.file_type, GgmlFileType::F16);
        assert!(g.shard_id.is_single());
    }

    #[test]
    fn parses_with_directory_components() {
        let path = "models/sub/llama-7b.Q4_K_M.gguf";
        let bare = "llama-7b.Q4_K_M.gguf";
        let p1 = GgufFileNameParse::from_fname(path).unwrap();
        let p2 = GgufFileNameParse::from_fname(bare).unwrap();
        assert_eq!(p1.base_name, p2.base_name);
        assert_eq!(p1.model_name, p2.model_name);
        assert_eq!(p1.file_type, p2.file_type);
        assert_eq!(p1.shard_id, p2.shard_id);
    }

    #[test]
    fn parses_unicode_filename() {
        let g = GgufFileNameParse::from_fname("ŚmolLM-1.7B-Q5_K_M.gguf").unwrap();
        assert_eq!(g.base_name, "ŚmolLM-1.7B");
        assert_eq!(g.file_type, GgmlFileType::Q5_K_M);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Error paths
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn error_empty_filename() {
        assert!(matches!(
            GgufFileNameParse::from_fname(""),
            Err(FileNameParseError::FnameEmpty)
        ));
    }

    #[test]
    fn error_missing_extension() {
        assert!(matches!(
            GgufFileNameParse::from_fname("llama"),
            Err(FileNameParseError::MissingExtension { .. })
        ));
    }

    #[test]
    fn error_incorrect_extension() {
        for bad in ["model.gguf2", "model.txt", "model.GGUF.backup"] {
            assert!(matches!(
                GgufFileNameParse::from_fname(bad),
                Err(FileNameParseError::IncorrectExtension { wanted: "gguf", .. })
            ));
        }
    }

    #[test]
    fn error_missing_encoding_tag() {
        for bad in ["llama-7b.gguf", "llama-7b-FOO.gguf"] {
            assert!(matches!(
                GgufFileNameParse::from_fname(bad),
                Err(FileNameParseError::MissingEncoding { .. })
            ));
        }
    }

    #[test]
    fn error_invalid_name_output() {
        for bad in ["Q4_K_M.gguf", "Q4_K_M-00001-of-00002.gguf"] {
            assert!(matches!(
                GgufFileNameParse::from_fname(bad),
                Err(FileNameParseError::InvalidNameOutput { .. })
            ));
        }
    }

    #[test]
    fn shard_error_bad_index_zero() {
        assert!(matches!(
            GgufFileNameParse::from_fname("model-00000-of-00008.gguf"),
            Err(FileNameParseError::ShardId(ShardIdError::BadIndex { .. }))
        ));
    }

    #[test]
    fn shard_error_index_exceeds_total() {
        assert!(matches!(
            GgufFileNameParse::from_fname("model-00009-of-00008.gguf"),
            Err(FileNameParseError::ShardId(
                ShardIdError::IndexExceedsTotal { .. }
            ))
        ));
    }

    #[test]
    fn pattern_mismatch_dash_required() {
        assert!(matches!(
            GgufFileNameParse::from_fname("llama7B.gguf"),
            Err(FileNameParseError::MissingEncoding { .. })
        ));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Tests migrated from `file_type_encoding.rs`
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn parses_basic_quant_tags() {
        let f = GgufFileNameParse::from_fname("Foo-Q4_0.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q4_0);

        let f = GgufFileNameParse::from_fname("bar-q5_K_M.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q5_K_M);

        let f = GgufFileNameParse::from_fname("Meta-Llama-F32.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::AllF32);
    }

    #[test]
    fn mixed_case_and_extra_tokens() {
        let f = GgufFileNameParse::from_fname("Llama-Instruct-q8_0.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q8_0);

        let f = GgufFileNameParse::from_fname("Llama-3-Q2_k_s.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q2_K_S);
    }

    #[test]
    fn rejects_unknown_or_bad_extension() {
        // wrong suffix
        assert!(matches!(
            GgufFileNameParse::from_fname("foo-Q4_0.bin"),
            Err(FileNameParseError::IncorrectExtension { .. })
        ));
        // unknown tag
        assert!(matches!(
            GgufFileNameParse::from_fname("foo-Q9_X.gguf"),
            Err(FileNameParseError::MissingEncoding { .. })
        ));
        // no extension
        assert!(matches!(
            GgufFileNameParse::from_fname("no_extension"),
            Err(FileNameParseError::MissingExtension { .. })
        ));
    }

    #[test]
    fn long_vs_short_tag_prefers_first_match() {
        let f = GgufFileNameParse::from_fname("foo-Q4_K_M.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q4_K_M);
    }

    #[test]
    fn multiple_dots_in_filename() {
        let f = GgufFileNameParse::from_fname("llama.v3.Q5_1.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q5_1);
    }

    #[test]
    fn bad_extension_variant_is_rejected() {
        assert!(matches!(
            GgufFileNameParse::from_fname("model.gguff"),
            Err(FileNameParseError::IncorrectExtension { .. })
        ));
    }

    #[test]
    fn last_of_two_valid_tags_wins() {
        let f = GgufFileNameParse::from_fname("pipe-Q4_0-to-Q5_0.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q5_0);
    }

    #[test]
    fn mixed_case_and_extension_case_migrated() {
        let f = GgufFileNameParse::from_fname("meta-llama-iq4_xs.GGuF").unwrap();
        assert_eq!(f.file_type, GgmlFileType::IQ4_XS);
    }

    #[test]
    fn delimiter_variations_are_accepted_migrated() {
        let f = GgufFileNameParse::from_fname("model--q4__0.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q4_0);

        let f = GgufFileNameParse::from_fname("model.q4.k.m.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q4_K_M);

        let f = GgufFileNameParse::from_fname("foo\u{2013}Q4_0.gguf").unwrap(); // en‑dash
        assert_eq!(f.file_type, GgmlFileType::Q4_0);
    }

    #[test]
    fn tag_at_string_edges_migrated() {
        let f = GgufFileNameParse::from_fname("Q5_0-foo.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q5_0);

        let f = GgufFileNameParse::from_fname("foo-Q5_0.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q5_0);
    }

    #[test]
    fn longest_match_precedence_when_both_present() {
        let f = GgufFileNameParse::from_fname("foo-Q4_K-Q4_K_M.gguf").unwrap();
        assert_eq!(f.file_type, GgmlFileType::Q4_K_M);
    }

    #[test]
    fn rejects_substrings_without_delimiters_migrated() {
        assert!(matches!(
            GgufFileNameParse::from_fname("alphaq4_0beta.gguf"),
            Err(FileNameParseError::MissingEncoding { .. })
        ));
        assert!(matches!(
            GgufFileNameParse::from_fname("q4k.gguf"),
            Err(FileNameParseError::MissingEncoding { .. })
        ));
    }

    #[test]
    fn directories_are_ignored_migrated() {
        let p1 = GgufFileNameParse::from_fname("some/path/to/Meta-Q5_0.gguf").unwrap();
        let p2 = GgufFileNameParse::from_fname("Meta-Q5_0.gguf").unwrap();
        assert_eq!(p1.file_type, p2.file_type);
    }
}
