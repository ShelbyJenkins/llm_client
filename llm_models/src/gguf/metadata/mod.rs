//! This module provides a single, strongly‑typed entry‑point—`GgufMeta`—that
//! collects **all** metadata we care about in a GGUF model file.
//!
//! * **Canonical spec sections** → parsed into domain structs (`General`, `Llm`).
//! * **Derived statistics**      → computed on the fly (`Profile`).
//! * **Tokenizer definition**    → preserved loss‑lessly (`EmbeddedTokenizer`).
//! * **Future / custom keys**    → stored verbatim in a JSON map (`CustomMetadata`).
//!
//! The goal is “one stop‑shop” for downstream tools: load once, then access every
//! standard, extended, and experimental field without re‑reading the GGUF blob.
//!
//! Taken from https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

pub mod general;
pub mod llm;
use std::collections::BTreeMap;

pub mod tokenizer;
use ggus::GGufMetaDataValueType;
use serde_json::Value;

use crate::gguf::metadata::{general::General, llm::Llm, tokenizer::EmbeddedTokenizer};

/// High‑level container that bundles every metadata section extracted from – or
/// computed from – a GGUF model file.
///
/// Constructed via [`GgufMeta::from_gguf_path`], which:
/// 1. Reads the file into memory.
/// 2. Parses the GGUF header + tensors with **`ggus`**.
/// 3. Delegates each sub‑section to its dedicated parser (or calculator).
pub struct GgufMeta {
    /// General model identity & provenance (GGUF *general* section).
    pub general: General,

    /// Core hyper‑parameters of the large‑language‑model (GGUF *llm.* keys).
    pub llm: Llm,

    /// Loss‑less tokenizer description (GGML, Hugging Face, RWKV, …).
    pub tokenizer: EmbeddedTokenizer,

    /// Any non‑standard or forward‑compatible metadata keys, stored verbatim
    /// as JSON for easy inspection and future use.
    pub custom: CustomMetadata,
}

impl GgufMeta {
    /// Load a GGUF file from `path`, parse all known sections, and compute the
    /// derived [`Profile`].
    ///
    /// # Panics
    /// * If the file cannot be read (`std::fs::read` error).
    /// * If `ggus` fails to parse the GGUF header.
    ///
    /// Prefer fallible wrappers in production code; this helper keeps the demo
    /// concise.
    pub fn from_gguf_path(path: &std::path::Path) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let gguf = ggus::GGuf::new(&bytes).unwrap();

        Self {
            general: General::new(&gguf).unwrap(),
            llm: Llm::new(&gguf),
            tokenizer: EmbeddedTokenizer::new(&gguf),
            custom: CustomMetadata::new(&gguf),
        }
    }

    pub fn from_gguf(gguf: &ggus::GGuf<'_>) -> Self {
        Self {
            general: General::new(gguf).unwrap(),
            llm: Llm::new(gguf),
            tokenizer: EmbeddedTokenizer::new(gguf),
            custom: CustomMetadata::new(gguf),
        }
    }
}

pub struct CustomMetadata(pub BTreeMap<String, Value>);

impl CustomMetadata {
    pub fn new(gguf: &ggus::GGuf) -> Self {
        let mut map = BTreeMap::new();
        for (&key, kv) in &gguf.meta_kvs {
            if let Some(v) = Self::kv_to_json(kv) {
                map.insert(key.to_owned(), v); // JSON = loss‑less & ergonomic
            }
        }
        Self(map)
    }

    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.0
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    fn kv_to_json(kv: &ggus::GGufMetaKV) -> Option<Value> {
        use GGufMetaDataValueType as Ty;
        let mut r = kv.value_reader();
        Some(match kv.ty() {
            Ty::U8 => Value::Number((r.read::<u8>().ok()? as u64).into()),
            Ty::I8 => Value::Number((r.read::<i8>().ok()? as i64).into()),
            Ty::U16 => Value::Number((r.read::<u16>().ok()? as u64).into()),
            Ty::I16 => Value::Number((r.read::<i16>().ok()? as i64).into()),
            Ty::U32 => Value::Number((r.read::<u32>().ok()? as u64).into()),
            Ty::I32 => Value::Number((r.read::<i32>().ok()? as i64).into()),
            Ty::U64 => Value::Number((r.read::<u64>().ok()?).into()),
            Ty::I64 => Value::Number((r.read::<i64>().ok()?).into()),
            Ty::F32 => Value::Number(serde_json::Number::from_f64(r.read::<f32>().ok()? as f64)?),
            Ty::F64 => Value::Number(serde_json::Number::from_f64(r.read::<f64>().ok()?)?),
            Ty::Bool => Value::Bool(r.read_bool().ok()?),
            Ty::String => Value::String(r.read_str().ok()?.to_owned()),
            Ty::Array => {
                // recursively convert arrays; skip on error to keep `new` infallible
                let (_, len) = r.read_arr_header().ok()?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    let fake_kv = unsafe { ggus::GGufMetaKV::new_unchecked(r.remaining()) };
                    if let Some(jv) = Self::kv_to_json(&fake_kv) {
                        arr.push(jv)
                    }
                    r = fake_kv.value_reader();
                }
                Value::Array(arr)
            }
        })
    }
}
