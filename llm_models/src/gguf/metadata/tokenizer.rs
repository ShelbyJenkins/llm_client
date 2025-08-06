use std::collections::BTreeMap;

use ggus::{GGuf, GGufMetaMapExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::{
    AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer,
    decoders::{self, byte_fallback::ByteFallback, fuse::Fuse, strip::Strip},
    models::bpe::{BPE, BpeBuilder, Vocab},
    normalizers::{self, Replace},
    pre_tokenizers, processors,
};

/// Concrete tokenizer definition embedded in a GGUF file.
///
/// GGUF can encode a tokenizer *four* different ways:
///
/// 1. **`Ggml` fields** – the original GGML metadata schema (keys `tokenizer.ggml.*`).
/// 2. **Hugging Face JSON** – a verbatim `tokenizer.json` string stored under
///    `tokenizer.huggingface.json`.
/// 3. **RWKV‑World vocab** – a plain‑text vocabulary (one token per line) used by
///    older RWKV checkpoints; builder still needs to create a byte‑level BPE.
/// 4. **Other / future** – any key starting with `tokenizer.` that doesn’t fit
///    the above; we preserve it verbatim so nothing is lost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmbeddedTokenizer {
    /// Tokenizer expressed via the GGML key family `tokenizer.ggml.*`.
    ///
    /// Almost every public GGUF LLM uses this layout.  All per‑field semantics
    /// mirror the [GGUF spec §*Tokenizers*](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
    Ggml {
        /// High‑level algorithm class – *SentencePiece‑BPE* or *Byte‑level BPE*.
        model: TokenizerKind,

        /// Vocabulary, **index = token‑ID**.  
        /// For `SpmBpe` this comes from `tokenizer.ggml.tokens` (SentencePiece
        /// serialized pieces).  For `ByteLevelBpe` it is the byte‑level BPE
        /// “strings” list (UTF‑8 with control‑char escapes).
        tokens: Vec<String>,

        /// Score per token (`tokenizer.ggml.scores`).  
        /// Only SentencePiece stores meaningful log‑probs; BPE tokenizers omit
        /// the field, so this is usually `None` for `ByteLevelBpe`.
        scores: Option<Vec<f32>>,

        /// **Token‑type / flags** (`tokenizer.ggml.token_type`).  
        /// Unused by current LLMs but defined for completeness; WordPiece models
        /// may store 0 = begin‑word, 1 = continuation, etc.
        token_type: Option<Vec<i32>>,

        /// Name of a pre‑tokenizer recipe (`tokenizer.ggml.pre`).  
        /// • `"qwen2"` – inserts spaces around CJK chars for Qwen‑2.<br>
        /// • `"llama-bpe"` – Llama‑3 regex pre‑step.<br>
        /// • `"default"` – no extra preprocessing.  
        /// Omitted for classic LLaMA‑2 and GPT‑2 tokenizers.
        pre: Option<PreTokenizerRecipe>,

        /// BPE merge rules (`tokenizer.ggml.merges`).  
        /// Present for both `SpmBpe` and `ByteLevelBpe`; absent for pure Unigram.
        merges: Option<Vec<String>>,

        /// Additional tokens appended **after** the original vocabulary
        /// (`tokenizer.ggml.added_tokens`).  Converters tack on `<|endoftext|>`
        /// or image tokens here.
        added_tokens: Option<Vec<String>>,

        /// Collected special‑token IDs *and* behavioural flags; see
        /// [`SpecialTokens`] for full details.
        special: SpecialTokens,

        /// Optional Jinja2 chat‑prompt template (`tokenizer.chat_template`).  
        /// Used by newer chat models (Llama‑3, Qwen‑2, Gemma‑Instruct, etc.) to
        /// ensure identical formatting across toolchains.
        chat_template: Option<String>,
    },

    /// Verbatim Hugging Face *`tokenizer.json`* payload (usually large).
    ///
    /// Lossless round‑trip: when present, we keep the exact JSON string so that
    /// downstream tools—especially Python’s *tokenizers*—can reproduce the
    /// original behaviour byte‑for‑byte.
    HuggingFace {
        /// Raw UTF‑8 JSON.
        json: String,
        /// Optional chat‑template string (same semantics as above).
        chat_template: Option<String>,
    },

    /// RWKV‑World format – a single text file with one token per line.
    ///
    /// At load time we still categorise it as `ByteLevelBpe` internally; the
    /// only difference is the *container*, not the algorithm.
    RwkvWorld {
        /// Entire vocabulary text.
        vocab_text: String,
        chat_template: Option<String>,
    },

    /// Catch‑all for future / experimental formats so we never throw data away.
    Other {
        /// Every `tokenizer.*` key we don’t recognise, in raw bytes → hex form.
        raw: BTreeMap<String, Value>,
    },
}

impl EmbeddedTokenizer {
    // We need to rewrite to return errors instead of None
    // Fields can be None, but parse errors should be handled
    pub fn new(gguf: &ggus::GGuf) -> Self {
        let chat_template = gguf.tokenizer_chat_template().ok().map(|s| s.to_owned());

        if let Ok(model) = gguf.tokenizer_ggml_model() {
            let tokens: Vec<String> = gguf
                .tokenizer_ggml_tokens()
                .ok()
                .map(|arr| arr.filter_map(Result::ok).map(str::to_owned).collect())
                .unwrap_or_default();

            let scores = gguf
                .tokenizer_ggml_scores()
                .ok()
                .and_then(|arr| arr.collect::<Result<Vec<f32>, _>>().ok());

            let token_type = gguf
                .tokenizer_ggml_token_type()
                .ok()
                .and_then(|arr| arr.collect::<Result<Vec<i32>, _>>().ok());

            let merges = gguf.tokenizer_ggml_merges().ok().and_then(|arr| {
                arr.map(|r| r.map(str::to_owned))
                    .collect::<Result<Vec<String>, _>>()
                    .ok()
            });

            let added_tokens = gguf.tokenizer_ggml_added_tokens().ok().and_then(|arr| {
                arr.map(|r| r.map(str::to_owned))
                    .collect::<Result<Vec<String>, _>>()
                    .ok()
            });

            return Self::Ggml {
                model: TokenizerKind::new(model),
                tokens,
                scores,
                token_type,
                pre: PreTokenizerRecipe::new(gguf.get_str("tokenizer.ggml.pre").ok()),
                merges,
                added_tokens,
                special: SpecialTokens::from_gguf(gguf),
                chat_template,
            };
        }

        if let Ok(json) = gguf.get_str("tokenizer.huggingface.json") {
            return Self::HuggingFace {
                json: json.to_owned(),
                chat_template,
            };
        }

        if let Ok(vocab_text) = gguf.tokenizer_rwkv_world() {
            return Self::RwkvWorld {
                vocab_text: vocab_text.to_owned(),
                chat_template,
            };
        }

        let mut raw = BTreeMap::new();
        for (&key, kv) in &gguf.meta_kvs {
            if key.starts_with("tokenizer.") {
                let value_str = format!("{:?}", kv.value_bytes());
                raw.insert(key.to_string(), Value::String(value_str));
            }
        }
        Self::Other { raw }
    }

    pub fn to_fast_tokenizer(&self) -> tokenizers::Tokenizer {
        match self {
            EmbeddedTokenizer::Ggml {
                model,
                tokens,
                merges,
                added_tokens,
                special,
                pre,
                ..
            } => build_tokenizer(
                model,
                tokens,
                merges,
                added_tokens,
                &pre,
                &special.bos,
                &special.eos,
                &special.unknown,
                &special.add_space_prefix,
            ),

            EmbeddedTokenizer::HuggingFace { json, .. } => {
                tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("invalid tokenizer.json")
            }

            EmbeddedTokenizer::RwkvWorld { .. } => {
                todo!("Handle other tokenizer formats")
            }
            EmbeddedTokenizer::Other { .. } => todo!("Handle other tokenizer formats"),
        }
    }
}

/// High‑level classification of the tokenizer **algorithm** recorded in a GGUF
/// file.
/// See https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py
///
/// ### `SpmBpe` – SentencePiece + BPE (“LLaMA‑style”)
/// * **Typical `tokenizer.ggml.model` strings:**  
///   `llama`, `replit`, `phi3`, `codellama`, `gemma`, `mistral`, `mixtral`,
///   `vicuna`, `alpaca`, `baichuan`, …  
/// * **Recipe:** SentencePiece *Unigram* vocabulary **+** BPE merges  
///   (`byte_fallback = true`, `fuse_unk = true`).  
/// * **Coverage:** Meta Llama‑2 & 3, Mistral/Mixtral, Gemma, Replit, Phind‑Phi3,
///   Vicuna, Alpaca, Baichuan—every descendant of the original LLaMA tokenizer.
/// * **GGUF quirk:** Converters almost always write the string **`"llama"`**
///   even for brands like *Mistral*; aliases above are included for safety in
///   case future converters store the brand name directly.
///
/// ### `ByteLevelBpe` – OpenAI byte‑level BPE (“GPT‑2‑style”)
/// * **Typical strings:** `gpt2`, `openai-gpt`, `gptneo`, `gptneox`, `bloom`,
///   `falcon`, `qwen`, `qwen2`, `rwkv`  
/// * **Recipe:** Classic GPT‑2 byte‑level BPE (no `scores`, no byte‑fallback).  
/// * **Coverage:** GPT‑2/3 family, GPT‑J/Neo/NeoX, Bloom, Falcon, StableLM,
///   Qwen‑2, Llama‑3 (yes—Meta switched to byte‑level BPE), and RWKV
///   “World” vocab checkpoints (65 536 flat tokens, treated as byte‑level BPE).
///
/// ### `Other(String)` – future‑proof bucket
/// Any unrecognised tag is preserved verbatim so that newer tokenizers (e.g.
/// pure SentencePiece Unigram, WordPiece, or Baichuan‑3) can be detected
/// without crashing older builds.  Down‑stream code can inspect
/// `Other(name)` and decide how to handle it.
///
/// The accompanying `TokenizerKind::new` helper normalises the raw metadata
/// string to lower‑case and performs a broad match against the lists above;
/// anything else passes through unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenizerKind {
    /// SentencePiece + BPE (“LLaMA‑style”) tokenizer.
    SpmBpe,
    /// OpenAI byte‑level BPE (“GPT‑2‑style”) tokenizer.
    ByteLevelBpe,
    /// Fallback for unrecognised or experimental tokenizer tags.
    Other(String),
}

impl TokenizerKind {
    /// Convert a raw `tokenizer.ggml.model` tag into a `TokenizerKind`.
    pub fn new(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        match m.as_str() {
            // SentencePiece‑BPE family
            "llama" | "codellama" | "gemma" | "replit" | "phi3" | "mistral" | "mixtral"
            | "vicuna" | "alpaca" | "baichuan" => Self::SpmBpe,

            // Byte‑level BPE family
            "gpt2" | "openai-gpt" | "gptneo" | "gptneox" | "bloom" | "falcon" | "qwen"
            | "qwen2" | "rwkv" | "opt" | "mpt" | "starcoder" | "xgen" | "deepseek" => {
                Self::ByteLevelBpe
            }

            // Anything else stays verbatim
            other => Self::Other(other.to_owned()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreTokenizerRecipe {
    // Mainstream byte‑level BPE families
    Gpt2,
    LlamaBpe,
    Qwen2,
    DeepseekV3,
    Tekken,
    Bloom,
    RobertaBpe,
    Starcoder,
    ChatGlmBpe,
    Dbrx,
    // WordPiece / other
    XlmRoberta,

    /// Anything unforeseen lands here.
    Unknown(String),
}

impl PreTokenizerRecipe {
    pub fn new<S: AsRef<str>>(pre: Option<S>) -> Option<Self> {
        let pre = if let Some(p) = pre {
            p.as_ref().to_ascii_lowercase()
        } else {
            return None;
        };
        let res = match pre.as_str() {
            // byte‑level BPE clusters
            "gpt2" | "gpt-2" | "mpt" | "stablelm2" | "command-r" | "olmo" => Self::Gpt2,
            "llama-bpe" | "llama3" | "smaug-bpe" | "llama4" => Self::LlamaBpe,
            "qwen2" => Self::Qwen2,
            "deepseek-v3" => Self::DeepseekV3,
            "deepseek-r1-qwen" => Self::DeepseekV3,
            "tekken" | "mistral-bpe" => Self::Tekken,

            "bloom" => Self::Bloom,
            "roberta-bpe" => Self::RobertaBpe,
            "starcoder" => Self::Starcoder,
            "chatglm-bpe" => Self::ChatGlmBpe,
            "dbrx" | "dbrx-bpe" | "gpt-4o" => Self::Dbrx,
            "xlm-roberta" | "hunyuan" => Self::XlmRoberta,
            other => Self::Unknown(other.to_owned()),
        };

        Some(res)
    }

    pub fn build(&self) -> PreTokenizerWrapper {
        match self {
            Self::Gpt2 | Self::Bloom | Self::RobertaBpe | Self::Starcoder | Self::Tekken => {
                pre_tokenizers::byte_level::ByteLevel::default().into()
            }

            // ───────────────────────────────────────────────────────────────
            // LLaMA‑3 / “llama‑bpe” – Regex ➜ ByteLevel sequence
            // ───────────────────────────────────────────────────────────────
            Self::LlamaBpe => {
                const LLAMA_BPE_REGEX: &str = concat!(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)",    // contractions
                    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",  // words
                    "|\\p{N}{1,3}",                    // 1‑3 digit numbers
                    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*", // punctuation possibly w/ newline
                    "|\\s*[\\r\\n]+",                  // newline runs
                    "|\\s+(?!\\S)",                    // inner whitespace
                    "|\\s+"                            // fallback whitespace
                );

                let split = pre_tokenizers::split::Split::new(
                    pre_tokenizers::split::SplitPattern::Regex(LLAMA_BPE_REGEX.into()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap();

                pre_tokenizers::sequence::Sequence::new(vec![
                    PreTokenizerWrapper::Split(split),
                    PreTokenizerWrapper::ByteLevel(pre_tokenizers::byte_level::ByteLevel::new(
                        false, false, false,
                    )),
                ])
                .into()
            }

            // ───────────────────────────────────────────────────────────────
            // Qwen‑2 and DeepSeek‑V3 – same regex strategy as LLaMA‑3
            // (DeepSeek keeps identical pattern; Qwen uses \p{N}+ for digits)
            // ───────────────────────────────────────────────────────────────
            Self::Qwen2 | Self::DeepseekV3 => {
                const QWEN_REGEX: &str = concat!(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
                    "|\\p{N}+",
                    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
                    "|\\s*[\\r\\n]+",
                    "|\\s+(?!\\S)",
                    "|\\s+"
                );

                let split = pre_tokenizers::split::Split::new(
                    pre_tokenizers::split::SplitPattern::Regex(QWEN_REGEX.into()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap();

                pre_tokenizers::sequence::Sequence::new(vec![
                    PreTokenizerWrapper::Split(split),
                    PreTokenizerWrapper::ByteLevel(pre_tokenizers::byte_level::ByteLevel::new(
                        false, false, false,
                    )),
                ])
                .into()
            }

            // ───────────────────────────────────────────────────────────────
            // ChatGLM‑BPE & XLM‑Roberta – SentencePiece‑style Metaspace
            // ───────────────────────────────────────────────────────────────
            Self::ChatGlmBpe | Self::XlmRoberta => decoders::metaspace::Metaspace::new(
                '▁',
                decoders::metaspace::PrependScheme::Always,
                false,
            )
            .into(),
            Self::Dbrx => {
                const GPT4_REGEX: &str = concat!(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
                    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|",
                    "\\p{N}{1,3}|",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|",
                    "\\s*[\\r\\n]+|",
                    "\\s+(?!\\S)|",
                    "\\s+"
                );
                let split = pre_tokenizers::split::Split::new(
                    pre_tokenizers::split::SplitPattern::Regex(GPT4_REGEX.into()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap();
                PreTokenizerWrapper::Split(split)
            }
            // ───────────────────────────────────────────────────────────────
            // Unknown tag – fall back to GPT‑2 ByteLevel as safest default
            // ───────────────────────────────────────────────────────────────
            Self::Unknown(t) => {
                unimplemented!(" PreTokenizerRecipe::Unknown({t}) not implemented yet")
            }
        }
    }
}

/// All special‑purpose token IDs and behaviour flags in a GGUF tokenizer.
///
/// Each field corresponds 1‑to‑1 with a documented GGUF metadata key, so you
/// can diff a `SpecialTokens` value against the raw file with confidence.
///
/// *Presence semantics*  
/// If the key is **missing** in the GGUF file we store `None`; if the key exists
/// but the ID is deliberately set to 0 × FFFF_FFFF that still parses as
/// `Some(0xffff_ffff)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// **BOS** – Beginning‑of‑Sequence token ID (`tokenizer.ggml.bos_token_id`).
    /// *Examples*  
    /// • LLaMA‑2: `<s>` (ID 1) is always added when `add_bos == true`.  
    /// • BERT:   `[CLS]` is BOS and also serves as classification token.
    pub bos: Option<u32>,

    /// **EOS** – End‑of‑Sequence token ID (`tokenizer.ggml.eos_token_id`).
    /// *Decoder‑only LLMs* often reuse EOS both for generation stop and padding.
    pub eos: Option<u32>,

    /// **UNK** – Unknown / out‑of‑vocabulary token ID
    /// (`tokenizer.ggml.unknown_token_id`).  
    /// Always present in SentencePiece; usually *absent* in GPT‑2 BPE where
    /// every byte sequence can be encoded.
    pub unknown: Option<u32>,

    /// **SEP** – Sequence‑separator token ID (`tokenizer.ggml.separator_token_id`).  
    /// Classic BERT uses `[SEP]` both between sentence‑pairs and as EOS.
    pub separator: Option<u32>,

    /// **PAD** – Padding token ID (`tokenizer.ggml.padding_token_id`).  
    /// Decoder‑only models seldom define this; encoder/seq2seq models often do.
    pub padding: Option<u32>,

    /// **EOT** – End‑of‑Text token ID (`tokenizer.ggml.eot_token_id`).  
    /// Distinct from EOS in some chat models that want a *soft* stop token.
    pub eot: Option<u32>,

    /// **EOM** – End‑of‑Message token ID (`tokenizer.ggml.eom_token_id`).  
    /// Newer multi‑turn chat checkpoints (e.g. *Llama 3 Chat*) mark the end of
    /// each message with this token.
    pub eom: Option<u32>,

    /// **MASK** – Mask token ID (`tokenizer.ggml.mask_token_id`).  
    /// Only encoder or prefix‑LM checkpoints that still support MLM pre‑training
    /// carry this (e.g. Gemma‑2B‑It can keep `[MASK]` for finetuning).
    pub mask: Option<u32>,

    /// Auto‑add **BOS** when encoding plain text
    /// (`tokenizer.ggml.add_bos_token`).  Common default for LLaMA.
    pub add_bos: Option<bool>,

    /// Auto‑add **EOS** when encoding (`tokenizer.ggml.add_eos_token`).
    /// Many chat templates expect trailing EOS when this is true.
    pub add_eos: Option<bool>,

    /// Auto‑insert **SEP** between input segments
    /// (`tokenizer.ggml.add_sep_token`).  Relevant to BERT‑style pair‑inputs.
    pub add_sep: Option<bool>,

    /// Whether the runtime **must prepend a single ASCII space (`" "`) to the very
    /// beginning of each prompt *before* tokenization**.
    ///
    /// In OpenAI‑style byte‑level BPE tokenizers (GPT‑2, GPT‑J, Falcon, RWKV …)
    /// the training data always contained a leading space, and Hugging Face’s fast
    /// `ByteLevel` pre‑tokenizer sets `add_prefix_space = true` to reproduce that
    /// behaviour.  
    ///
    /// * `Some(true)` → insert the space automatically.  
    /// * `Some(false)` or `None` → encode the prompt exactly as given.
    ///
    /// LLaMA / Mistral / SentencePiece tokenizers leave this flag unset because
    /// they rely on the `▁` marker instead of a literal space.
    pub add_space_prefix: Option<bool>,
}

impl SpecialTokens {
    pub fn from_gguf(gguf: &GGuf) -> Self {
        Self {
            bos: gguf.tokenizer_ggml_bos_token_id().ok(),
            eos: gguf.tokenizer_ggml_eos_token_id().ok(),
            unknown: gguf.tokenizer_ggml_unknown_token_id().ok(),
            separator: gguf.tokenizer_ggml_separator_token_id().ok(),
            padding: gguf.tokenizer_ggml_padding_token_id().ok(),
            eot: gguf.get_u32("tokenizer.ggml.eot_token_id").ok(),
            eom: gguf.get_u32("tokenizer.ggml.eom_token_id").ok(),
            mask: gguf.get_u32("tokenizer.ggml.mask_token_id").ok(),
            add_bos: gguf.get_bool("tokenizer.ggml.add_bos_token").ok(),
            add_eos: gguf.get_bool("tokenizer.ggml.add_eos_token").ok(),
            add_sep: gguf.get_bool("tokenizer.ggml.add_sep_token").ok(),
            add_space_prefix: gguf.get_bool("tokenizer.ggml.add_space_prefix").ok(),
        }
    }
}

pub fn build_tokenizer(
    kind: &TokenizerKind,
    tokens: &[String],
    merges: &Option<Vec<String>>,
    added_tokens: &Option<Vec<String>>,
    pre: &Option<PreTokenizerRecipe>,
    bos: &Option<u32>,
    eos: &Option<u32>,
    unknown: &Option<u32>,
    add_space_prefix: &Option<bool>,
) -> Tokenizer {
    let add_prefix = add_space_prefix.unwrap_or(false);
    let merges_txt = merges.as_ref().expect("GGUF tokenizer missing `merges`");

    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    let merges: Vec<(String, String)> = merges_txt
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_owned(), b.to_owned()))
                .expect("invalid merge line")
        })
        .collect::<Vec<_>>();

    let mut builder = BpeBuilder::new().vocab_and_merges(vocab, merges);

    if let Some(unk_id) = unknown {
        builder = builder.unk_token(tokens[*unk_id as usize].clone());
    }

    match kind {
        TokenizerKind::SpmBpe => {
            builder = builder.byte_fallback(true).fuse_unk(true);
        }
        TokenizerKind::ByteLevelBpe => { /* defaults fine */ }
        TokenizerKind::Other(tokenizer_type) => {
            todo!("Tokenizer type not yet implemented: {}", tokenizer_type);
        }
    }

    let model: BPE = builder.build().unwrap();
    let mut tokenizer = Tokenizer::new(ModelWrapper::from(model));

    let pre_tok_opt: Option<PreTokenizerWrapper> = match (kind, pre) {
        (_, Some(recipe)) => Some(recipe.build()),

        (TokenizerKind::ByteLevelBpe, _) => {
            Some(pre_tokenizers::byte_level::ByteLevel::new(add_prefix, false, false).into())
        }

        _ => None,
    };

    if let Some(pt) = pre_tok_opt {
        tokenizer.with_pre_tokenizer(Some(pt));
    }

    match kind {
        TokenizerKind::SpmBpe => {
            // Normalizer: prepend "▁" + convert spaces → "▁"
            let llama_norm = NormalizerWrapper::from(normalizers::Sequence::new(vec![
                normalizers::Prepend::new("▁".to_string()).into(),
                normalizers::Replace::new(" ", "▁").unwrap().into(),
            ]));
            tokenizer.with_normalizer(Some(llama_norm));

            // Decoder: Replace▁→space, ByteFallback, Fuse, Strip leading space.

            let llama_dec = DecoderWrapper::from(decoders::sequence::Sequence::new(vec![
                Replace::new("▁", " ").unwrap().into(),
                ByteFallback::default().into(),
                Fuse::default().into(),
                Strip::new(' ', 1, 0).into(),
            ]));

            tokenizer.with_decoder(Some(llama_dec));
        }

        TokenizerKind::ByteLevelBpe => {
            let bl_dec = decoders::byte_level::ByteLevel::new(add_prefix, false, false);
            let bl_post = processors::byte_level::ByteLevel::new(add_prefix, false, false);

            let decoder: DecoderWrapper = bl_dec.into();
            let postproc: PostProcessorWrapper = bl_post.into();

            tokenizer.with_decoder(Some(decoder));
            tokenizer.with_post_processor(Some(postproc));
        }

        TokenizerKind::Other(tokenizer_type) => {
            todo!("Tokenizer type not yet implemented: {}", tokenizer_type);
        }
    }

    for id in [bos, eos, unknown] {
        if let Some(idx) = id {
            tokenizer.add_special_tokens(&[AddedToken::from(tokens[*idx as usize].clone(), true)]);
        }
    }
    if let Some(extra) = added_tokens {
        tokenizer.add_tokens(
            &extra
                .iter()
                .map(|t| AddedToken::from(t.clone(), false))
                .collect::<Vec<_>>(),
        );
    }

    tokenizer
}
