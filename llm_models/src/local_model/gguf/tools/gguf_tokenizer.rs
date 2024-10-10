//! Adapted from: https://github.com/EricLBuehler/mistral.rs/blob/master/feature = "mistral_rs"-core/src/gguf/gguf_tokenizer.rs

use crate::local_model::metadata::tokenizer::{GgmlTokenizerMetadata, GgmlTokenizerModel};
use anyhow::Context;
use tokenizers::{
    decoders::{
        self, byte_fallback::ByteFallback, byte_level::ByteLevel, fuse::Fuse, strip::Strip,
    },
    models::unigram::Unigram,
    normalizers::{self, Prepend, Replace},
    AddedToken, DecoderWrapper, NormalizerWrapper, Tokenizer,
};

struct GgufTokenizerProps<'a> {
    model: GgmlTokenizerModel,
    tokens: &'a Vec<String>,
    added_tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
    unk: Option<u32>,
    eos: u32,
    bos: u32,
}

impl<'a> GgufTokenizerProps<'a> {
    fn new(ggml: &'a GgmlTokenizerMetadata) -> crate::Result<Self> {
        let props = Self {
            model: ggml.model.clone(),
            tokens: &ggml.tokens,
            added_tokens: ggml.added_tokens.clone(),
            scores: ggml.scores.clone(),
            merges: ggml.merges.clone(),
            unk: ggml.unknown_token_id,
            eos: ggml.eos_token_id,
            bos: ggml.bos_token_id,
        };

        Ok(props)
    }
}

/// Some quants have gpt2 as the tokenizer model, but the actual tokenizer should be 'llama'
pub fn convert_gguf_to_hf_tokenizer(ggml: &GgmlTokenizerMetadata) -> crate::Result<Tokenizer> {
    let props = GgufTokenizerProps::new(ggml)?;
    let tokenizer = match props.model {
        GgmlTokenizerModel::Llama | GgmlTokenizerModel::Replit | GgmlTokenizerModel::Gpt2 => {
            unigram_tokenizer(&props)?
        }
        _ => {
            anyhow::bail!("Tokenizer model `{:?}` not supported.", props.model);
        }
    };

    println!(
        "GGUF tokenizer model is `{:?}`, num tokens: {}, num added tokens: {}, num merges: {}, num scores: {}",
        props.model,
        tokenizer.get_vocab_size(true),
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
    );
    Ok(tokenizer)
}

/// Add the special tokens and return their string representations
fn add_special_tokens(
    p: &GgufTokenizerProps,
    tokenizer: &mut Tokenizer,
    bos: u32,
    eos: u32,
    unk: Option<u32>,
) -> crate::Result<()> {
    // A little bit awkward here since eos/bos are assumed not options so we need to handle an Option
    for token_id in [Some(bos), Some(eos), unk].into_iter().flatten() {
        let token = get_token(p, token_id)?;
        tokenizer.add_special_tokens(&[AddedToken::from(token.to_string(), true)]);
    }
    Ok(())
}

fn get_token(p: &GgufTokenizerProps, token_id: u32) -> crate::Result<String> {
    p.tokens
        .get(token_id as usize)
        .map(ToString::to_string)
        .with_context(|| format!("Token not found for ID: {}", token_id))
}

fn unigram_tokenizer(p: &GgufTokenizerProps) -> crate::Result<Tokenizer> {
    let GgufTokenizerProps { unk, eos, bos, .. } = *p;
    // Unigram (SentencePiece) default UNK is 0
    let unk = unk.unwrap_or(0);

    // Create the Tokenizer model:
    let model = {
        let vocab: Vec<(String, f64)> = if let Some(s) = p.scores.as_ref() {
            let scores = s.iter().cloned().map(|f_32| f_32 as f64);
            p.tokens.iter().cloned().zip(scores).collect()
        } else {
            println!("unigram tokenizer is missing required metadata `tokenizer.ggml.scores`");
            p.tokens.iter().cloned().map(|s| (s, -10000.0)).collect()
        };

        Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?
    };

    // Decoder + Normalizer config reference:
    // https://github.com/EricLBuehler/mistral.rs/pull/389#discussion_r1630620763
    let decoder = Decoder::Sequence(vec![
        Decoder::Replace("▁", " "),
        Decoder::ByteFallback,
        Decoder::Fuse,
        Decoder::Strip(' ', 1, 0),
    ]);

    let normalizer = Normalizer::Sequence(vec![
        Normalizer::Prepend("▁"),
        Normalizer::Replace(" ", "▁"),
    ]);
    let mut tokenizer = Tokenizer::new(model);

    let d = DecoderWrapper::try_from(decoder)?;
    tokenizer.with_decoder(Some(d));

    let n = NormalizerWrapper::try_from(normalizer)?;
    tokenizer.with_normalizer(Some(n));

    // Add special tokens (bos, eos, unk):
    add_special_tokens(p, &mut tokenizer, bos, eos, Some(unk))?;

    Ok(tokenizer)
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/decoders/enum.DecoderWrapper.html
#[allow(dead_code)]
enum Decoder<'a> {
    ByteFallback,
    Fuse,
    Replace(&'a str, &'a str),
    Strip(char, usize, usize),
    Sequence(Vec<Self>),
    ByteLevel(bool, bool, bool),
}

// Convert into upstream type wrapped enum variants:
impl TryFrom<Decoder<'_>> for DecoderWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Decoder) -> crate::Result<Self, Self::Error> {
        let value: DecoderWrapper = match variant {
            Decoder::ByteFallback => ByteFallback::default().into(),
            Decoder::Fuse => Fuse::default().into(),
            Decoder::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Decoder::Strip(content, start, stop) => Strip::new(content, start, stop).into(),
            Decoder::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(DecoderWrapper::try_from)
                    .collect::<crate::Result<Vec<DecoderWrapper>>>()?;

                decoders::sequence::Sequence::new(seq).into()
            }
            Decoder::ByteLevel(add_prefix_space, trim_offsets, use_regex) => {
                ByteLevel::new(add_prefix_space, trim_offsets, use_regex).into()
            }
        };

        Ok(value)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/normalizers/enum.NormalizerWrapper.html
enum Normalizer<'a> {
    Prepend(&'a str),
    Replace(&'a str, &'a str),
    Sequence(Vec<Self>),
}

impl TryFrom<Normalizer<'_>> for NormalizerWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Normalizer) -> crate::Result<Self, Self::Error> {
        let value: NormalizerWrapper = match variant {
            Normalizer::Prepend(prepend) => Prepend::new(prepend.to_owned()).into(),
            Normalizer::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Normalizer::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(NormalizerWrapper::try_from)
                    .collect::<crate::Result<Vec<NormalizerWrapper>>>()?;

                normalizers::Sequence::new(seq).into()
            }
        };

        Ok(value)
    }
}
