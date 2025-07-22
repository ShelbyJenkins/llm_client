use bon::Builder;
use cmdstruct::Arg;
use serde::{Deserialize, Serialize};

/// Represents the prompt for a completion request, which can be provided as text or tokens.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Tokens(TokenIds),
    Text(String),
}

impl From<String> for Prompt {
    fn from(text: String) -> Self {
        Prompt::Text(text)
    }
}

impl From<&String> for Prompt {
    fn from(text: &String) -> Self {
        Prompt::Text(text.to_owned())
    }
}

impl From<&str> for Prompt {
    fn from(text: &str) -> Self {
        Prompt::Text(text.into())
    }
}

impl<T> From<T> for Prompt
where
    T: Into<TokenIds>,
{
    fn from(t: T) -> Self {
        Prompt::Tokens(t.into())
    }
}
/// Wrapper around a vector of token-IDs.
///
/// This transparent new-type lets callers pass `Vec<u32>`, `Vec<u64>` or
/// `Vec<usize>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenIds(pub Vec<u64>);

impl From<Vec<u32>> for TokenIds {
    #[inline]
    fn from(v: Vec<u32>) -> Self {
        TokenIds(v.into_iter().map(|n| n as u64).collect())
    }
}

impl From<Vec<u64>> for TokenIds {
    #[inline]
    fn from(v: Vec<u64>) -> Self {
        TokenIds(v)
    }
}

impl From<Vec<usize>> for TokenIds {
    #[inline]
    fn from(v: Vec<usize>) -> Self {
        TokenIds(v.into_iter().map(|n| n as u64).collect())
    }
}

/// Low-level image data structure for multimodal models.
#[derive(Clone, Serialize, Debug, Deserialize)]
pub struct ImageData {
    /// Raw base64 image bytes.
    pub data: String,
    /// Identifier referenced in the prompt (e.g. `[img-12]` in prompt corresponds to `id = 12`).
    pub id: i64,
}

/// Specifies a LoRA (Low-Rank Adaptation) adapter to apply.
#[derive(Clone, Serialize, Debug, Deserialize)]
pub struct LoraAdapter {
    pub id: i64,
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<isize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<usize>,

    // ── Other fields seen in the server payload but not documented in the API ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_lazy: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_triggers: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<Vec<LoraAdapter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<isize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_discard: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_sampling_probs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preserved_tokens: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_in_content: Option<bool>,

    // dotted keys → flat field names with serde renames
    #[serde(rename = "speculative.n_max", skip_serializing_if = "Option::is_none")]
    pub speculative_n_max: Option<u32>,
    #[serde(rename = "speculative.n_min", skip_serializing_if = "Option::is_none")]
    pub speculative_n_min: Option<u32>,
    #[serde(rename = "speculative.p_min", skip_serializing_if = "Option::is_none")]
    pub speculative_p_min: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_forced_open: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings_per_token: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n_sigma: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xtc_probability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xtc_threshold: Option<f32>,
    #[serde(flatten)]
    pub sampling: SamplingParams,
}

// ──────────────────────────────────────────────────────────────────────────────
//  Sampling parameters (all knobs that influence *how* tokens are chosen)
// ──────────────────────────────────────────────────────────────────────────────
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
#[builder(on(String, into))]
#[builder(derive(Debug, Clone))]
pub struct SamplingParams {
    // ───────────────────────── basic samplers ────────────────────────────────
    /// Temperature to control randomness in sampling.
    ///
    /// Higher values (e.g. `1.0` and above) produce more random output, while
    /// lower values (e.g. `0.2`) make the output more deterministic and focused.
    /// When using temperature, recommended values are typically between
    /// `0.1` and `2.0`.  
    /// *Default:* `0.8`.  
    /// *Note:* If a negative temperature is given, the model will instead use
    /// greedy sampling, i.e. always pick the highest‑probability token (you can
    /// still request probabilities via `n_probs` in that case).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Limit the next token selection to the K most probable tokens.
    ///
    /// The top-K sampler will only consider this many most-likely tokens at each generation step. A smaller `top_k` means the model has fewer options (making it more deterministic), while a larger `top_k` (or 0) means essentially no restriction.
    /// Default: `40`. (Set `0` to disable top-K filtering entirely.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Limit the next token selection to a cumulative probability.
    ///
    /// The top-p (nucleus) sampler considers only the smallest set of tokens whose combined probability mass exceeds this threshold `p`. This dynamically limits the vocabulary considered at each step to a subset that sums to `p`. For example, `0.95` means ~95% of probable tokens are considered.
    /// Default: `0.95`. (Set `1.0` to disable top-p filtering.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Adjust the "typical probability" sampler threshold (locally typical sampling).
    ///
    /// This controls locally typical sampling with parameter `p`. When set below 1.0, at each step the model will prefer tokens whose probability is closer to the expected distribution (entropy) of the remaining options. A value of `1.0` disables this mechanism (no effect, since 100% typical).
    /// Default: `1.0` (disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,

    /// Ensure a minimum probability for tokens to be considered (min-p sampler).
    ///
    /// This sets a floor on token probabilities relative to the most likely token. At each step, any token with probability less than `min_p * (probability of best token)` will be excluded.
    /// Default: `0.05`. (If set to `0.0`, the min-p filter is disabled.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    // ──────────────────────── dynamic temperature ────────────────────────────
    /// Dynamic temperature range for sampling.
    ///
    /// If set, the effective temperature will be randomly chosen for each token within ± this range of the base `temperature`. For example, if `temperature = 0.8` and `dynatemp_range = 0.1`, then the actual temperature for each token will be in [0.7, 0.9].
    /// Default: `0.0` (disabled, no dynamic variation in temperature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynatemp_range: Option<f32>,

    /// Dynamic temperature exponent.
    ///
    /// This exponent modifies the distribution from which dynamic temperatures are drawn (when `dynatemp_range` is used). It influences how biased the random temperature selection is towards the edges or center of the range.
    /// Default: `1.0` (linear distribution).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynatemp_exponent: Option<f32>,

    // ───────────────────── repetition / presence penalties ───────────────────
    /// Repetition penalty factor.
    ///
    /// When greater than `1.0`, the model will penalize tokens that have already appeared, reducing their likelihood (discouraging repetition). A value of `1.0` means no repetition penalty (disabled), and values below `1.0` would *increase* the likelihood of repeats (not commonly used).
    /// Default: `1.1` (slightly discouraging repetition).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,

    /// Last-N repetition penalty context length.
    ///
    /// Only the last `repeat_last_n` tokens are considered when applying the repetition penalty. For example, if set to 64, the model will look at the 64 most recent tokens to penalize repeats. If set to `0`, repetition penalty is disabled entirely. If set to `-1`, the model's full context window is used.
    /// Default: `64`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i64>,

    /// Presence penalty (OpenAI-style).
    ///
    /// A positive presence penalty reduces the probability of any token that has already appeared in the text, regardless of frequency. It encourages the model to talk about new topics.
    /// Default: `0.0` (no presence penalty).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (OpenAI-style).
    ///
    /// A positive frequency penalty reduces the probability of tokens in proportion to how often they have already appeared in the text. This helps prevent the model from repeating the same token frequently.
    /// Default: `0.0` (no frequency penalty).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    // ──────────────────────── DRY (“don’t repeat yourself”) ──────────────────
    /// DRY (Don't Repeat Yourself) penalty multiplier.
    ///
    /// This enables an alternative repetition penalty mechanism. A multiplier > 0 activates DRY sampling: when the model starts to generate a sequence that repeats a recent sequence, it will incur an additional penalty. The penalty applied is determined by this multiplier and grows with the length of the repeating sequence (see `dry_base` and `dry_allowed_length`).
    /// Default: `0.0` (DRY penalty disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f32>,

    /// DRY penalty base value for exponential growth.
    ///
    /// When a repeating sequence is detected and exceeds the allowed length, the penalty is calculated exponentially as `multiplier * (dry_base)^(repetition_length - dry_allowed_length)`.
    /// Default: `1.75`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f32>,

    /// Allowed repetition length for DRY penalty.
    ///
    /// Sequences of tokens can repeat up to this length without penalty. Once a repetition exceeds this length, the DRY penalty will start increasing exponentially.
    /// Default: `2` (the penalty starts applying once a sequence of 3 or more tokens repeats).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<u64>,

    /// DRY penalty window (last-N tokens to consider for repetition).
    ///
    /// Determines how far back the model looks for repeated sequences when applying the DRY penalty. A value of `-1` means the entire context is considered. `0` disables DRY repetition checking entirely.
    /// Default: `-1` (use the full context for detecting repeats).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_penalty_last_n: Option<i64>,

    /// Sequence breakers for DRY sampling.
    ///
    /// A list of strings that, when encountered, will break the sequence for the purpose of DRY repetition detection. In other words, sequences separated by any of these "breakers" are not considered continuous for repetition penalty. By the breakers are newline (`"\n"`), colon (`":"`), double quote (`"\""`), and asterisk (`"*"`), which helps reset repetition at sentence or list boundaries, etc.
    /// Providing a custom list will replace the defaults. You can also specify `["none"]` to indicate that no sequence breakers should be used at all.
    /// Default: `["\n", ":", "\"", "*"]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_sequence_breakers: Option<Vec<String>>,

    // ──────────────────────────── XTC sampling ───────────────────────────────
    /// XTC (eXtreme Token Compression) removal probability.
    ///
    /// XTC sampling randomly removes low-probability tokens to potentially improve generation quality. This setting is the probability of applying the removal at each step. For example, `0.1` means a 10% chance that low-probability tokens (below the threshold) will be removed from consideration at each token generation step.
    /// Default: `0.0` (XTC disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xtc_probability: Option<f32>,

    /// XTC removal probability threshold.
    ///
    /// If XTC is active (see `xtc_probability`), this value is the minimum probability a token must have to *avoid* being removed. Tokens with probability below this threshold are candidates for removal when the XTC mechanism triggers.
    /// **Note:** If `xtc_threshold` is above `0.5`, XTC will effectively be disabled (since the threshold would be too high to remove any token in practice).
    /// Default: `0.1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xtc_threshold: Option<f32>,

    // ─────────────────────────── Mirostat sampling ───────────────────────────
    /// Enable Mirostat sampling for dynamic perplexity control.
    ///
    /// Mirostat is an adaptive sampling algorithm that adjusts token selection to maintain a target perplexity (information content) in the generated text.
    /// - Set to `1` to enable Mirostat (version 1.0), or `2` to enable Mirostat 2.0.
    /// - If Mirostat is enabled, other sampling methods like top-K, top-p, and typical-p are ignored (Mirostat takes control of the token selection process).
    /// Default: `0` (disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u8>,

    /// Mirostat target entropy (tau).
    ///
    /// This is the target entropy value for Mirostat sampling (often denoted as τ). It roughly corresponds to the desired perplexity of the model's responses. A higher value means allowing more uncertainty (higher entropy).
    /// Default: `5.0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,

    /// Mirostat learning rate (eta).
    ///
    /// This controls how quickly Mirostat adjusts its sampling to reach the target entropy. A smaller value makes the adjustments more gradual, while a larger value makes it adapt more aggressively.
    /// Default: `0.1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,

    // ───────────────────────── fine‑grained controls ─────────────────────────
    /// Modify the likelihood of specific tokens appearing in the completion.
    ///
    /// This parameter allows fine-grained control over token selection by adjusting logits. It expects an array of pairs `[[token, bias], ...]` where `token` can be specified by its integer ID or as a string, and `bias` is a float value (or `false`). The bias is added to the model's logit for that token before sampling.
    /// - A positive bias increases the token's probability (e.g. `1.0` adds a moderate boost, while `100` would make a token nearly guaranteed to be selected if possible).
    /// - A negative bias decreases the token's probability (e.g. `-1.0` makes it less likely, and a very large negative like `-100` can effectively ban the token).
    /// - You can also use a boolean `false` as the bias to explicitly forbid a token (equivalent to a very large negative bias).
    /// **Examples:** `[[15043, 1.0]]` might increase the likelihood of the token with ID 15043 (which could be the word "Hello"), whereas `[[15043, false]]` would prevent that token from ever being generated. You can target sequences by providing text: e.g. `[["Hello, World!", -0.5]]` will reduce the likelihood of the exact sequence "Hello, World!" (by applying a penalty to each token in that sequence), similar to a specialized presence penalty for that phrase.
    /// Default: `[]` (no bias adjustments; all tokens are considered with their default probabilities).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Vec<Vec<serde_json::Value>>>,

    /// Return top-N token probabilities with each generated token.
    ///
    /// If set to a value > 0, the response will include an array of the top N tokens (and their probabilities or log-probs) for each step of generation, in addition to the generated text. For example, if `n_probs = 5`, for each token generated the model will output the top 5 candidate tokens it considered and their probabilities at that step.
    /// *Note:* When `temperature < 0` (greedy sampling) or other samplers are in effect, these probabilities are still computed from the raw model logits via a softmax, ignoring the filtering of those samplers.
    /// Default: `0` (no probability info returned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_probs: Option<u32>,

    /// Ensure samplers keep a minimum number of tokens.
    ///
    /// If > 0, this guarantees that each sampler in the chain will always leave at least this many token options. For instance, setting `min_keep = 1` ensures no sampler ever eliminates all tokens; setting `min_keep = 5` ensures at least 5 tokens remain possible after each sampling stage.
    /// This can help avoid situations where aggressive sampling filters (like very low top_p combined with top_k) might remove all candidates.
    /// Default: `0` (no minimum enforced beyond each sampler's own threshold).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_keep: Option<u32>,

    // ───────────────────── sampler chain & reproducibility ───────────────────
    /// Sampler identifiers, encoded the same way the CLI/server expects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samplers: Option<Vec<Sampler>>,

    /// Random seed for generation.
    ///
    /// If you want reproducible results, set this to a specific integer. Use `-1` to randomize on each request.
    /// Default: `-1` (random seed, different each run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// Custom sampler ordering for generation.
///
/// By the model applies samplers in a specific chain (penalties, then dry run, top-K, typical-P, top-P, min-P, XTC, temperature). This field allows you to override that order or choose a subset of samplers. Provide an array of sampler names in the exact order you want them applied.
/// Available sampler names include: `"dry"`, `"top_k"`, `"typ_p"`, `"top_p"`, `"min_p"`, `"xtc"`, `"temperature"`, and `"penalties"` (which covers repeat and presence/frequency penalties). If a sampler name is omitted from the list, that sampling method will not be used. If a name appears multiple times, that sampler will be applied multiple times in sequence.
/// Default order (using all samplers) is: `["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampler {
    Dry,
    TopK,
    TypP,
    TopP,
    MinP,
    Xtc,
    Temperature,
    Penalties,
    /// Any sampler we don’t recognise yet (future-proofing).
    #[serde(other)]
    Unknown,
}

impl std::fmt::Display for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Sampler::Dry => "dry",
            Sampler::TopK => "top_k",
            Sampler::TypP => "typ_p",
            Sampler::TopP => "top_p",
            Sampler::MinP => "min_p",
            Sampler::Xtc => "xtc",
            Sampler::Temperature => "temperature",
            Sampler::Penalties => "penalties",
            Sampler::Unknown => "unknown",
        })
    }
}

/// Pooling strategy for embedding vectors. Applicable if `embeddings_only` mode
/// is used. Options: `"none"` (no pooling, possibly return per-token embeddings),
/// `"mean"` (average all token embeddings), `"cls"` (use the first token's embedding,
/// e.g., [CLS] token), `"last"` (use the last token's embedding), `"rank"` (use a
/// specialized pooling for reranker models).  
///
/// *The lowercase Serde mapping means `"None"` serializes as `"none"`*,
/// which is exactly what the llama.cpp server expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    None,
    Mean,
    Cls,
    Last,
    Rank, // used by reranker models
    /// Any value we don’t model yet (future-proofing).
    #[serde(other)]
    Unknown,
}

impl Arg for Pooling {
    fn append_arg(&self, command: &mut std::process::Command) {
        command.arg(format!("{}", self));
    }
}

impl std::fmt::Display for Pooling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write the exact lowercase string the server flag expects.
        f.write_str(match self {
            Pooling::None => "none",
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::Last => "last",
            Pooling::Rank => "rank",
            Pooling::Unknown => "unknown",
        })
    }
}
