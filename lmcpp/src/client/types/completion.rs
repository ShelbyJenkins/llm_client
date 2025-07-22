use std::ops::Not;

use bon::Builder;
use serde::{Deserialize, Serialize};

use super::generation_settings::*;

#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
#[builder(on(String, into))]
#[builder(derive(Debug, Clone), state_mod(vis = "pub"))]
pub struct CompletionRequest {
    /// The input prompt for text generation. This can be a string of text or a sequence of token IDs (or a mixed list of both).
    ///
    /// If `cache_prompt` is enabled and this prompt has a common prefix with the previous request, the shared prefix can be reused from the cache (avoiding re-processing it).
    /// A beginning-of-stream token will be inserted automatically at the start of the prompt if required by the model (when the prompt is text and the model's tokenizer metadata indicates BOS is needed).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub prompt: Option<Prompt>,

    /// A BNF-style grammar to constrain the generated text. Provide the grammar rules as a single string.
    ///
    /// When set, the model will only produce outputs that conform to this grammar (using grammar-based sampling). See the [GBNF format library](https://github.com/richardanaya/gbnf) for the grammar syntax and examples.
    /// Default: no grammar (unconstrained generation, except by other sampling parameters).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,

    /// A JSON Schema to constrain the generated text, using grammar-based sampling.
    ///
    /// Provide a JSON object that defines a schema (following the JSON Schema standard, e.g. Draft 2020-12). The output will be constrained to be a JSON structure that validates against this schema. For example:
    /// ```json
    /// { "items": { "type": "string" }, "minItems": 10, "maxItems": 100 }
    /// ```
    /// would constrain the output to a JSON array of strings with length between 10 and 100. An empty schema `{}` will allow any JSON object.
    /// *Note:* If your schema uses external `$ref`s, you may need to convert it to grammar format (see `json_schema_to_grammar.py` in the llama.cpp repository).
    /// Default: None (no JSON schema constraint).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,

    /// The maximum number of tokens to generate in this completion.
    ///
    /// **Note:** The model may generate a few more tokens than this limit if the last token is a multibyte character. If set to `0`, no new tokens will be generated at all (the request will only evaluate the prompt into the cache). A special value of `-1` is treated as "infinite" (no specific limit).
    /// Default: `-1` (no fixed limit on generation length).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<u64>,

    /// Specify the minimum indent (number of leading whitespace characters) for each new line in the generated text.
    ///
    /// This is useful for code completion tasks to ensure the model continues with the correct indentation level.
    /// Default: `0` (no forced indentation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_indent: Option<u32>,

    /// Specify the number of tokens from the prompt to retain in the context when the context window is exceeded.
    ///
    /// If the prompt plus generated tokens would exceed the model's context length, older tokens will be dropped. This parameter sets how many initial prompt tokens to keep (excluding any BOS token) rather than dropping the entire prompt.
    /// By default this is `0`, meaning no prompt tokens are kept (the prompt can be entirely discarded if needed). Use `-1` to keep all prompt tokens regardless of length.
    /// Default: `0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_keep: Option<i64>,

    /// Whether to stream the response tokens as they are generated.
    ///
    /// If set to `true`, the completion will be sent back token-by-token in a streaming fashion (using Server-Sent Events), rather than waiting for the full completion. In streaming mode, interim partial responses will contain the next token and a flag indicating if generation is complete.
    /// Default: `false` (no streaming; the response will contain the full generated text once complete).
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub stream: bool,

    /// A list of stop strings that will halt generation when encountered.
    ///
    /// If any of these substrings appear in the output, the generation will stop immediately before including them. These stopping strings are not included in the final output, so if you want them in a continued conversation you should manually add them back to the next prompt.
    /// Default: `[]` (no specific stopping sequences, aside from the model's EOS if not disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Ignore the end-of-stream (EOS) token and continue generation.
    ///
    /// If true, the model will not stop when it predicts the end-of-stream token. Internally, this is typically implemented by biasing the EOS token to have negative infinite probability (effectively banning it).
    /// Use this if you want the model to continue generating beyond where it would normally stop.
    /// Default: `false` (the model stops at EOS as usual, unless a `stop` string or context limit intervenes).
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub ignore_eos: bool,

    /// Maximum generation time in milliseconds.
    ///
    /// If set to a value > 0, the generation phase (after prompt processing) will be limited to approximately this amount of time. Once at least one newline character has been generated and the time limit is exceeded, the generation will stop as if a stop condition were met. This is useful for limiting how long the model can continue, for instance in real-time applications or fill-in-the-middle tasks to prevent it from rambling indefinitely.
    /// Default: `0` (no time limit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t_max_predict_ms: Option<u64>,

    /// An array of image data to be used with multimodal models.
    ///
    /// Use this field to send images (as base64-encoded strings) to a multimodal model. Each image is accompanied by an `id` number. In the prompt text, you can reference an image by writing a placeholder like `[img-<id>]`. When processing the prompt, the server will replace that placeholder with the embedding of the corresponding image provided here.
    /// For example, if your prompt is `User: [img-12] Describe the image.\nAssistant:` and you include `image_data: [{ "data": "<BASE64>", "id": 12 }]`, the model will see the image with id 12 at that point in the prompt.
    /// Only use this with models that support image inputs (e.g. LLaVA). Default: None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Vec<ImageData>>,

    /// Specify a slot ID for this request (advanced usage for multi-slot servers).
    ///
    /// Some server deployments allow multiple model slots/contexts to run in parallel. This field can bind the request to a specific slot index.
    /// - If set to `-1`, the server will automatically choose any available idle slot.
    /// - If set to a non-negative integer, the request will only run on that specific slot (and will wait if it's busy).
    /// Default: `-1` (no slot preference, use any idle slot).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<i32>,

    /// Re-use the KV cache from a previous request if possible to speed up processing of a shared prompt prefix.
    ///
    /// When enabled, the server will attempt to reuse the existing attention key/value cache for the initial part of the prompt if this prompt starts with the same tokens as the last prompt. This means only the new suffix of the prompt (that wasn't seen before) will be processed by the model, which can significantly improve performance for repeated interactions with a fixed prefix (like system or conversation context).
    /// **Warning:** Due to differences in batch processing vs. single processing, enabling cache re-use may result in slightly nondeterministic outputs (logits might not be exactly the same as a full re-evaluation of the prompt).
    /// Default: `true` (the server will reuse cache by default if a common prefix is detected).
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub cache_prompt: bool,

    /// Include the raw generated token IDs in the response.
    ///
    /// If `true`, the response JSON will contain a field `tokens` with the sequence of token IDs that were generated (in addition to the textual `content`). If `false`, the `tokens` field will be omitted or empty (unless in streaming mode, where tokens are sent as they stream).
    /// Default: `false`.
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub return_tokens: bool,

    /// Include per-token timing information in the response.
    ///
    /// If true, the server will report how long it took to process the prompt and to generate each token (tokens per second, etc.) in the response. This is useful for debugging or monitoring performance, as each response can include detailed timing breakdowns.
    /// Default: `false`.
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub timings_per_token: bool,

    /// Return probabilities instead of log probabilities after sampling.
    ///
    /// If enabled (and `n_probs > 0`), the `completion_probabilities` in the response will contain actual probabilities (0.0 to 1.0) for tokens instead of log-probabilities. Additionally, the field `top_logprobs` in the response will be replaced by `top_probs`. Essentially, this converts the reported log probabilities to normalized probabilities after all sampling adjustments have been applied.
    /// Default: `false`.
    #[serde(default, skip_serializing_if = "<&bool>::not")]
    #[builder(default)]
    pub post_sampling_probs: bool,

    /// A list of specific fields to include in the response JSON.
    ///
    /// By default, the server returns a standard set of fields (e.g. `content`, `tokens`, `stop_reason`, etc.). Using `response_fields`, you can request only certain fields or additional fields that are normally omitted. For example:
    /// ```json
    /// "response_fields": ["content", "generation_settings/n_predict"]
    /// ```
    /// would include the generated content and also pull out the `n_predict` field from the internal generation settings into the top-level response (renaming it as needed). Fields with slashes (`generation_settings/n_predict`) indicate nested fields to extract. If a requested field is not available in the response, it will simply be omitted without error.
    /// Default: None (the server will return its default set of fields).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_fields: Option<Vec<String>>,

    /// LoRA adapters to apply to this request.
    ///
    /// Provide a list of LoRA adapter configurations, each with an `id` (referring to a loaded LoRA by its index or identifier) and a `scale` (the strength with which to apply that adapter). This allows modifying the base model's behavior on-the-fly if LoRAs are available.
    /// If a LoRA is not listed here, it will be assumed to have a scale of 0. Example usage: `[{ "id": 0, "scale": 0.5}, { "id": 1, "scale": 1.1 }]` would apply two LoRA adapters with specified scales.
    /// *Note:* Requests that use different sets of LoRA adapters will not be batchable together by the server (it may reduce throughput if each request has unique LoRA settings).
    /// Default: None (no LoRA adapters applied, or all scales are 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<Vec<LoraAdapter>>,

    /// All sampling‑related knobs (temperature, top‑k, penalties, etc.).
    ///
    /// The fields of `SamplingParams` are *flattened* so the public JSON
    /// schema and the builder API remain perfectly flat and backward‑compatible.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingParams>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct CompletionResponse {
    /// Completion result as a single string (excluding any stopping word).
    /// In streaming mode, this may contain only the **next** token instead of the full completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Raw generated token IDs corresponding to the completion.
    /// Populated only if `return_tokens` or `stream` was true in the request (otherwise None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u64>>,

    /// Sequence index of this completion in a batch (if multiple completions were requested).
    /// This is None for single-completion requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,

    /// ID of the server slot used for this completion (if using a multi-slot llama.cpp server).
    /// Helps identify which context/kv-cache slot was utilized. None if not applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<usize>,

    /// **Streaming flag** – `true` once generation has stopped, only relevant in streaming mode.
    /// In non-streaming usage this may remain None or false. (Not to be confused with the request's stopping criteria.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<bool>,

    /// Per-token log-probability data for the generated tokens.
    /// Contains one entry per predicted token (length should equal the number of tokens generated).
    /// If `post_sampling_probs` was requested, probabilities are given instead of log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_probabilities: Option<Vec<TokenProbability>>,

    /// Name or identifier of the model that produced this completion.
    /// Typically the model alias (for the actual model path, see server properties if needed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The prompt used for this request, after any preprocessing.
    /// This may include inserted special tokens or formatting (e.g., BOS token or system prompt) as applied by the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Prompt>,

    /// Generation parameters and settings that were used.
    /// This includes all options (like `n_predict`, sampling settings, etc.) except the prompt itself.
    /// *Note:* We keep this as a bundled `GenerationSettings` struct; its sub-fields are not expanded here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_settings: Option<GenerationSettings>,

    /// Timing information for this request and completion (in milliseconds and tokens/second rates).
    /// Provides performance metrics such as prompt processing time and prediction throughput.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,

    /// Indicates how/why the generation stopped. Possible values include:
    /// - `"none"`: Generation has not stopped (still in progress).
    /// - `"eos"`: Stopped upon encountering the end-of-sequence token.
    /// - `"limit"`: Stopped because the maximum number of tokens (`n_predict`) was reached before an EOS or stop word.
    /// - `"word"`: Stopped upon encountering a user-provided stopping **word/sequence**.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_type: Option<String>,

    /// The specific stopping word (or sequence) that triggered the stop condition.
    /// This will be an empty string if generation stopped due to EOS or token limit rather than a stop word.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopping_word: Option<String>,

    /// Whether the last generated token was a newline (line break).
    /// If true, the content ends with a newline character (which might be auto-appended by the model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_new_line: Option<bool>,

    /// Number of prompt tokens reused from a previous cache (prompt overlap).
    /// This indicates how many tokens from the prompt were **not** re-evaluated because they were loaded from the model's KV cache (i.e., `n_past` optimization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_cached: Option<usize>,

    /// Number of prompt tokens that were actually evaluated for this request.
    /// This is essentially the prompt length minus any `tokens_cached`. It represents how many prompt tokens had to be processed by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_evaluated: Option<usize>,

    /// Number of tokens that were generated/predicted in this completion (excluding the prompt).
    /// This is the length of the generated content in tokens. It may equal the length of `tokens` vector if that is returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_predicted: Option<usize>,

    /// `true` if the context was truncated due to length limits.
    /// If the sum of prompt tokens (`tokens_evaluated`) and generated tokens exceeded the model context `n_ctx`, the prompt was truncated and this flag is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct TokenProbability {
    /// Token ID of the predicted token.
    pub id: usize,

    /// Natural log probability of this token (if `post_sampling_probs` is **false**).
    /// This may be absent (None) if only post-sampling probabilities were requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,

    /// Linear probability (0.0–1.0) of this token (if `post_sampling_probs` is **true**).
    /// This field is present only when probabilities were requested instead of log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prob: Option<f32>,

    /// Text of the token (decoded string form).
    pub token: String,

    /// Byte representation of the token.
    pub bytes: Vec<u8>,

    /// Top-N alternate tokens *before* sampling, with their log probabilities.
    /// Only populated if `n_probs` > 0 and `post_sampling_probs` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopTokenProbability>>,

    /// Top-N alternate tokens *before* sampling, with linear probabilities.
    /// This is populated if `n_probs` > 0 and `post_sampling_probs` is true (same concept as `top_logprobs` but after conversion to probs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_probs: Option<Vec<TopTokenProbability>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct TopTokenProbability {
    /// Token ID of the alternate prediction.
    pub id: usize,

    /// Log probability of this token (absent if using linear probabilities).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,

    /// Linear probability (0.0–1.0) for this token (present when using post-sampling probabilities).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prob: Option<f32>,

    /// Text of the token alternative.
    pub token: String,

    /// Byte sequence for the token alternative.
    pub bytes: Vec<u8>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct Timings {
    /// Total time spent generating predicted tokens (milliseconds).
    pub predicted_ms: f32,

    /// Average time per prompt token (milliseconds/token) for prompt processing.
    pub prompt_per_token_ms: f32,

    /// Average time per generated token (milliseconds/token) during generation.
    pub predicted_per_token_ms: f32,

    /// Total time spent processing the prompt (milliseconds).
    pub prompt_ms: f32,

    /// Throughput of prompt processing (tokens per second).
    pub prompt_per_second: f32,

    /// Number of tokens generated (predicted) in this completion.
    pub predicted_n: f32,

    /// Number of tokens in the prompt that were processed.
    pub prompt_n: f32,

    /// Throughput of generation (predicted tokens per second).
    pub predicted_per_second: f32,
}
