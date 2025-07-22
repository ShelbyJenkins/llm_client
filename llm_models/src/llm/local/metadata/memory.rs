use crate::llm::{base::DEFAULT_CONTEXT_LENGTH, local::gguf::gguf_tensors::GgmlDType};

// This is converted from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/main/LLMVRAMCalculator/LLMVRAMCalculator.py
// Estimates from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/70c0241bc90e8025218a8d9667346aa72f60f472/LLMVRAMCalculator/LLMVRAMCalculator.py#L6
// also see https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763
pub fn estimate_context_size_bytes(
    ctx_size: Option<u64>,
    embedding_length: u64,
    head_count: u64,
    head_count_kv: u64,
    block_count: u64,
    batch_size: Option<u64>,
) -> u64 {
    let ctx_size = ctx_size.unwrap_or(DEFAULT_CONTEXT_LENGTH);

    let batch_size = batch_size.unwrap_or(512);
    // Input buffer
    let input_buffer =
        ((batch_size * 3) + (embedding_length * batch_size) + (batch_size * ctx_size) + ctx_size)
            as f64;
    // Compute buffer
    let compute_buffer =
        (ctx_size as f64 / 1024f64 * 2f64 + 0.75) * head_count as f64 * 1024f64 * 1024f64;
    // Key-value cache
    let cache_bit = 16;
    let gqa = head_count / head_count_kv;
    let n_embd_gqa = embedding_length / gqa;
    let n_elements = n_embd_gqa * (block_count * ctx_size);
    let size = 2 * n_elements;
    let kv_cache = size as f64 * (cache_bit as f64 / 8f64);

    let context_bytes = input_buffer + kv_cache + compute_buffer;
    context_bytes as u64
}

// pub fn estimate_context_size_bytes(
//     ctx_size: u64,
//     embedding_length: u64,
//     head_count: u64,
//     head_count_kv: u64,
//     block_count: u64,
// ) -> u64 {
//     2 * ctx_size * block_count * ((embedding_length / head_count) * 2) * head_count_kv
// }

pub fn estimate_max_quantization_level(
    params_in_billions: f64,
    use_memory_bytes: u64,
    ctx_memory_size_bytes: u64,
) -> crate::Result<u8> {
    let available_memory_bytes = use_memory_bytes - ctx_memory_size_bytes;

    let quantization_types = [
        GgmlDType::Q8_0,
        GgmlDType::Q6K,
        GgmlDType::Q5_1,
        GgmlDType::Q4_0,
        GgmlDType::Q3K,
        GgmlDType::Q2K,
    ];
    let params = (params_in_billions * 1_000_000_000.0) as u64;
    for &ggml_d_type in &quantization_types {
        if available_memory_bytes >= estimate_model_size_dtype(params, &ggml_d_type) as u64 {
            return Ok(match ggml_d_type {
                GgmlDType::Q8_0 | GgmlDType::Q8K => 8,
                GgmlDType::Q6K => 6,
                GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q5K => 5,
                GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q4K => 4,
                GgmlDType::Q3K => 3,
                GgmlDType::Q2K => 2,
                _ => unreachable!(),
            });
        }
    }

    crate::bail!("Not enough VRAM!")
}

pub fn estimate_quant_memory_usage_bytes(
    params_in_billions: f64,
    ctx_memory_size_bytes: u64,
    ggml_d_type: GgmlDType,
) -> u64 {
    let params = (params_in_billions * 1_000_000_000.0) as u64;
    estimate_model_size_dtype(params, &ggml_d_type) as u64 + ctx_memory_size_bytes
}

pub fn estimate_memory_usage_range_bytes(
    params_in_billions: f64,
    ctx_memory_size_bytes: u64,
) -> Vec<u64> {
    let quantization_types = [
        GgmlDType::Q8_0,
        GgmlDType::Q6K,
        GgmlDType::Q5_1,
        GgmlDType::Q4_0,
        GgmlDType::Q3K,
        GgmlDType::Q2K,
    ];
    let params = (params_in_billions * 1_000_000_000.0) as u64;
    let mut memory_usage_range = Vec::new();
    for &ggml_d_type in &quantization_types {
        let memory_usage = estimate_model_size_dtype(params, &ggml_d_type) as u64;
        memory_usage_range.push(memory_usage + ctx_memory_size_bytes);
    }

    memory_usage_range
}

pub fn estimate_model_size_dtype(number_of_parameters: u64, ggml_d_type: &GgmlDType) -> f64 {
    let size = number_of_parameters as f64 * ggml_d_type.bits_per_weight() / 8.0;
    size
}
