use crate::local_models::gguf::tools::gguf_tensors::GgmlDType;

// This is converted from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/main/LLMVRAMCalculator/LLMVRAMCalculator.py
// Estimates from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/70c0241bc90e8025218a8d9667346aa72f60f472/LLMVRAMCalculator/LLMVRAMCalculator.py#L6
// also see https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763
pub fn estimate_context_size(
    ctx_size: usize,
    embedding_length: usize,
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    batch_size: Option<usize>,
) -> usize {
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
    context_bytes as usize
}

// pub fn estimate_context_size(
//     ctx_size: u64,
//     embedding_length: u64,
//     head_count: u64,
//     head_count_kv: u64,
//     block_count: u64,
// ) -> u64 {
//     2 * ctx_size * block_count * ((embedding_length / head_count) * 2) * head_count_kv
// }

pub fn estimate_quantization_level(
    params: f64,
    available_ram_bytes: usize,
    ctx_memory_size_bytes: usize,
) -> crate::Result<u8> {
    let available_memory_bytes = available_ram_bytes - ctx_memory_size_bytes;

    let quantization_types = [
        GgmlDType::Q8_0,
        GgmlDType::Q6K,
        GgmlDType::Q5_1,
        GgmlDType::Q4_0,
        GgmlDType::Q3K,
        GgmlDType::Q2K,
    ];
    for &dtype in &quantization_types {
        if available_memory_bytes >= estimate_model_size_dtype(params, dtype) as usize {
            return Ok(match dtype {
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

pub(crate) fn estimate_model_size_dtype(params: f64, dtype: GgmlDType) -> f64 {
    let size = params * dtype.bits_per_weight() / 8.0;
    size
}

pub(crate) fn estimate_model_size_level(params: f64, level: u8) -> f64 {
    let dtype = match level {
        8 => GgmlDType::Q8_0,
        6 => GgmlDType::Q6K,
        5 => GgmlDType::Q5_1,
        4 => GgmlDType::Q4_0,
        3 => GgmlDType::Q3K,
        2 => GgmlDType::Q2K,
        1 => GgmlDType::Q2K,
        _ => panic!("Invalid quantization level: {level}"),
    };

    let size = params * dtype.bits_per_weight() / 8.0;
    size
}
