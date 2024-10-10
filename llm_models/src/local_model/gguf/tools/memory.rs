use std::{
    cmp::{max, min},
    vec::Vec,
};

pub struct MemoryEstimate {
    // How many layers we predict we can load
    pub layers: i32,
    // The size of the graph which occupies the main GPU
    pub graph: u64,
    // The total size of the model if loaded into VRAM. If all layers are loaded, vram_size == total_size
    pub total_size: u64,
}

pub fn estimate_gpu_layers(
    gguf: &super::GgufModel,
    opts: Options,
) -> crate::Result<MemoryEstimate> {
    // Graph size for a partial offload, applies to all GPUs
    let mut graph_partial_offload: u64 = 0;
    // Graph size when all layers are offloaded, applies to all GPUs
    let mut graph_full_offload: u64 = 0;
    // Final graph offload once we know full or partial
    let mut graph_offload: u64 = 0;
    // Conditional output size on GPU 0
    let mut memory_layer_output: u64 = 0;
    // The sizes of a layer
    let mut layer_size: u64 = 0;
    // The sum of all the layer sizes (just for logging)
    let mut memory_weights: u64 = 0;
    // True if all the layers are loaded
    let mut fully_loaded: bool = false;
    // Overflow that didn't fit into the GPU
    let mut overflow: u64 = 0;
    let mut memory_weights: u64 = 0;

    // add one layer worth of memory as a buffer
    layer_size = gguf.layers.get("blk.0").map_or_else(
        || {
            eprintln!("model missing blk.0 layer size");
            0
        },
        |blk0| blk0.size(),
    );

    // fp16 k,v = sizeof(float16) * n_ctx * n_layer * (n_embd_head_k + n_embd_head_v) * n_head_kv
    let kv: u64 = 2
        * opts.num_ctx as u64
        * gguf.metadata.block_count()?
        * (gguf.metadata.embedding_head_count_k()? + gguf.metadata.embedding_head_count_v()?)
        * gguf.metadata.head_count_kv()?;

    // KV is proportional to the number of layers
    layer_size += kv / gguf.metadata.block_count()?;

    let (graph_partial_offload, graph_full_offload) = graph_size(
        gguf,
        opts.num_ctx as u64,
        min(opts.num_ctx, opts.num_batch) as u64,
    )?;

    graph_partial_offload = if graph_partial_offload == 0 {
        gguf.metadata.gqa()? * kv / 6
    } else {
        graph_partial_offload
    };

    graph_full_offload = if graph_full_offload == 0 {
        graph_partial_offload
    } else {
        graph_full_offload
    };

    memory_layer_output += gguf
        .layers
        .get("output_norm")
        .map_or(0, |layer| layer.size());
    memory_layer_output += gguf.layers.get("output").map_or_else(
        || {
            gguf.layers
                .get("token_embd")
                .map_or(0, |layer| layer.size())
        },
        |layer| layer.size(),
    );

    let mut layer_count = 0;
    let mut layer_counts = vec![0; gpus.len()];
    let mut gpu_allocations = vec![0; gpus.len()];

    let gpus_with_space: Vec<GpuWithSpace> = gpus
        .into_iter()
        .enumerate()
        .filter(|(i, gpu)| {
            let available = gpu.free_memory as i64
                - overhead as i64
                - if *i == 0 { gpu_zero_overhead as i64 } else { 0 };
            available > layer_size as i64
        })
        .map(|(i, gpu)| GpuWithSpace { index: i, gpu })
        .collect();

    let mut gpus_with_space: Vec<GpuWithSpace> = Vec::new();

    // For all the layers, find where they can fit on the GPU(s)
    for i in 0..gguf.metadata.block_count()? {
        // Some models have inconsistent layer sizes
        if let Some(blk) = gguf.layers.get(&format!("blk.{}", i)) {
            layer_size = blk.size();
            layer_size += kv / gguf.metadata.block_count()?;
        }

        layer_count += 1;
    }

    fully_loaded = layer_count >= gguf.metadata.block_count()? as i32;
    if !fully_loaded {
        for _ in layer_count..gguf.metadata.block_count()? as i32 {
            overflow += layer_size;
        }
    }

    // Determine if we need to consider output then find where it fits
    if memory_layer_output > 0 && (opts.num_gpu < 0 || layer_count < opts.num_gpu) {
        let mut j = gpus_with_space.len();
        while j > 0 {
            let g = &gpus_with_space[layer_count % j];
            let used = gpu_allocations[g.index] + max(graph_partial_offload, graph_full_offload);
            if (g.gpu.free_memory as i64 - overhead as i64) > (used + memory_layer_output) as i64 {
                gpu_allocations[g.index] += memory_layer_output;
                layer_counts[g.index] += 1;
                layer_count += 1;
                break;
            }
            j -= 1;
        }
        if layer_count < gguf.metadata.block_count()? as i32 + 1 {
            fully_loaded = false;
            overflow += memory_layer_output;
        }
    }

    // Add the applicable (full or partial) graph allocations
    for i in 0..gpus.len() {
        if layer_counts[i] <= 0 {
            continue;
        }
        if fully_loaded {
            gpu_allocations[i] += graph_full_offload;
        } else {
            gpu_allocations[i] += graph_partial_offload;
        }
    }

    let graph_offload = if fully_loaded {
        graph_full_offload
    } else {
        graph_partial_offload
    };

    // Summaries for the log
    let memory_required_partial: u64 = gpu_allocations.iter().sum();
    let memory_required_total = memory_required_partial + overflow;

    let estimate = MemoryEstimate {
        total_size: memory_required_total,
        layers: layer_count,
        graph: graph_offload,
    };

    Ok(estimate)
}

fn graph_size(gguf: &super::GgufModel, context: u64, batch: u64) -> crate::Result<(u64, u64)> {
    let embedding = gguf.metadata.embedding_length()?;
    let heads = gguf.metadata.head_count()?;
    let heads_kv = gguf.metadata.head_count_kv()?;
    let vocab = gguf.metadata.tokens()?.len() as u64;

    let embedding_heads = gguf.metadata.embedding_head_count()?;
    let embedding_heads_k = gguf.metadata.embedding_head_count_k()?;

    let (mut partial_offload, mut full_offload) = (0, 0);

    match gguf.metadata.architecture()?.as_str() {
        "llama" => {
            full_offload = max(
                4 * batch * (1 + 4 * embedding + context * (1 + heads)),
                4 * batch * (embedding + vocab),
            );

            partial_offload = 4 * batch * embedding;
            partial_offload += max(
                4 * batch * (1 + embedding + max(context, embedding))
                    + embedding * embedding * 9 / 16
                    + 4 * context * (batch * heads + embedding_heads * heads_kv),
                4 * batch * (embedding + vocab) + embedding * vocab * 105 / 128,
            );

            if let Some(ffn_gate_exps_weight) = gguf
                .layers
                .get("blk.0")
                .and_then(|blk| blk.get("ffn_gate_exps.weight"))
            {
                // mixtral 8x22b
                let ff = gguf.metadata.feed_forward_length()?;
                partial_offload = max(
                    3 * ffn_gate_exps_weight.size()
                        + 4 * batch
                            * (2 * ff
                                + heads_kv
                                + embedding
                                + context
                                + embedding_heads * heads_kv),
                    4 * (context * batch * heads
                        + context * embedding_heads * heads_kv
                        + batch * 1024
                        + embedding_heads * heads_kv * batch),
                );
            } else if let Some(ffn_gate_weight) = gguf
                .layers
                .get("blk.0")
                .and_then(|blk| blk.get("ffn_gate.0.weight"))
            {
                // mixtral 8x7b
                let ffn_gate_weight1 = ffn_gate_weight.shape[1] as u64;
                full_offload = 4
                    * batch
                    * (2 + 3 * embedding + context * (1 + heads) + 2 * heads_kv + ffn_gate_weight1);
                partial_offload = max(
                    4 * batch
                        * (3 + embedding_heads * heads_kv
                            + embedding
                            + context * (1 + heads)
                            + ffn_gate_weight1)
                        + (embedding * embedding + 3 * embedding * heads_kv * ffn_gate_weight1) * 9
                            / 16,
                    4 * batch * (1 + 2 * embedding + context * (1 + heads))
                        + embedding * (6 * context * heads_kv / heads + embedding * 9 / 16),
                );
            }
        }
        "gemma" | "gemma2" => {
            full_offload = max(
                4 * batch * (embedding + vocab),
                4 * batch
                    * (2 + context
                        + context * heads
                        + 2 * embedding
                        + 2 * embedding_heads_k * heads),
            );

            partial_offload = max(
                4 * embedding * batch + embedding * vocab * 105 / 128 + 4 * vocab * batch,
                4 * batch
                    * (2 * embedding
                        + 1
                        + 2 * embedding_heads_k * heads
                        + context
                        + context * heads)
                    + 4 * embedding_heads_k * context * 8
                    + embedding * embedding_heads_k * heads * 9 / 16,
            );
        }
        "command-r" => {
            full_offload = max(
                4 * batch * (embedding + vocab),
                4 * batch * (2 + 4 * embedding + context * (1 + heads)),
            );

            partial_offload = max(
                4 * batch * (embedding + vocab) + embedding * vocab * 105 / 128,
                4 * batch * (1 + 2 * embedding + context * (1 + heads))
                    + 4 * embedding * context
                    + embedding * embedding * 9 / 16,
            );
        }
        "qwen2" => {
            full_offload = max(
                4 * batch * (embedding + vocab),
                4 * batch * (1 + 2 * embedding + context + context * heads),
            );

            partial_offload = max(
                4 * batch * (embedding + vocab) + embedding * vocab * 105 / 128,
                4 * (batch * (1 + 2 * embedding + context * (1 + heads))
                    + embedding * (1 + context)),
            );
        }
        "phi2" => {
            full_offload = max(
                4 * batch * (embedding + vocab),
                4 * batch * (1 + 4 * embedding + context + context * heads),
            );

            partial_offload = max(
                4 * batch * (2 * embedding + vocab) + embedding * vocab * 105 / 128,
                4 * batch * (2 + 3 * embedding + context + context * heads),
            );
        }
        "stablelm" => {
            full_offload = 4 * batch * (context * (1 + heads) + 3 * embedding + 2);
            partial_offload = max(4 * batch * (vocab + 2 * embedding), full_offload);
        }
        "deepseek2" => {
            full_offload = max(
                4 * batch * (3 * embedding + vocab),
                4 * batch
                    * (3 * embedding
                        + 2
                        + context * (1 + heads_kv)
                        + 2 * embedding_heads_k * heads_kv),
            );

            partial_offload = max(
                4 * batch * (3 * embedding + vocab) + embedding * vocab * 105 / 128,
                4 * batch
                    * (2 * embedding
                        + 1
                        + 2 * embedding_heads_k * heads_kv
                        + context
                        + context * heads_kv)
                    + 4 * embedding_heads_k * context * heads_kv
                    + embedding * embedding_heads_k * heads_kv * 9 / 16,
            );
        }
        "chatglm" => {
            full_offload = 4 * batch * (embedding + vocab);
            partial_offload = 4 * batch * (embedding + vocab) + embedding * vocab * 105 / 128;
            if let Some(qkv_bias) = gguf
                .layers
                .get("blk.0")
                .and_then(|blk| blk.get("attn_qkv.bias"))
            {
                full_offload = max(
                    full_offload,
                    4 * batch
                        * (2 + 2 * embedding
                            + context
                            + context * heads
                            + embedding_heads_k * heads
                            + qkv_bias.shape[0] as u64),
                );

                partial_offload = max(
                    partial_offload,
                    4 * batch
                        * (1 + 2 * embedding
                            + embedding_heads_k * heads
                            + context
                            + context * heads)
                        + 4 * embedding_heads_k * context
                        + 4 * context * embedding_heads_k
                        + 4 * qkv_bias.shape[0] as u64,
                );
            }
        }
        _ => {}
    }

    Ok((partial_offload, full_offload))
}
