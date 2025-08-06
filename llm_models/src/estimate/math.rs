//! ## Memory–sizing helpers for transformer inference
//!
//! This module groups a set of **light‑weight, allocation‑free helpers** used
//! by the runtime scheduler to answer three practical questions:
//!
//! 1. **How much device memory do we need for the *context‑dependent* state?**  
//!    (`estimate_context_bytes`)  
//!    *Context* covers the KV‑cache and per‑GPU scratch buffers, and therefore
//!    grows with sequence length, batch size, and MoE router `top‑k`.
//!
//! 2. **How big is a single transformer block once the context cost is spread
//!    over all blocks?**  
//!    (`average_layer_size_bytes`, `average_layer_size_bytes_with_moe`)  
//!    These helpers return a **conservative upper bound** so the scheduler can
//!    place blocks on devices without running out of memory at runtime.
//!
//! 3. **Which tensor precision (ggml file‑type) can we afford on our target
//!    devices?**  
//!    (`find_largest_compatible_dense_type`, `find_largest_compatible_moe_type`)
//!
//! ### Formulas (summary)
//!
//! ```text
//! ───────────────────────────────────────── context bytes ──────────────
//! ContextBytes = KV_total + Σ_i M_comp(i)
//!
//! KV_total = 2 · (E / G) · L · C · B · (bits_kv / 8)
//!   G = N_head / N_head_kv
//!
//! M_comp(i) ≈ ([(C / 1024)·2 + 0.75] · N_head · 2²⁰ · scale)
//!   scale = max(B,128)/128 · k   (k = router top‑k; dense ⇒ 1)
//!
//! ─────────────────── layer‑average helpers (rounded **up**) ───────────
//! Dense‑only: ⌈(total_blocks_bytes + context_size_bytes) / block_count⌉
//!
//! With MoE:
//!   dense_blocks_avg   = ctx_per_block + dense_share_per_block
//!   experts_blocks_avg = dense_blocks_avg + expert_share_per_moe
//!
//!   ctx_per_block         = ⌈context_size_bytes / block_count⌉
//!   dense_share_per_block = ⌈dense_tensor_bytes / block_count⌉
//!   expert_share_per_moe  = ⌈expert_blocks_bytes / experts_block_count⌉
//!   dense_tensor_bytes    = total_blocks_bytes − expert_blocks_bytes
//! ```
//!
//! ### Design notes
//! * **Static model weights are *out of scope*.**  Only `average_layer_*`
//!   helpers touch parameter tensors, and even there they treat the totals as
//!   opaque byte counts provided by the caller.
//! * All helpers either `assert!` invalid inputs (fail‑fast) or return
//!   `Result< _, MathProblem >` with overflow information, matching the
//!   surrounding codebase’s error philosophy.
//! * Integer divisions use **ceiling semantics** to guarantee that the returned
//!   size never under‑estimates the true requirement.
//!
//! These utilities have no external dependencies beyond `core` and are safe to
//! call in const‑contexts or host‑side build scripts.

use crate::{
    estimate::memory::{DeviceMemSpec, DeviceTypeSpec, RuntimeMemorySpec},
    manifest::file_encoding_type::GgmlFileType,
};

#[derive(Debug, thiserror::Error)]
pub enum MathProblem {
    #[error("memory estimate overflow: {0}")]
    MemoryEstimateOverflow(f64),

    #[error("missing required model profile parameter: {0}")]
    MissingRequiredParameter(&'static str),
}

/// Estimates the **context‑dependent GPU memory** (in **bytes**) required
/// by a transformer **at runtime**—that is, everything *except* the static
/// model / expert weights.  The estimate covers  
/// * the **KV‑cache** that grows with sequence length and batch size, and  
/// * the **scratch / compute buffers** allocated on each *compute* device
///   during attention & FFN execution.
///
/// # Formula
/// ```text
/// ContextBytes = KV_total + Σ_{compute i}  M_comp(i)
///
/// KV_total  = 2 · (E / G) · L · C · B · (bits_kv / 8)
///               └─────────── KV elements ───────────┘
///   where  G = N_head / N_head_kv      (grouped‑query factor)
///
///   • E  – embedding length          (`n_embd`)
///   • L  – block / layer count       (`block_count`)
///   • C  – context length            (`runtime.inference_ctx_size`
///                                      or `model_ctx_size`)
///   • B  – logical batch size        (`runtime.batch_size`)
///   • bits_kv – **average** bits per KV element  
///               (`kv_cache_type.bits_per_weight()`; may be fractional)
///
/// KV_total is stored **once** if `runtime.shard_kv == true`;  
/// otherwise it is duplicated on every *compute* GPU.
///
/// ────────────────────────────────────────────────────────────────────
/// M_comp(i) – scratch / compute‑buffer bytes for *compute* GPU _i_
///
///   • If `device_specs[i].compute_buffer_bytes` is `Some(v)`, use *v*.
///   • Else use the heuristic
///
///       M_comp(i) ≈ ( [(C / 1024) · 2  +  0.75] · N_head · 2²⁰ · scale )
///
///       with   scale = (max(B, 128) / 128) · k
///              k     = `expert_used_count` (router **top‑k**; 1 for dense)
///
/// Devices tagged as `MoeOffload` **do not** participate in this estimate:
/// they store only expert weights and have no KV or scratch memory.
///
/// # Notes
/// * Fractional `bits_kv` (e.g. 4.5 bits for some block‑quant formats)
///   are fully supported; the final byte total is `ceil`‑ed.
/// * The function performs all arithmetic in `f64`, then validates that
///   the result fits in `u64`.
pub fn estimate_context_size_bytes_checked(
    model_ctx_size: Option<u64>,
    n_embd: Option<u64>,
    n_head: Option<u64>,
    n_head_kv: Option<u64>,
    block_count: u64,
    expert_used_count: Option<u64>,
    mem_spec: &RuntimeMemorySpec,
) -> Result<u64, MathProblem> {
    // ── unwrap required parameters, failing fast if any are missing ──────────
    let model_ctx_size =
        model_ctx_size.ok_or(MathProblem::MissingRequiredParameter("model_ctx_size"))?;
    let n_embd = n_embd.ok_or(MathProblem::MissingRequiredParameter("n_embd"))?;
    let n_head = n_head.ok_or(MathProblem::MissingRequiredParameter("n_head"))?;
    estimate_context_size_bytes(
        model_ctx_size,
        n_embd,
        n_head,
        n_head_kv,
        block_count,
        expert_used_count,
        mem_spec,
    )
}

pub fn estimate_context_size_bytes(
    model_ctx_size: u64,
    n_embd: u64,
    n_head: u64,
    n_head_kv: Option<u64>,
    block_count: u64,
    expert_used_count: Option<u64>,
    mem_spec: &RuntimeMemorySpec,
) -> Result<u64, MathProblem> {
    // ── Derive list of *compute* devices ──────────────────────────────
    let compute_devices: Vec<&DeviceMemSpec> = mem_spec
        .device_specs
        .iter()
        .filter_map(|d| match d {
            // ignore MoE‑offload
            DeviceTypeSpec::Compute(spec) => Some(spec),
            _ => None,
        })
        .collect();

    // Fallback: assume one unnamed compute device.
    let gpu_count = if compute_devices.is_empty() {
        1
    } else {
        compute_devices.len()
    } as f64;

    // ── Field shorthands ──────────────────────────────────────────────
    let c = mem_spec.inference_ctx_size.unwrap_or(model_ctx_size) as f64;
    let b = mem_spec.batch_size as f64;
    let e = n_embd as f64;
    let nh = n_head as f64;
    let nk = n_head_kv.unwrap_or(n_head) as f64;
    let g = nh / nk;
    let l = block_count as f64;
    let k = expert_used_count.unwrap_or(1) as f64;

    // ── Decode element width ─────────────────────────────────────────
    let bits_kv = mem_spec.kv_cache_type.bits_per_weight();

    // ── KV‑cache bytes ───────────────────────────────────────────────
    let kv_elems = 2.0 * (e / g) * l * c * b;
    let kv_bytes_one = kv_elems * (bits_kv / 8.0);
    let kv_bytes_total = if mem_spec.shard_kv {
        kv_bytes_one
    } else {
        kv_bytes_one * gpu_count
    };

    // ── Scratch / compute buffers (compute GPUs only) ────────────────
    let scratch_total: f64 = if compute_devices.is_empty() {
        // Heuristic for the implicit single GPU.
        let scale = k * (b.max(128.0) / 128.0);
        (((c / 1024.0) * 2.0) + 0.75) * nh * 1_048_576.0 * scale
    } else {
        compute_devices
            .iter()
            .map(|spec| {
                if let Some(bytes) = spec.compute_buffer_bytes {
                    bytes as f64
                } else {
                    let scale = k * (b.max(128.0) / 128.0);
                    (((c / 1024.0) * 2.0) + 0.75) * nh * 1_048_576.0 * scale
                }
            })
            .sum()
    };

    // ── Final check & cast ───────────────────────────────────────────
    let total = kv_bytes_total + scratch_total;
    if total > u64::MAX as f64 {
        return Err(MathProblem::MemoryEstimateOverflow(total));
    }
    Ok(total.ceil() as u64)
}

/// Returns a **conservative (rounded‑up) average** of the total
/// *parameter‑and‑context* bytes per transformer block in a **uniform,
/// dense‑only** model—i.e. one where every block carries the same tensor
/// shapes.
///
/// This helper is the dense‑only counterpart to
/// `average_layer_size_bytes_with_moe`.  It uses **ceiling division** so
/// the caller reserves at least as much memory as any single block
/// requires.
pub fn average_layer_size_bytes(
    block_count: u64,
    total_blocks_bytes: u64,
    context_size_bytes: u64,
) -> Result<u64, MathProblem> {
    // ── Pre‑condition ────────────────────────────────────────────────
    assert!(block_count > 0, "block_count must be positive");

    // ── Safe ceiling‑division: ⌈(A + B) / N⌉ ─────────────────────────
    // 1. sum = A + B               (checked to avoid wraparound)
    // 2. num = sum + (N - 1)       (also checked)
    // 3. avg = num / N             (integer division → ceiling)
    let sum = total_blocks_bytes.checked_add(context_size_bytes).ok_or(
        MathProblem::MemoryEstimateOverflow(
            (total_blocks_bytes as f64) + (context_size_bytes as f64),
        ),
    )?;

    let num = sum
        .checked_add(block_count - 1) // block_count > 0 is guaranteed
        .ok_or(MathProblem::MemoryEstimateOverflow(
            (sum as f64) + ((block_count - 1) as f64),
        ))?;

    Ok(num / block_count)
}

/// Calculates an **upper‑bound** estimate of the average memory (bytes)
/// carried by
/// * a **dense‑only block**, and  
/// * a **MoE block** (dense sub‑layers + experts)  
/// in a heterogeneous transformer.
///
/// The estimate is conservative: each division is rounded **up** so the
/// caller can safely reserve at least that much memory per assigned block.
pub fn average_layer_size_bytes_with_moe(
    block_count: u64,
    total_blocks_bytes: u64,
    context_size_bytes: u64,
    experts_block_count: u64,
    expert_blocks_bytes: u64,
) -> (
    u64, /* dense_blocks_avg */
    u64, /* experts_blocks_avg */
) {
    // ── Preconditions (fail‑fast) ────────────────────────────────────
    assert!(block_count > 0, "block_count must be positive");
    assert!(
        experts_block_count > 0,
        "experts_block_count must be positive"
    );
    assert!(
        block_count >= experts_block_count,
        "block_count must be ≥ experts_block_count"
    );
    assert!(
        total_blocks_bytes >= expert_blocks_bytes,
        "total_blocks_bytes must be ≥ expert_blocks_bytes"
    );

    // ── Shared & per‑block quantities (rounded *up*) ─────────────────
    let dense_tensor_bytes = total_blocks_bytes - expert_blocks_bytes;

    let ctx_per_block = context_size_bytes.saturating_add(block_count - 1) / block_count;

    let dense_share_per_block = dense_tensor_bytes.saturating_add(block_count - 1) / block_count;

    let expert_share_per_moe =
        expert_blocks_bytes.saturating_add(experts_block_count - 1) / experts_block_count;

    // ── Final upper‑bound averages ───────────────────────────────────
    let experts_blocks_avg = expert_share_per_moe + ctx_per_block + dense_share_per_block;
    let dense_blocks_avg = dense_share_per_block + ctx_per_block;

    (dense_blocks_avg, experts_blocks_avg)
}

/// Selects the **highest‑precision** dense tensor format that still fits
/// into the *compute* device’s memory budget.
pub fn find_largest_compatible_dense_type(
    available_memory_bytes_compute: u64,
    candidates: &[(GgmlFileType, u64 /* estimated_total_bytes */)],
) -> Option<GgmlFileType> {
    assert!(
        available_memory_bytes_compute > 0,
        "available_memory_bytes_compute must be positive"
    );
    assert!(!candidates.is_empty(), "candidates must not be empty");

    candidates
        .iter()
        .filter(|(_, est)| *est <= available_memory_bytes_compute)
        .max_by_key(|(_, est)| *est)
        .map(|(ty, _)| *ty)
}

/// Selects the **highest‑precision** tensor format for a model that
/// contains MoE blocks, subject to *two* memory budgets:
/// * a **compute** device (runs attention, FFN, router) and  
/// * a **MoE‑offload** device (stores all expert weights).
///
/// The estimate for each candidate is assumed to be:  
/// `total_bytes = dense_bytes + expert_blocks_bytes`  
/// where `expert_blocks_bytes` is constant across candidates (i.e. expert
/// weights are kept in the **same** precision irrespective of the tensor
/// format used for dense layers).
pub fn find_largest_compatible_moe_type(
    available_memory_bytes_compute: u64,
    available_memory_bytes_moe_offload: u64,
    expert_blocks_bytes: u64,
    candidates: &[(GgmlFileType, u64 /* estimated_total_bytes */)],
) -> Option<GgmlFileType> {
    assert!(
        available_memory_bytes_compute > 0,
        "available_memory_bytes_compute must be positive"
    );
    assert!(
        available_memory_bytes_moe_offload > 0,
        "available_memory_bytes_moe_offload must be positive"
    );
    assert!(
        expert_blocks_bytes > 0,
        "expert_blocks_bytes must be positive"
    );
    assert!(!candidates.is_empty(), "candidates must not be empty");

    // If the experts alone cannot fit on the off‑load device, no candidate is viable.
    if expert_blocks_bytes > available_memory_bytes_moe_offload {
        return None;
    }

    candidates
        .iter()
        // Reject any candidate whose *total* estimate is smaller than the
        // expert pool—it cannot be self‑consistent.
        .filter(|(_, total)| {
            *total >= expert_blocks_bytes && {
                let dense_bytes = *total - expert_blocks_bytes;
                dense_bytes <= available_memory_bytes_compute
            }
        })
        .max_by_key(|(_, total)| *total)
        .map(|(ty, _)| *ty)
}
