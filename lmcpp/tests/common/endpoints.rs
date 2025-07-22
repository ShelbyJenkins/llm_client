//! Thin, *import-free* wrappers that exercise every public inference
//! endpoint exposed by `LmcppServer`.
//!
//! They are intentionally lightweight: tiny prompts, single calls, and only
//! basic sanity assertions so the suite runs in a few hundred ms even on
//! low-core CI runners.

use lmcpp::*;
/// Runs a happy-path request against every endpoint and returns early on the
/// first error.
///
/// The caller decides whether to launch the server over UDS or HTTP; this
/// helper is transport-agnostic.
///
/// ### Guarantees
/// * No network or GPU is strictly required (CPU backend + local tiny model).
/// * Leaves the `LmcppServer` instance alive; shutdown is handled by `Drop`.
pub fn exercise_all(server: &lmcpp::LmcppServer) -> lmcpp::error::LmcppResult<()> {
    /* ── Completion ──────────────────────────────────────────────── */
    let completion = server.completion(
        CompletionRequest::builder()
            .prompt("Hello from integration-tests!")
            .n_predict(16),
    )?;
    assert!(
        completion.content.is_some(),
        "Completion should not return an empty string",
    );
    assert!(
        !completion.content.unwrap().is_empty(),
        "Completion should return non-empty content",
    );

    // /* ── Embeddings ─────────────────────────────────────────────── */
    // Only supported with "embeddings_only" flag.
    // let embeddings = server.embeddings(EmbeddingsRequest::builder().input("integration-tests"))?;
    // assert_eq!(
    //     embeddings.len(),
    //     1,
    //     "Embeddings endpoint should return one vector for single input",
    // );

    /* ── Tokenise / Detokenise round-trip ───────────────────────── */
    let tokenised = server.tokenize(TokenizeRequest::builder().content("round-trip"))?;
    assert!(
        !tokenised.tokens.is_empty(),
        "Tokenise should yield at least one token",
    );

    let mut tokens = Vec::new();
    for token in &tokenised.tokens {
        match token {
            TokenOutput::Id(id) => {
                tokens.push(*id);
            }
            TokenOutput::IdWithPiece { id, .. } => {
                tokens.push(*id);
            }
        }
    }

    let detokenised = server.detokenize(DetokenizeRequest::builder().tokens(tokens).build())?;

    assert!(
        detokenised.content.contains("round-trip"),
        "Detokenise should reconstruct original text",
    );

    // /* ── Infill (prefix / suffix) ───────────────────────────────── */
    // Only supports `infill` models.
    // let infill = server.infill(
    //     InfillRequest::builder()
    //         .input_prefix("Rust is")
    //         .input_suffix(".")
    //         .completion(
    //             CompletionRequest::builder()
    //                 .n_predict(8)
    //                 .prompt("Tell the truth.")
    //                 .build(),
    //         ),
    // )?;
    // assert!(
    //     infill.content.is_some(),
    //     "Infill should generate non-empty result",
    // );
    // assert!(
    //     !infill.content.unwrap().is_empty(),
    //     "Infill should return non-empty content",
    // );

    Ok(())
}
