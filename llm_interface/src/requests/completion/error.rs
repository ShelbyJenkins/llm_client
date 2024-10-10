#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    // Break on these types
    #[error("RequestBuilderError: {0}")]
    RequestBuilderError(String),
    #[error("ClientError: {0}")]
    ClientError(#[from] crate::llms::api::error::ClientError),
    #[error("LocalClientError: {0}")]
    LocalClientError(String),
    #[error("RequestTokenLimitError: {0}")]
    RequestTokenLimitError(#[from] llm_prompt::RequestTokenLimitError),
    #[error("StopReasonUnsupported: {0}")]
    StopReasonUnsupported(String),
    #[error("ExceededRetryCount")]
    ExceededRetryCount {
        message: String,
        errors: Vec<CompletionError>,
    },
    // Continue on these types
    #[error("ReponseContentEmpty: Response had no content")]
    ReponseContentEmpty,
    #[error("StopLimitRetry: stopped_limit == true && retry_stopped_limit == true")]
    StopLimitRetry,
    #[error(
        "NoRequiredStopSequence: One of the sequences is required, but response has has None."
    )]
    NoRequiredStopSequence,
    #[error(
        "NonMatchingStopSequence: One of the sequences is required, but response's stop sequence was: {0}."
    )]
    NonMatchingStopSequence(String),
}
