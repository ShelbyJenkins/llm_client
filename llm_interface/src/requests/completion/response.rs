use crate::requests::{
    res_components::{GenerationSettings, InferenceProbabilities, TimingUsage, TokenUsage},
    stop_sequence::StoppingSequence,
};

pub struct CompletionResponse {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// If batched, the index of the choice in the list of choices.
    pub index: Option<u32>,
    /// The generated completion.
    pub content: String,
    pub finish_reason: CompletionFinishReason,
    pub completion_probabilities: Option<Vec<InferenceProbabilities>>,
    /// True if the context size was exceeded during generation, i.e. the number of tokens provided in the prompt (tokens_evaluated) plus tokens generated (tokens predicted) exceeded the context size (n_ctx)
    pub truncated: bool,
    pub generation_settings: GenerationSettings,
    pub timing_usage: TimingUsage,
    pub token_usage: TokenUsage,
}

impl std::fmt::Display for CompletionResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "CompletionResponse:")?;
        writeln!(f, "    content: {:?}", self.content)?;
        writeln!(f, "    finish_reason: {}", self.finish_reason)?;
        write!(f, "    generation_settings: {}", self.generation_settings)?;
        write!(f, "    timing_usage: {}", self.timing_usage)?;
        write!(f, "    token_usage: {}", self.token_usage)
    }
}

#[derive(PartialEq)]
pub enum CompletionFinishReason {
    /// The completion finished because the model generated the EOS token.
    Eos,
    /// The completion finished because the model generated a stop sequence that matches one of the provided stop sequences.
    MatchingStoppingSequence(StoppingSequence),
    /// The completion finished because the model generated a stop sequence that does not match any of the provided stop sequences.
    NonMatchingStoppingSequence(Option<String>),
    /// The completion finished because the model reached the maximum token limit.
    StopLimit,
}

impl std::fmt::Display for CompletionFinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompletionFinishReason::Eos => write!(f, "Eos"),
            CompletionFinishReason::MatchingStoppingSequence(seq) => {
                write!(f, "MatchingStoppingSequence({})", seq.as_str())
            }
            CompletionFinishReason::NonMatchingStoppingSequence(seq) => {
                write!(f, "NonMatchingStoppingSequence({:?})", seq)
            }
            CompletionFinishReason::StopLimit => write!(f, "StopLimit"),
        }
    }
}
