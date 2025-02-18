use std::fmt::Debug;
use std::sync::Arc;

/// A trait for tokenizers that can be used with the prompt management system.
///
/// This trait defines the core functionality needed for any tokenizer to work with
/// the prompt system. Implementors must provide methods to both tokenize text into
/// token IDs and count tokens in a given input. The trait requires thread safety
/// through Send + Sync bounds, making it suitable for use in concurrent contexts.
pub trait PromptTokenizer: Send + Sync + Debug {
    /// Converts a text string into a sequence of token IDs.
    ///
    /// This method should tokenize the input text according to the tokenizer's
    /// vocabulary and rules, returning the corresponding sequence of token IDs.
    ///
    /// # Arguments
    ///
    /// * `input` - The text string to tokenize
    ///
    /// # Returns
    ///
    /// A vector of token IDs (usize) representing the tokenized input
    fn tokenize(&self, input: &str) -> Vec<usize>;

    /// Counts the number of tokens in a text string.
    ///
    /// This method should return the number of tokens that would be produced
    /// by tokenizing the input text. It may be more efficient than calling
    /// tokenize() and counting the results.
    ///
    /// # Arguments
    ///
    /// * `input` - The text string to count tokens for
    ///
    /// # Returns
    ///
    /// The number of tokens in the input text
    fn count_tokens(&self, input: &str) -> usize;
}

impl PromptTokenizer for Arc<dyn PromptTokenizer> {
    fn tokenize(&self, input: &str) -> Vec<usize> {
        (**self).tokenize(input)
    }

    fn count_tokens(&self, input: &str) -> usize {
        (**self).count_tokens(input)
    }
}
