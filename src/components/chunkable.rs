use super::{instruct_request::InstructRequest, request_config::BaseRequestConfig};
use anyhow::{anyhow, Result};
use llm_utils::{
    prompting::token_count::RequestTokenLimitError,
    text_utils::TextChunker,
    tokenizer::LlmTokenizer,
};
use std::sync::Arc;

pub trait ChunkableRequestTrait {
    fn config_mut(&mut self) -> &mut BaseRequestConfig;
    fn req_mut(&mut self) -> &mut InstructRequest;

    fn get_tokenizer(&self) -> Arc<LlmTokenizer>;

    fn split_if_exceeds_token_limits(&mut self, user_content: &str) -> Result<Vec<String>> {
        match self.req_mut().set_max_tokens_for_request() {
            Ok(_) => Ok(vec![user_content.to_owned()]),
            Err(e) => match e {
                RequestTokenLimitError::PromptTokensExceeds { .. } => {
                    let total_tokens =
                        self.req_mut().prompt.total_prompt_tokens.unwrap() as f32 * 1.2;
                    let chunk_count = (total_tokens / self.config_mut().ctx_size as f32).ceil();
                    let chunk_size = (total_tokens / chunk_count).ceil() as u32;

                    match TextChunker::new_with_tokenizer(&self.get_tokenizer())
                        .max_chunk_token_size(chunk_size)
                        .run(user_content)
                    {
                        Some(splits) => Ok(splits),
                        None => Err(anyhow!("{}", e)),
                    }
                }
                _ => Err(anyhow!("{}", e)),
            },
        }
    }
}
