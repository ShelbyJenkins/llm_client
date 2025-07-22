use std::borrow::Cow;

pub const DEFAULT_CONTEXT_LENGTH: u64 = 8192;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LlmModelBase {
    pub model_id: Cow<'static, str>,
    pub friendly_name: Cow<'static, str>,
    pub model_ctx_size: u64,
    pub inference_ctx_size: u64,
}

impl LlmModelBase {
    pub fn new(
        model_id: &str,
        friendly_name: Option<&str>,
        model_ctx_size: u64,
        inference_ctx_size: Option<u64>,
    ) -> Self {
        let model_id: Cow<'_, str> = model_id.to_string().into();
        let friendly_name: Cow<'_, str> = if let Some(friendly_name) = friendly_name {
            friendly_name.to_string().into()
        } else {
            model_id.clone()
        };
        Self {
            model_id,
            friendly_name,
            model_ctx_size,
            inference_ctx_size: inference_ctx_size.unwrap_or(model_ctx_size),
        }
    }

    // pub fn set_prompt(
    //     &self,
    //     prompt: &mut LlmPrompt,
    //     tokenizer: std::sync::Arc<dyn PromptTokenizer>,
    // ) -> crate::Result<()> {
    //     match self {
    //         LlmModel::Local(local_llm) => {
    //             let chat_template = local_llm.chat_template();
    //             prompt.new_local_prompt(
    //                 tokenizer,
    //                 &chat_template.chat_template,
    //                 chat_template.bos_token.as_deref(),
    //                 &chat_template.eos_token,
    //                 chat_template.unk_token.as_deref(),
    //                 chat_template.base_generation_prefix.as_deref(),
    //             )?;
    //         }
    //         LlmModel::Cloud(cloud_llm) => {
    //             prompt.new_api_prompt(
    //                 tokenizer,
    //                 Some(cloud_llm.tokens_per_message),
    //                 cloud_llm.tokens_per_name,
    //             )?;
    //         }
    //     }
    //     Ok(())
    // }
}
