use super::base_request::BaseRequestConfig;
use anyhow::{anyhow, Result};
use llm_utils::prompting::{
    LlmPrompt,
    PromptConcatenator,
    PromptConcatenatorTrait,
    PromptMessage,
    PromptMessageType,
};

#[derive(Clone)]
pub struct InstructPrompt {
    pub instructions: Option<PromptMessage>,
    pub supporting_material: Option<PromptMessage>,
    pub prompt: LlmPrompt,
}

impl InstructPrompt {
    pub fn new(config: &BaseRequestConfig) -> Self {
        Self {
            instructions: None,
            supporting_material: None,
            prompt: config.new_prompt(),
        }
    }

    pub fn reset_instruct_prompt(&mut self) {
        self.instructions = None;
        self.supporting_material = None;
    }

    pub fn build_instructions(&mut self) -> Option<String> {
        if let Some(instructions) = &mut self.instructions {
            instructions.concatenator = self.prompt.concatenator.clone();
            if instructions.requires_build() {
                instructions.build();
            };
            instructions
                .built_prompt_string
                .as_ref()
                .map(|string| string.to_owned())
        } else {
            None
        }
    }

    pub fn build_supporting_material(&mut self) -> Option<String> {
        if let Some(supporting_material) = &mut self.supporting_material {
            supporting_material.concatenator = self.prompt.concatenator.clone();
            if supporting_material.requires_build() {
                supporting_material.build();
            };
            supporting_material
                .built_prompt_string
                .as_ref()
                .map(|string| string.to_owned())
        } else {
            None
        }
    }

    pub fn build_instruct_prompt(&mut self, supporting_material_first: bool) -> Result<String> {
        Ok(
            match (self.build_instructions(), self.build_supporting_material()) {
                (Some(instructions), Some(supporting_material)) => {
                    if supporting_material_first {
                        format!(
                            "{}{}{}",
                            supporting_material,
                            self.prompt.concatenator.as_str(),
                            instructions
                        )
                    } else {
                        format!(
                            "{}{}{}",
                            instructions,
                            self.prompt.concatenator.as_str(),
                            supporting_material
                        )
                    }
                }
                (Some(instructions), None) => instructions,
                (None, Some(supporting_material)) => supporting_material,

                (None, None) => {
                    return Err(anyhow!("No instructions or supporting material found"))
                }
            },
        )
    }
}

impl PromptConcatenatorTrait for InstructPrompt {
    fn concate_mut(&mut self) -> &mut PromptConcatenator {
        &mut self.prompt.concatenator
    }

    fn clear_built(&mut self) {
        self.prompt.clear_built_prompt();
    }
}

pub trait InstructPromptTrait {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt;

    fn set_instructions<T: AsRef<str>>(&mut self, instructions: T) -> &mut Self {
        self.instructions().set_content(instructions);
        self
    }

    fn instructions(&mut self) -> &mut PromptMessage {
        if self.instruct_prompt_mut().instructions.is_none() {
            self.instruct_prompt_mut().instructions = Some(PromptMessage::new(
                &PromptMessageType::User,
                &self.instruct_prompt_mut().prompt.concatenator,
            ));
        }
        self.instruct_prompt_mut().instructions.as_mut().unwrap()
    }

    fn set_supporting_material<T: AsRef<str>>(&mut self, supporting_material: T) -> &mut Self {
        self.supporting_material().set_content(supporting_material);
        self
    }

    fn supporting_material(&mut self) -> &mut PromptMessage {
        if self.instruct_prompt_mut().supporting_material.is_none() {
            self.instruct_prompt_mut().supporting_material = Some(PromptMessage::new(
                &PromptMessageType::User,
                &self.instruct_prompt_mut().prompt.concatenator,
            ));
        }

        self.instruct_prompt_mut()
            .supporting_material
            .as_mut()
            .unwrap()
    }
}
