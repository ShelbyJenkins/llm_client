use llm_interface::requests::CompletionRequest;

use super::step::{CascadeStep, StepConfig};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct CascadeRound {
    pub task: String,
    pub unresolved_steps: VecDeque<CascadeStep>,
    pub resolved_steps: VecDeque<CascadeStep>,
    pub step_separator: Option<char>,
}

impl CascadeRound {
    pub fn new<T: Into<String>>(task: T) -> CascadeRound {
        CascadeRound {
            task: task.into(),
            unresolved_steps: VecDeque::new(),
            resolved_steps: VecDeque::new(),
            step_separator: Some(' '),
        }
    }

    pub fn step_separator(&mut self, separator: char) -> &mut Self {
        self.step_separator = Some(separator);
        self
    }

    pub fn add_inference_step(&mut self, step_config: &StepConfig) -> &mut CascadeStep {
        self.unresolved_steps
            .push_back(CascadeStep::new_inference_step(
                step_config.clone(),
                self.unresolved_steps.len() + 1,
            ));
        self.unresolved_steps.back_mut().unwrap()
    }

    pub fn add_guidance_step<T: Into<String>>(
        &mut self,
        step_config: &StepConfig,
        llm_content: T,
    ) -> &mut CascadeStep {
        self.unresolved_steps
            .push_back(CascadeStep::new_guidance_step(
                step_config.clone(),
                self.unresolved_steps.len() + 1,
                llm_content,
            ));
        self.unresolved_steps.back_mut().unwrap()
    }

    pub fn generation_prefix(&self, current_step: &CascadeStep) -> crate::Result<Option<String>> {
        let mut generation_prefix = String::new();
        for step in &self.resolved_steps {
            if generation_prefix.is_empty() {
                generation_prefix.push_str(&step.display_step_outcome()?);
            } else {
                if let Some(step_separator) = self.step_separator {
                    generation_prefix.push(step_separator);
                }
                generation_prefix.push_str(&step.display_step_outcome()?);
            };
        }
        if let Some(step_prefix) = current_step.display_step_prefix() {
            if generation_prefix.is_empty() {
                generation_prefix.push_str(&step_prefix);
            } else {
                if let Some(step_separator) = self.step_separator {
                    generation_prefix.push(step_separator);
                }
                generation_prefix.push_str(&step_prefix);
            };
        }

        if generation_prefix.is_empty() {
            Ok(None)
        } else {
            Ok(Some(generation_prefix))
        }
    }

    pub fn display_outcome(&self) -> crate::Result<String> {
        let mut round_outcome = String::new();
        for step in self.resolved_steps.iter() {
            let step_outcome = step.display_step_outcome()?;
            if round_outcome.is_empty() {
                round_outcome.push_str(&step_outcome);
            } else {
                if let Some(step_separator) = self.step_separator {
                    round_outcome.push(step_separator);
                }
                round_outcome.push_str(&step_outcome);
            }
        }
        Ok(round_outcome)
    }

    pub async fn run_all_steps(&mut self, base_req: &mut CompletionRequest) -> crate::Result<()> {
        let mut error_counter = 0;
        base_req.prompt.add_user_message()?.set_content(&self.task);
        while !self.unresolved_steps.is_empty() {
            match self.run_next_step(base_req).await {
                Ok(_) => {}
                Err(e) => {
                    error_counter += 1;
                    if error_counter > 3 {
                        let mut resolved = std::mem::take(&mut self.resolved_steps);
                        resolved.append(&mut self.unresolved_steps);
                        self.unresolved_steps = resolved;
                        crate::bail!(e);
                    }
                }
            }
        }

        let outcome = self.display_outcome()?;
        base_req
            .prompt
            .add_assistant_message()?
            .set_content(outcome);
        Ok(())
    }

    pub async fn run_next_step(&mut self, base_req: &mut CompletionRequest) -> crate::Result<()> {
        let mut current_step = self.unresolved_steps.pop_front().unwrap();
        crate::info!("Running step: {:?}", current_step);
        let generation_prefix = self.generation_prefix(&current_step)?;
        match current_step
            .run_step(generation_prefix.as_deref(), base_req)
            .await
        {
            Ok(..) => {
                self.resolved_steps.push_back(current_step);
                Ok(())
            }
            Err(e) => {
                self.unresolved_steps.push_front(current_step);
                crate::bail!(e);
            }
        }
    }

    pub async fn cache_next_step(&mut self, base_req: &mut CompletionRequest) -> crate::Result<()> {
        let mut current_step = self.unresolved_steps.pop_front().unwrap();
        let generation_prefix = self.generation_prefix(&current_step)?;
        match current_step
            .set_cache_up_to_step(generation_prefix.as_deref(), base_req)
            .await
        {
            Ok(..) => {
                self.resolved_steps.push_back(current_step);
                Ok(())
            }
            Err(e) => {
                self.unresolved_steps.push_front(current_step);
                Err(e)
            }
        }
    }

    pub fn primitive_result(&self) -> Option<String> {
        if let Some(step) = self.resolved_steps.back() {
            step.primitive_result()
        } else {
            None
        }
    }

    pub fn open_round(&mut self, base_req: &mut CompletionRequest) -> crate::Result<()> {
        base_req.prompt.add_user_message()?.set_content(&self.task);
        Ok(())
    }

    pub fn last_step(&mut self) -> crate::Result<&mut CascadeStep> {
        match self.resolved_steps.back_mut() {
            Some(step) => Ok(step),
            None => crate::bail!("No steps in round"),
        }
    }

    pub fn drop_last_step(&mut self) -> crate::Result<()> {
        match self.resolved_steps.pop_back() {
            Some(..) => Ok(()),
            None => crate::bail!("No steps in round"),
        }
    }

    pub fn close_round(&mut self, base_req: &mut CompletionRequest) -> crate::Result<()> {
        base_req
            .prompt
            .add_assistant_message()?
            .set_content(self.display_outcome()?);

        Ok(())
    }

    // pub async fn set_cache_up_to_last_step(
    //     &mut self,
    //     base_req: &mut CompletionRequest,
    // ) -> crate::Result<()> {
    //     let mut last_step = self.resolved_steps.pop_back().unwrap();
    //     let generation_prefix = self.generation_prefix(&last_step)?;
    //     match last_step
    //         .set_cache_up_to_step(generation_prefix.as_deref(), base_req)
    //         .await
    //     {
    //         Ok(..) => {
    //             self.resolved_steps.push_back(last_step);
    //             Ok(())
    //         }
    //         Err(e) => {
    //             self.resolved_steps.push_back(last_step);
    //             Err(e)
    //         }
    //     }
    // }
}

impl std::fmt::Display for CascadeRound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn print_step(
            i: usize,
            step: &CascadeStep,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            writeln!(f)?;
            let color = STEP_GRADIENT[i % STEP_GRADIENT.len()];
            if let Ok(outcome) = step.display_step_outcome() {
                writeln!(f, "\x1b[1m{color}step {}\x1b[0m: '{}'", i + 1, outcome)?;
            } else {
                writeln!(f, "\x1b[1m{color}step {}\x1b[0m: 'No outcome'", i + 1,)?;
            }
            Ok(())
        }

        writeln!(f)?;
        writeln!(
            f,
            "\x1b[1m{}task\x1b[0m: '{}'",
            STEP_GRADIENT.last().unwrap(),
            self.task
        )?;
        if !self.unresolved_steps.is_empty() {
            writeln!(f, "\x1b[1munresolved_steps\x1b[0m")?;
            for (i, step) in self.unresolved_steps.iter().enumerate() {
                print_step(i, step, f)?;
            }
            writeln!(f)?;
            if !self.resolved_steps.is_empty() {
                writeln!(f, "\x1b[1mresolved_steps\x1b[0m")?;
                for (i, step) in self.resolved_steps.iter().enumerate() {
                    print_step(i, step, f)?;
                }
            }
        } else if !self.resolved_steps.is_empty() {
            for (i, step) in self.resolved_steps.iter().enumerate() {
                print_step(i, step, f)?;
            }
        }

        Ok(())
    }
}

static STEP_GRADIENT: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
    vec![
        "\x1B[38;2;0;142;250m",
        "\x1B[38;2;53;138;249m",
        "\x1B[38;2;77;133;248m",
        "\x1B[38;2;95;128;246m",
        "\x1B[38;2;111;123;243m",
        "\x1B[38;2;125;118;239m",
        "\x1B[38;2;138;112;234m",
        "\x1B[38;2;150;106;228m",
        "\x1B[38;2;160;100;222m",
        "\x1B[38;2;170;93;214m",
        "\x1B[38;2;179;86;206m",
        "\x1B[38;2;187;79;198m",
        "\x1B[38;2;194;71;189m",
        "\x1B[38;2;200;63;179m",
        "\x1B[38;2;206;54;169m",
        "\x1B[38;2;210;45;158m",
        "\x1B[38;2;214;36;147m",
        "\x1B[38;2;216;26;136m",
        "\x1B[38;2;218;13;124m",
        "\x1B[38;2;219;0;113m",
    ]
});
