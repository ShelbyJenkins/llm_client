use llm_client::CompletionResponse;
use llm_utils::models::local_model::gguf::preset::LlmPreset;

use crate::backends::TestBackendConfig;

const PROMPT: &str =
    "write a buzzfeed style listicle for the given input: Boy howdy, how ya'll doing? Actually make it a blog post, I'm feeling fancy today.";
const MAX_TOKENS: [u32; 4] = [100, 200, 400, 800];

pub struct SpeedBenchmark {
    pub prompt: String,
    pub max_tokens: Vec<u32>,
    pub models: Vec<LlmPreset>,
    pub backends: Vec<TestBackendConfig>,
    pub start_time: std::time::Instant,
    pub duration: std::time::Duration,
    pub backend_results: Vec<BackendResult>,
}

impl Default for SpeedBenchmark {
    fn default() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            duration: std::time::Duration::default(),
            models: Vec::new(),
            backends: Vec::new(),
            prompt: PROMPT.to_string(),
            max_tokens: MAX_TOKENS.to_vec(),
            backend_results: Vec::new(),
        }
    }
}

impl SpeedBenchmark {
    pub fn new() -> Self {
        Self::default()
    }

    pub(super) fn finalize(&mut self) {
        self.duration = self.start_time.elapsed();
        for result in self.backend_results.iter_mut() {
            result.finalize();
        }
    }
}

pub struct BackendResult {
    pub backend: String,
    start_time: std::time::Instant,
    pub duration: std::time::Duration,
    pub model_results: Vec<ModelResult>,
}

impl BackendResult {
    pub(super) fn new(backend: &TestBackendConfig) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            backend: backend.to_string(),
            duration: std::time::Duration::default(),
            model_results: Vec::new(),
        }
    }

    fn finalize(&mut self) {
        self.duration = self.start_time.elapsed();
        for result in self.model_results.iter_mut() {
            result.finalize();
        }
    }
}

pub struct ModelResult {
    pub model_id: String,
    start_time: std::time::Instant,
    pub duration: std::time::Duration,
    pub total_completion_tokens: u32,
    pub average_prompt_tokens_per_second: f32,
    pub average_completion_tokens_per_second: f32,
    pub runs: Vec<RunResult>,
}

impl ModelResult {
    pub(super) fn new(model_id: &str) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            model_id: model_id.to_string(),
            duration: std::time::Duration::default(),
            total_completion_tokens: 0,
            average_prompt_tokens_per_second: 0.0,
            average_completion_tokens_per_second: 0.0,
            runs: Vec::new(),
        }
    }

    fn finalize(&mut self) {
        self.duration = self.start_time.elapsed();
        self.total_completion_tokens = self.runs.iter().map(|r| r.response_tokens).sum();

        self.average_completion_tokens_per_second = self
            .runs
            .iter()
            .map(|r| r.generation_tok_per_secs)
            .sum::<f32>()
            / self.runs.len() as f32;

        self.average_prompt_tokens_per_second =
            self.runs.iter().map(|r| r.prompt_tok_per_sec).sum::<f32>() / self.runs.len() as f32;
    }
}

pub struct RunResult {
    pub duration: std::time::Duration,
    pub requested_tokens: u32,
    pub response_tokens: u32,
    pub prompt_tok_per_sec: f32,
    pub generation_tok_per_secs: f32,
    pub response: CompletionResponse,
}

impl RunResult {
    pub(super) fn new(max_tok: u32, response: CompletionResponse) -> Self {
        Self {
            duration: response.timing_usage.total_time,
            requested_tokens: max_tok,
            response_tokens: response.token_usage.completion_tokens,
            prompt_tok_per_sec: response
                .timing_usage
                .prompt_tok_per_sec
                .expect("prompt_tok_per_sec not found"),
            generation_tok_per_secs: response
                .timing_usage
                .generation_tok_per_sec
                .expect("generation_tok_per_sec not found"),
            response,
        }
    }
}

impl std::fmt::Display for SpeedBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "SpeedBenchmark duration: {:?}", self.duration)?;
        // writeln!(f, " Prompt: '{}'", self.prompt)?;
        writeln!(f, " Runs at N max_tokens: {:?}", self.max_tokens)?;
        for result in self.backend_results.iter() {
            write!(f, "{}", result)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for BackendResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "  Backend: '{}'", self.backend)?;
        writeln!(f, "  BackendResult duration: {:?}", self.duration)?;
        for result in self.model_results.iter() {
            writeln!(f)?;
            write!(f, "{}", result)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ModelResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "   ModelResult for model_id: {}", self.model_id)?;
        writeln!(f, "    duration: {:?}", self.duration)?;
        writeln!(
            f,
            "    total_completion_tokens: {}",
            self.total_completion_tokens
        )?;
        writeln!(
            f,
            "    average_prompt_tokens_per_second: {}",
            self.average_prompt_tokens_per_second
        )?;
        writeln!(
            f,
            "    average_completion_tokens_per_second: {}",
            self.average_completion_tokens_per_second
        )?;
        for (i, run) in self.runs.iter().enumerate() {
            writeln!(f)?;
            writeln!(f, "     run {}:", i + 1)?;
            write!(f, "{}", run)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for RunResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "       duration: {:?}", self.duration)?;
        writeln!(f, "       requested_tokens: {:?}", self.requested_tokens)?;
        writeln!(f, "       response_tokens: {:?}", self.response_tokens)?;
        writeln!(
            f,
            "       prompt_tok_per_sec: {:?}",
            self.prompt_tok_per_sec
        )?;
        writeln!(
            f,
            "       generation_tok_per_secs: {:?}",
            self.generation_tok_per_secs
        )
    }
}
