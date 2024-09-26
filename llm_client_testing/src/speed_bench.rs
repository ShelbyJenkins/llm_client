use super::*;

const PROMPT: &str =
    "write a buzzfeed style listicle for the given input: Boy howdy, how ya'll doing? Actually make it a blog post, I'm feeling fancy today.";
const MAX_TOKENS: [u32; 4] = [100, 200, 400, 800];

#[derive(Default)]
pub struct SpeedBenchResult {
    pub model_id: String,
    pub total_benchmark_duration: std::time::Duration,
    pub total_completion_tokens: u32,
    pub average_prompt_tokens_per_second: f32,
    pub average_completion_tokens_per_second: f32,
    pub runs: Vec<SpeedBenchSingleResult>,
}

pub struct SpeedBenchSingleResult {
    pub single_benchmark_duration: std::time::Duration,
    pub requested_completion_tokens: u32,
    pub response_completion_tokens: u32,
    pub prompt_tok_per_sec: f32,
    pub generation_tok_per_secs: f32,
    pub response: CompletionResponse,
}

pub async fn token_generation(
    llm_client: &LlmClient,
    prompt: Option<String>,
    max_tokens: Vec<u32>,
) -> crate::Result<SpeedBenchResult> {
    let prompt = prompt.unwrap_or_else(|| PROMPT.to_string());
    let max_tokens = max_tokens
        .is_empty()
        .then(|| MAX_TOKENS.to_vec())
        .unwrap_or(max_tokens);

    let start_time = std::time::Instant::now();

    let mut all_results = SpeedBenchResult::default();
    all_results.model_id = llm_client.backend.model_id().to_string();
    for max_tok in max_tokens {
        let res = run_token_generation(llm_client, max_tok, &prompt).await?;
        all_results.runs.push(SpeedBenchSingleResult {
            single_benchmark_duration: res.timing_usage.total_time,
            requested_completion_tokens: max_tok,
            response_completion_tokens: res.token_usage.completion_tokens,
            prompt_tok_per_sec: res
                .timing_usage
                .prompt_tok_per_sec
                .expect("prompt_tok_per_sec not found"),
            generation_tok_per_secs: res
                .timing_usage
                .generation_tok_per_sec
                .expect("generation_tok_per_sec not found"),
            response: res,
        });
    }

    let total_completion_toks_per_secs: f32 = all_results
        .runs
        .iter()
        .map(|r| r.generation_tok_per_secs)
        .sum();
    all_results.average_completion_tokens_per_second =
        total_completion_toks_per_secs / all_results.runs.len() as f32;
    let total_prompt_toks_per_secs: f32 =
        all_results.runs.iter().map(|r| r.prompt_tok_per_sec).sum();
    all_results.average_prompt_tokens_per_second =
        total_prompt_toks_per_secs / all_results.runs.len() as f32;
    all_results.total_completion_tokens = all_results
        .runs
        .iter()
        .map(|r| r.response_completion_tokens)
        .sum();
    all_results.total_benchmark_duration = start_time.elapsed();
    Ok(all_results)
}

async fn run_token_generation(
    llm_client: &LlmClient,
    max_tok: u32,
    prompt: &str,
) -> crate::Result<CompletionResponse> {
    let mut gen = llm_client.basic_completion();

    gen.prompt().add_user_message()?.set_content(prompt);
    gen.max_tokens(max_tok.into());
    gen.run().await
}

impl std::fmt::Display for SpeedBenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "SpeedBenchResult:")?;
        writeln!(f, "    model_id: {}", self.model_id)?;
        writeln!(
            f,
            "    total_benchmark_duration: {:?}",
            self.total_benchmark_duration
        )?;
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
            writeln!(f, "    run {}:", i + 1)?;
            write!(f, "{}", run)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for SpeedBenchSingleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "       single_benchmark_duration: {:?}",
            self.single_benchmark_duration
        )?;
        writeln!(
            f,
            "       requested_completion_tokens: {:?}",
            self.requested_completion_tokens
        )?;
        writeln!(
            f,
            "       response_completion_tokens: {:?}",
            self.response_completion_tokens
        )?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "llama_cpp_backend")]
    #[tokio::test]
    #[ignore]
    pub async fn test_llama() -> crate::Result<()> {
        let mut results = vec![];
        let llm = llama_cpp_tiny_llm().await?;
        let res = token_generation(&llm, None, vec![]).await?;
        results.push(res);
        llm.shutdown();

        let llm = llama_cpp_medium_llm().await?;
        let res = token_generation(&llm, None, vec![]).await?;
        results.push(res);
        llm.shutdown();

        let llm = llama_cpp_large_llm().await?;
        let res = token_generation(&llm, None, vec![]).await?;
        results.push(res);
        llm.shutdown();

        let llm = llama_cpp_max_llm().await?;
        let res = token_generation(&llm, None, vec![]).await?;
        results.push(res);

        for res in results {
            println!("{}", res);
        }
        Ok(())
    }

    #[cfg(feature = "mistral_rs_backend")]
    #[tokio::test]
    #[ignore]
    pub async fn test_mistral() -> crate::Result<()> {
        let llm = mistral_rs_tiny_llm().await?;
        let res = token_generation(&llm, None, vec![]).await?;
        println!("{}", res);
        Ok(())
    }
}
