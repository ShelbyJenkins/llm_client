use super::ResultRow;
use crate::DecisionParserType;
use anyhow::Result;
use csv::ReaderBuilder;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;

pub fn questions_from_csv(questions_path: &str) -> Result<Vec<QuestionRow>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let questions_path = manifest_dir.join("src").join(questions_path);
    let mut question_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(questions_path)?;

    let mut questions = vec![];
    for result in question_reader.deserialize() {
        let question_row: QuestionRow = result?;
        questions.push(QuestionRow {
            test_type: question_row.test_type,
            question: question_row.question,
            correct_answer: question_row.correct_answer,
        });
    }
    Ok(questions)
}

pub const DEFAULT_QUESTIONS_CSV_PATH: &str = "benchmark/inputs/llm_benchmark_questions.csv";
pub const TEST_QUESTIONS_CSV_PATH: &str = "benchmark/inputs/llm_benchmark_test_questions.csv";

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TestType {
    Boolean,
    Numeric,
    BasicText,
}

fn deserialize_lowercase<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(s.to_lowercase())
}

#[derive(Clone, Deserialize)]
pub struct QuestionRow {
    pub question: String,
    pub test_type: TestType,
    #[serde(deserialize_with = "deserialize_lowercase")]
    pub correct_answer: String,
}

pub fn models_from_csv(models_path: &str) -> Result<Vec<ModelRow>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models_path = manifest_dir.join("src").join(models_path);
    let mut model_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(models_path)?;

    let mut models = vec![];
    for result in model_reader.deserialize() {
        let model_row: ModelRow = result?;
        models.push(ModelRow {
            backend: model_row.backend,
            model_id: model_row.model_id,
            model_url: model_row.model_url,
            available_vram: model_row.available_vram,
            ctx_size: model_row.ctx_size,
            n_gpu_layers: model_row.n_gpu_layers,
            decision_parser_type: model_row.decision_parser_type,
        });
    }
    Ok(models)
}
pub const DEFAULT_MODELS_CSV_PATH: &str = "benchmark/inputs/llm_benchmark_models.csv";
pub const TEST_MODELS_CSV_PATH: &str = "benchmark/inputs/llm_benchmark_test_models.csv";

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    LlamaCpp,
    OpenAI,
    Anthropic,
}

#[derive(Deserialize, Clone)]
pub struct ModelRow {
    pub backend: Backend,
    pub model_id: Option<String>,
    pub model_url: Option<String>,
    pub available_vram: Option<u32>,
    pub ctx_size: Option<u32>,
    pub n_gpu_layers: Option<u16>,
    pub decision_parser_type: Option<DecisionParserType>,
}

pub fn read_model_results_from_csv(path: &str) -> Result<Vec<ResultRow>> {
    let mut results_reader = ReaderBuilder::new().has_headers(true).from_path(path)?;

    let mut results = vec![];
    for result in results_reader.deserialize() {
        let result_row: ResultRow = result?;
        results.push(result_row);
    }
    Ok(results)
}
