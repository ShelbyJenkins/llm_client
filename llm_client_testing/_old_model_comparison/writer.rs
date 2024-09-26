use super::TestType;
use anyhow::Result;
use chrono::FixedOffset;
use csv::Writer;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    path::PathBuf,
};

pub fn create_results_path() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let timestamp = chrono::Utc::now()
        .with_timezone(&FixedOffset::east_opt(0).unwrap())
        .format("%Y_%m_%d_%H_%M")
        .to_string();
    let output_dir = manifest_dir.join("benchmark_results").join(timestamp);
    output_dir.to_str().unwrap().to_string().to_lowercase()
}

pub fn create_results_dir(results_path: &str) -> Result<()> {
    let output_dir = PathBuf::from(results_path.to_lowercase());
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir)?;
    }
    Ok(())
}

pub fn save_model_results_to_csv(result_rows: &Vec<ResultRow>, path: &str) -> Result<()> {
    println!("Saving results to: {}", path.to_lowercase());
    let file = File::create(path.to_lowercase())?;
    let mut writer = Writer::from_writer(file);
    // Write the result rows
    for result in result_rows {
        writer.serialize(result)?;
    }

    writer.flush()?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct ResultRow {
    pub duration: String,
    pub test_type: TestType,
    pub question: String,
    pub correct_answer: String,
    pub llm_response: String,
    pub response_correct: bool,
    pub human_correction: String,
    pub votes_for_choice: u8,
    pub total_decision_votes: u8,
    pub confidence: f32,
    pub justifications: String,
    pub parse_method: String,
}

pub fn save_benchmark_results_to_csv(
    benchmark_results: &Vec<ModelBenchmarkResult>,
    path: &str,
) -> Result<()> {
    let path = format!("{path}/benchmark_results.csv").to_lowercase();
    println!("Saving benchmark results to: {}", path);
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    // Write the benchmark results
    for result in benchmark_results {
        writer.serialize(result)?;
    }

    writer.flush()?;
    Ok(())
}

#[derive(Serialize)]
pub struct ModelBenchmarkResult {
    pub model_id: String,
    pub model_url: Option<String>,
    pub overall_score: f32,
    pub run_time: String,
    pub average_question_time: String,
    pub llm_backend: String,
    pub model_results_csv_path: String,
    pub dynamic_temperature: bool,
    pub best_of_n_votes: u8,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
}
