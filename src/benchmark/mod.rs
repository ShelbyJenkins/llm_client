pub mod reader;
mod writer;
use crate::{
    agents::{request::RequestConfigTrait, DecisionParserType},
    LlmClient,
    RequestConfig,
};
use anyhow::Result;
use chrono::Duration;
use core::panic;
use llm_utils::models::OpenSourceModelType;
use reader::*;
use std::path::PathBuf;
use writer::*;

pub struct LlmBenchmark {
    best_of_n_votes: u8,
    custom_model_list_csv_path: Option<String>,
    custom_questions_csv_path: Option<String>,
    decision_parser_type: Option<DecisionParserType>,
    decision_justification_token_count: u32,
    default_request_config: RequestConfig,
    dynamic_temperature: bool,
    models: Vec<ModelRow>,
    questions: Vec<QuestionRow>,
    bechmark_results: Vec<ModelBenchmarkResult>,
    results_path: String,
    restart_from_path: bool,
}
impl Default for LlmBenchmark {
    fn default() -> Self {
        Self::new()
    }
}
impl LlmBenchmark {
    pub fn new() -> Self {
        LlmBenchmark {
            best_of_n_votes: 3,
            custom_model_list_csv_path: None,
            custom_questions_csv_path: None,
            decision_parser_type: None,
            decision_justification_token_count: 300,
            default_request_config: RequestConfig::default(),
            dynamic_temperature: true,
            models: vec![],
            questions: vec![],
            bechmark_results: vec![],
            results_path: create_results_path(),
            restart_from_path: false,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        self.set_questions()?;
        self.set_models()?;

        let mut benchmarked_models = vec![];
        for mut model in self.models.clone() {
            match model.backend {
                Backend::LlamaCpp => {
                    let mut llm_client = if let Some(model_url) = &model.model_url {
                        LlmClient::llama_backend().model_url(model_url)
                    } else if let Some(model_id) = &model.model_id {
                        LlmClient::llama_backend()
                            .open_source_model_type(OpenSourceModelType::from_model_id(model_id))
                    } else {
                        panic!("Model id or model url not found for LlamaCpp model");
                    };
                    if let Some(ctx_size) = model.ctx_size {
                        llm_client = llm_client.ctx_size(ctx_size);
                    };
                    if let Some(available_vram) = model.available_vram {
                        llm_client = llm_client.available_vram(available_vram);
                    };
                    if let Some(n_gpu_layers) = model.n_gpu_layers {
                        llm_client = llm_client.n_gpu_layers(n_gpu_layers);
                    };
                    let mut llm_client = llm_client.init().await?;

                    llm_client.default_request_config = self.default_request_config.clone();
                    model.model_url = Some(llm_client.backend.get_model_url());
                    model.model_id = Some(self.run_model_benchmark(&llm_client).await?);
                }
                Backend::OpenAI => {
                    if let Some(decision_parser_type) = &self.decision_parser_type {
                        if decision_parser_type == &DecisionParserType::LogitBias {
                            eprintln!("LogitBias decision backend not supported for OpenAI models");
                            continue;
                        }
                    }

                    let model_id = if let Some(id) = &model.model_id {
                        id.clone()
                    } else {
                        panic!("Model id not found for OpenAI model");
                    };
                    let mut llm_client = LlmClient::openai_backend()
                        .from_model_id(&model_id)
                        .init()?;
                    llm_client.default_request_config = self.default_request_config.clone();
                    model.model_url = Some(llm_client.backend.get_model_url());
                    model.model_id = Some(self.run_model_benchmark(&llm_client).await?);
                }
                Backend::Anthropic => {
                    if let Some(decision_parser_type) = &self.decision_parser_type {
                        if decision_parser_type == &DecisionParserType::Grammar {
                            eprintln!(
                                "Grammar decision backend not supported for Anthropic models"
                            );
                            continue;
                        }
                        if decision_parser_type == &DecisionParserType::LogitBias {
                            eprintln!(
                                "LogitBias decision backend not supported for Anthropic models"
                            );
                            continue;
                        }
                    }

                    let model_id = if let Some(id) = &model.model_id {
                        id.clone()
                    } else {
                        panic!("Model id not found for OpenAI model");
                    };
                    let mut llm_client = LlmClient::anthropic_backend()
                        .from_model_id(&model_id)
                        .init()?;
                    llm_client.default_request_config = self.default_request_config.clone();
                    model.model_url = Some(llm_client.backend.get_model_url());
                    model.model_id = Some(self.run_model_benchmark(&llm_client).await?);
                }
            };
            benchmarked_models.push(model);
        }
        self.models = benchmarked_models;
        self.create_final_results()?;
        save_benchmark_results_to_csv(&self.bechmark_results, &self.get_results_path()?)?;
        Ok(())
    }

    pub async fn restart_from_path(&mut self, results_path: &str) -> Result<()> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output_dir = manifest_dir.join(results_path);
        self.results_path = output_dir.to_str().unwrap().to_string();
        self.restart_from_path = true;
        self.run().await?;
        Ok(())
    }

    pub fn with_custom_questions_csv_path(&mut self, custom_questions_csv_path: &str) -> &mut Self {
        self.custom_questions_csv_path = Some(custom_questions_csv_path.to_owned());
        self
    }

    pub fn with_custom_model_list_csv_path(
        &mut self,
        custom_model_list_csv_path: &str,
    ) -> &mut Self {
        self.custom_model_list_csv_path = Some(custom_model_list_csv_path.to_owned());
        self
    }

    pub fn best_of_n_votes(&mut self, best_of_n_votes: u8) -> &mut Self {
        self.best_of_n_votes = best_of_n_votes;
        self
    }

    pub fn dynamic_temperature(&mut self, dynamic_temperature: bool) -> &mut Self {
        self.dynamic_temperature = dynamic_temperature;
        self
    }

    pub fn decision_justification_token_count(&mut self, token_count: u32) -> &mut Self {
        self.decision_justification_token_count = token_count;
        self
    }

    pub fn use_logit_bias_backend(&mut self) -> &mut Self {
        self.decision_parser_type = Some(DecisionParserType::LogitBias);
        self
    }

    pub fn use_grammar_backend(&mut self) -> &mut Self {
        self.decision_parser_type = Some(DecisionParserType::Grammar);
        self
    }

    pub fn use_basic_backend(&mut self) -> &mut Self {
        self.decision_parser_type = Some(DecisionParserType::Basic);
        self
    }

    fn set_questions(&mut self) -> Result<()> {
        let questions_csv_path = if let Some(path) = &self.custom_questions_csv_path {
            path.to_owned()
        } else {
            DEFAULT_QUESTIONS_CSV_PATH.to_owned()
        };
        self.questions = questions_from_csv(&questions_csv_path)?;
        Ok(())
    }

    fn set_models(&mut self) -> Result<()> {
        let model_list_csv_path = if let Some(path) = &self.custom_model_list_csv_path {
            path.to_owned()
        } else {
            DEFAULT_MODELS_CSV_PATH.to_owned()
        };
        let models = models_from_csv(&model_list_csv_path)?;

        self.models = models;
        Ok(())
    }

    async fn run_model_benchmark(&self, llm_client: &LlmClient) -> Result<String> {
        println!(
            "\nbenchmarking model_id: {:?}",
            llm_client.backend.get_model_id()
        );
        if self.skip_if_restart_and_benched(&llm_client.backend.get_model_id())? {
            return Ok(llm_client.backend.get_model_id());
        }

        let mut result_rows = vec![];

        for question in self.questions.clone() {
            let start_timestamp = chrono::Utc::now();
            let result_row = match question.test_type {
                TestType::BasicText => {
                    let res = llm_client
                        .text()
                        .basic_text()
                        .user_content(&question.question)
                        .run()
                        .await;
                    let total_time = chrono::Utc::now() - start_timestamp;
                    match res {
                        Ok(res) => ResultRow {
                            duration: total_time.num_milliseconds().to_string(),
                            test_type: question.test_type,
                            question: question.question,
                            correct_answer: question.correct_answer.clone(),
                            llm_response: res.to_owned(),
                            response_correct: true,
                            human_correction: "".to_string(),
                            votes_for_choice: 0,
                            total_decision_votes: 0,
                            confidence: 1.0,
                            justifications: "".to_string(),
                            decision_parser_type: "".to_string(),
                        },
                        Err(e) => ResultRow {
                            duration: total_time.num_milliseconds().to_string(),
                            test_type: question.test_type,
                            question: question.question,
                            correct_answer: question.correct_answer.clone(),
                            llm_response: format!("Question failed with error: {:?}", e),
                            response_correct: true,
                            human_correction: "".to_string(),
                            votes_for_choice: 0,
                            total_decision_votes: 0,
                            confidence: 1.0,
                            justifications: "".to_string(),
                            decision_parser_type: "".to_string(),
                        },
                    }
                }
                _ => {
                    let decider = llm_client
                        .decider()
                        .decision_justification_token_count(self.decision_justification_token_count)
                        .dynamic_temperature(self.dynamic_temperature)
                        .best_of_n_votes(self.best_of_n_votes);

                    let decider = if let Some(decision_parser_type) = &self.decision_parser_type {
                        match decision_parser_type {
                            DecisionParserType::Basic => decider.use_basic_backend(),
                            DecisionParserType::LogitBias => decider.use_logit_bias_backend(),
                            DecisionParserType::Grammar => decider.use_grammar_backend(),
                        }
                    } else {
                        decider
                    };

                    let decision_parser_type = match &decider.decider_config.decision_parser_type {
                        DecisionParserType::Basic => "Basic".to_string(),
                        DecisionParserType::LogitBias => "LogitBias".to_string(),
                        DecisionParserType::Grammar => "Grammar".to_string(),
                    };
                    let res = match question.test_type {
                        TestType::Boolean => {
                            decider
                                .boolean()
                                .user_content(&question.question)
                                .run_with_result()
                                .await
                        }
                        TestType::Numeric => {
                            decider
                                .integer()
                                .upper_bound(9)
                                .user_content(&question.question)
                                .run_with_result()
                                .await
                        }
                        TestType::BasicText => unreachable!("BasicText should not be here"),
                    };
                    let total_time = chrono::Utc::now() - start_timestamp;
                    match res {
                        Ok(res) => {
                            let llm_response = match question.test_type {
                                TestType::Boolean => res.choice_value,
                                TestType::Numeric => res.choice_index.to_string(),
                                TestType::BasicText => unreachable!("BasicText should not be here"),
                            };

                            ResultRow {
                                duration: total_time.num_milliseconds().to_string(),
                                test_type: question.test_type,
                                question: question.question,
                                correct_answer: question.correct_answer.clone(),
                                response_correct: llm_response == question.correct_answer,
                                llm_response,
                                human_correction: "".to_string(),
                                votes_for_choice: res.votes_for_choice,
                                total_decision_votes: res.total_decision_votes,
                                confidence: res.confidence,
                                justifications: res.justifications.join("; "),
                                decision_parser_type,
                            }
                        }
                        Err(e) => ResultRow {
                            duration: total_time.num_milliseconds().to_string(),
                            test_type: question.test_type,
                            question: question.question,
                            correct_answer: question.correct_answer.clone(),
                            llm_response: format!("Question failed with error: {:?}", e),
                            response_correct: false,
                            human_correction: "".to_string(),
                            votes_for_choice: 0,
                            total_decision_votes: 0,
                            confidence: 0.0,
                            justifications: "".to_string(),
                            decision_parser_type,
                        },
                    }
                }
            };

            result_rows.push(result_row);
        }
        let model_id = llm_client.backend.get_model_id();
        let path = format!("{}/{}.csv", self.get_results_path()?, model_id);
        save_model_results_to_csv(&result_rows, &path)?;
        Ok(model_id)
    }

    fn get_results_path(&self) -> Result<String> {
        let results_path = self.results_path.clone();
        create_results_dir(&results_path)?;
        Ok(results_path)
    }

    fn skip_if_restart_and_benched(&self, model_id: &str) -> Result<bool> {
        if self.restart_from_path {
            let path = format!(
                "{}/{}.csv",
                self.get_results_path()?,
                model_id.to_lowercase()
            );
            let path = std::path::Path::new(&path);
            if path.exists() {
                println!(
                    "Skipping: {:?} as it already exists",
                    format!(
                        "{}/{}.csv",
                        self.get_results_path()?,
                        model_id.to_lowercase()
                    )
                );
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn create_final_results(&mut self) -> Result<()> {
        for model in &self.models {
            let model_id = if let Some(model_id) = &model.model_id {
                model_id.clone()
            } else {
                panic!("model_id should be set here.");
            };
            let path = format!(
                "{}/{}.csv",
                self.get_results_path()?,
                &model_id.to_lowercase()
            );
            if !std::path::Path::new(&path).exists() {
                eprintln!("Model results not found for model_id: {:?}", path);
                continue;
            } else {
                self.bechmark_results
                    .push(self.create_model_benchmark_result(model, path)?);
            }
        }
        Ok(())
    }

    fn create_model_benchmark_result(
        &self,
        model: &ModelRow,
        path: String,
    ) -> Result<ModelBenchmarkResult> {
        let result_rows: Vec<ResultRow> = read_model_results_from_csv(&path)?;
        let mut total_correct = 0;
        let mut total_questions = 0;
        let mut total_duration: Duration = Duration::milliseconds(0);
        for result in result_rows {
            total_questions += 1;
            if result.response_correct {
                total_correct += 1;
            }
            let duration = result.duration.parse::<i64>().unwrap();
            total_duration += Duration::milliseconds(duration)
        }
        let overall_score = total_correct as f32 / total_questions as f32;
        let average_time = total_duration / total_questions;

        let llm_backend = match model.backend {
            Backend::LlamaCpp => "LlamaCpp".to_string(),
            Backend::OpenAI => "OpenAI".to_string(),
            Backend::Anthropic => "Anthropic".to_string(),
        };

        let run_time = format!(
            "{:02}:{:02}:{:02}.{:03}",
            total_duration.num_hours(),
            total_duration.num_minutes() % 60,
            total_duration.num_seconds() % 60,
            total_duration.num_milliseconds() % 1000
        );
        let average_question_time = format!(
            "{:02}:{:02}:{:02}.{:03}",
            average_time.num_hours(),
            average_time.num_minutes() % 60,
            average_time.num_seconds() % 60,
            average_time.num_milliseconds() % 1000
        );

        let benchmark_result = ModelBenchmarkResult {
            model_id: model.model_id.clone().unwrap(),
            model_url: model.model_url.clone(),
            overall_score,
            run_time,
            average_question_time,
            llm_backend,
            model_results_csv_path: path,
            dynamic_temperature: self.dynamic_temperature,
            best_of_n_votes: self.best_of_n_votes,
            frequency_penalty: self.default_request_config.frequency_penalty,
            presence_penalty: self.default_request_config.presence_penalty,
            temperature: self.default_request_config.temperature,
            top_p: self.default_request_config.top_p,
        };

        Ok(benchmark_result)
    }
}

impl RequestConfigTrait for LlmBenchmark {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.default_request_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    #[tokio::test]
    #[serial]
    pub async fn test() -> Result<()> {
        // let mut benchmark = LlmBenchmark::new();
        // benchmark.with_custom_model_list_csv_path(TEST_MODELS_CSV_PATH);
        // benchmark.with_custom_questions_csv_path(TEST_QUESTIONS_CSV_PATH);
        // benchmark.run().await?;

        let mut benchmark = LlmBenchmark::new();
        benchmark.use_basic_backend();
        benchmark
            .restart_from_path("benchmark_results/2024_05_27_22_14")
            .await?;
        Ok(())
    }
}
