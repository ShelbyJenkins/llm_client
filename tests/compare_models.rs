use chrono::Local;
use llm_client::agents::prelude::{basic_text_gen, boolean_classifier, split_by_topic};
use llm_client::prelude::{LlmDefinition, ProviderClient};
use llm_client::providers::llama_cpp::models::{LlamaDef, LlamaPromptFormat};
use llm_client::providers::llama_cpp::server::start_server;
use llm_client::providers::llm_openai::models::OpenAiDef;
use llm_client::text_utils::load_content;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

const BEST_OF_N_TRIES: u8 = 5;
const RETRY_AFTER_FAIL_N_TIMES: u8 = 3;

const OPENAI_GPT35: LlmDefinition = LlmDefinition::OpenAiLlm(OpenAiDef::Gpt35Turbo);
const OPENAI_GPT4: LlmDefinition = LlmDefinition::OpenAiLlm(OpenAiDef::Gpt4);
const MISTRAL7BINSTRUCT_MODEL_URL: &str = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q8_0.gguf";
const MISTRAL7BCHAT_MODEL_URL: &str =
    "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q5_K_M.gguf";
const MIXTRAL8X7BINSTRUCT_MODEL_URL: &str =
    "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/blob/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf";
const SOLAR107BINSTRUCTV1_MODEL_URL: &str =
    "https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF/blob/main/solar-10.7b-instruct-v1.0.Q8_0.gguf";
const CONTEXT_SIZE: u16 = 9001;
const GPU_LAYERS: u16 = 23;
const SERVER_THREADS: u16 = 4;

// Text gen test
const TEXT_GEN_BASE_PROMPT: &str = "A test base prompt";
const TEXT_GEN_BASE_USER_INPUT: &str = "Tell me about the rabits.";
const TEXT_GEN_PROMPT_TEMPLATE_PATH: &str = "tests/prompt_templates/basic_text_gen.yaml";

// Boolean classifier test
const BOOLEAN_PROMPT_TEMPLATE_PATH: &str = "tests/prompt_templates/boolean_classifier.yaml";
const TRUE_TEST_1: &str = "three + three = six.";
const TRUE_TEST_2: &str = "Mixing the colors red and blue makes the color purple.";
const TRUE_TEST_3: &str = "The sun is made of hot gas.";
const TRUE_TEST_4: &str = "In the lion king, Timon said, 'Pumbaa, with you, everything's gas.'";
const TRUE_TEST_5: &str = "An email with the subject 'Your order has shipped' is a notification that your order has shipped.";
const TRUE_TEST_6: &str = "An email with the subject 'These TV prices are unbelievably low.' is an advertisement or a spam email.";

const FALSE_TEST_1: &str = "four + four = nine.";
const FALSE_TEST_2: &str = "Mixing the colors white and black makes the color red.";
const FALSE_TEST_3: &str = "Paris is the capital of Mexico.";
const FALSE_TEST_4: &str = "The moon is made of cheese.";
const FALSE_TEST_5: &str =
    "An email with the subject 'Your order has shipped' is an advertisement or a spam email.";
const FALSE_TEST_6: &str = "An email with the subject '☕️ Google admits that a Gemini AI demo video was staged.' is an advertisement or a spam email.";

// Summarizer test
const SUMMARIZER_PROMPT_TEMPLATE_PATH: &str = "tests/prompt_templates/split_by_topic.yaml";
const SUMMARIZER_TEST_CONTENT_PATH: &str = "tests/prompt_templates/split_by_topic_content.yaml";

#[tokio::test]
async fn test_runner() -> Result<(), Box<dyn std::error::Error>> {
    let zephyr_7b_instruct: LlmDefinition = LlmDefinition::LlamaLlm(LlamaDef::new(
        MISTRAL7BINSTRUCT_MODEL_URL,
        LlamaPromptFormat::Mistral7BInstruct,
        Some(CONTEXT_SIZE),
        None,
        None,
        None,
        None,
    ));

    let zephyr_7b_chat: LlmDefinition = LlmDefinition::LlamaLlm(LlamaDef::new(
        MISTRAL7BCHAT_MODEL_URL,
        LlamaPromptFormat::Mistral7BChat,
        Some(CONTEXT_SIZE),
        None,
        None,
        None,
        None,
    ));
    let mixtral_8x7b_instruct: LlmDefinition = LlmDefinition::LlamaLlm(LlamaDef::new(
        MIXTRAL8X7BINSTRUCT_MODEL_URL,
        LlamaPromptFormat::Mixtral8X7BInstruct,
        Some(CONTEXT_SIZE),
        None,
        None,
        None,
        None,
    ));
    let solar_10b_instruct: LlmDefinition = LlmDefinition::LlamaLlm(LlamaDef::new(
        SOLAR107BINSTRUCTV1_MODEL_URL,
        LlamaPromptFormat::SOLAR107BInstructv1,
        Some(CONTEXT_SIZE),
        None,
        None,
        None,
        None,
    ));

    // let llms = vec![mixtral_8x7b_instruct];
    let llms = vec![
        mixtral_8x7b_instruct,
        zephyr_7b_instruct,
        zephyr_7b_chat,
        solar_10b_instruct,
        OPENAI_GPT35,
        OPENAI_GPT4,
    ];
    #[allow(clippy::type_complexity)]
    let mut output: HashMap<String, HashMap<String, HashMap<String, HashMap<String, String>>>> =
        HashMap::new();

    let true_tests = vec![
        TRUE_TEST_1,
        TRUE_TEST_2,
        TRUE_TEST_3,
        TRUE_TEST_4,
        TRUE_TEST_5,
        TRUE_TEST_6,
    ];
    let false_tests = vec![
        FALSE_TEST_1,
        FALSE_TEST_2,
        FALSE_TEST_3,
        FALSE_TEST_4,
        FALSE_TEST_5,
        FALSE_TEST_6,
    ];

    for llm_definition in &llms {
        let mut server_process: Option<std::process::Child> = None;
        let llm_client = ProviderClient::new(llm_definition, None).await;
        let mut llm_result: HashMap<String, HashMap<String, HashMap<String, String>>> =
            HashMap::new();
        let llm_name: String;

        match llm_definition {
            LlmDefinition::LlamaLlm(model_definition) => {
                llm_name = llm_client.model_params.model_id.clone().to_string();

                server_process = Some(
                    start_server(
                        &model_definition.model_id,
                        &model_definition.model_filename,
                        None,
                        SERVER_THREADS,
                        model_definition.max_tokens_for_model,
                        GPU_LAYERS,
                        false,
                        true,
                    )
                    .await,
                );
            }
            LlmDefinition::OpenAiLlm(_) => {
                llm_name = llm_client.model_params.model_id.to_string();
            }
        }
        eprintln!("\nTesting : {:?}", llm_name);
        // Text gen test
        let text_gen_result = test_basic_text_gen(llm_definition).await;
        llm_result.insert("text_gen_results".to_string(), text_gen_result);

        // Boolean classifier test
        let true_results = test_boolean_classifier(llm_definition, &true_tests, true).await;
        let false_results = test_boolean_classifier(llm_definition, &false_tests, false).await;
        llm_result.insert(
            "boolean_classifier_true_results".to_string(),
            true_results.clone(),
        );
        llm_result.insert(
            "boolean_classifier_false_results".to_string(),
            false_results.clone(),
        );
        let boolean_classifier_score = HashMap::from([(
            "score_of_all_tests".to_string(),
            score_classifier_tests(&true_results, &false_results).to_string(),
        )]);
        let classifier_scores =
            HashMap::from([("boolean_classifier".to_string(), boolean_classifier_score)]);
        llm_result.insert("classifier_scores".to_string(), classifier_scores);

        // Summarizer test
        let split_and_summarize_result = test_split_and_summarize(llm_definition).await;
        llm_result.insert(
            "split_and_summarize_results".to_string(),
            split_and_summarize_result,
        );
        if let Some(mut server_process) = server_process {
            let _ = server_process.kill();
            let _ = server_process.wait();
        }
        output.insert(llm_name, llm_result);
    }

    eprintln!("output: {:?}", output);
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("tests/outputs");
    fs::create_dir_all(&output_dir)?;
    let now = Local::now();
    let filename = format!("test_{}.yaml", now.format("%Y_%m_%d_%H_%M_%S"));
    let yaml_file_path = output_dir.join(filename);
    let serialized = serde_yaml::to_string(&output)?;
    let mut file = File::create(yaml_file_path)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

async fn test_boolean_classifier(
    llm_definition: &LlmDefinition,
    tests: &[&str],
    correct_answer: bool,
) -> HashMap<String, HashMap<String, String>> {
    let mut output: HashMap<String, HashMap<String, String>> = HashMap::new();
    for (index, test) in tests.iter().enumerate() {
        let mut test_result = HashMap::from([("test_question".to_string(), test.to_string())]);
        let test_number = (index + 1).to_string();

        let response = boolean_classifier::classify(
            llm_definition,
            None,
            Some(test),
            Some(BOOLEAN_PROMPT_TEMPLATE_PATH),
            Some(RETRY_AFTER_FAIL_N_TIMES),
            Some(BEST_OF_N_TRIES),
        )
        .await;

        if let Err(error) = &response {
            test_result.insert("error".to_string(), error.to_string());
            output.insert(test_number, test_result);
            panic!("Test: {}\nerror: {}", test, error)
        } else {
            let (given_answer, true_count, false_count) = response.unwrap();
            let total_responses = true_count + false_count;
            let correct_responses = if correct_answer {
                true_count
            } else {
                false_count
            };
            let score = 100.0 * (correct_responses as f64) / (total_responses as f64);

            test_result.insert("given_answer".to_string(), given_answer.to_string());
            test_result.insert("correct_answer".to_string(), correct_answer.to_string());
            test_result.insert("true_count".to_string(), true_count.to_string());
            test_result.insert("false_count".to_string(), false_count.to_string());
            test_result.insert("score".to_string(), score.to_string());

            output.insert(test_number, test_result);
        };
    }
    output
}

fn score_classifier_tests(
    true_results: &HashMap<String, HashMap<String, String>>,
    false_results: &HashMap<String, HashMap<String, String>>,
) -> f64 {
    let mut total_score = 0.0;
    let mut total_tests = 0.0;
    for (_, test_result) in true_results.iter() {
        let score = test_result.get("score").unwrap().parse::<f64>().unwrap();
        total_score += score;
        total_tests += 1.0;
    }
    for (_, test_result) in false_results.iter() {
        let score = test_result.get("score").unwrap().parse::<f64>().unwrap();
        total_score += score;
        total_tests += 1.0;
    }
    total_score / total_tests
}

async fn test_basic_text_gen(
    llm_definition: &LlmDefinition,
) -> HashMap<String, HashMap<String, String>> {
    let mut output: HashMap<String, HashMap<String, String>> = HashMap::new();
    let response = basic_text_gen::generate(
        llm_definition,
        Some(TEXT_GEN_BASE_PROMPT),
        Some(TEXT_GEN_BASE_USER_INPUT),
        Some(TEXT_GEN_PROMPT_TEMPLATE_PATH),
        Some(0.5),
    )
    .await;
    if let Err(error) = &response {
        output.insert(
            "text_gen_1".to_string(),
            HashMap::from([("error".to_string(), error.to_string())]),
        );
    }
    output.insert(
        "text_gen_1".to_string(),
        HashMap::from([("response".to_string(), response.unwrap())]),
    );
    output
}
async fn test_split_and_summarize(
    llm_definition: &LlmDefinition,
) -> HashMap<String, HashMap<String, String>> {
    let mut output: HashMap<String, String> = HashMap::new();
    let feature = load_content(SUMMARIZER_TEST_CONTENT_PATH);
    let response = split_by_topic::summarize(
        llm_definition,
        &feature,
        Some(SUMMARIZER_PROMPT_TEMPLATE_PATH),
        Some(RETRY_AFTER_FAIL_N_TIMES),
        Some(0.75),
    )
    .await;
    if let Err(error) = &response {
        output.insert("error".to_string(), error.to_string());
    } else {
        let (splits, pre_split_output) = response.unwrap();

        output.insert("number_of_splits".to_string(), splits.len().to_string());
        output.insert("pre_split_output".to_string(), pre_split_output);
    }

    HashMap::from([("split_and_summarize".to_string(), output)])
}
