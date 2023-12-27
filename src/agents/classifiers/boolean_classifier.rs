use std::collections::HashMap;
use std::error::Error;
use std::io;

use crate::{prompting, text_utils, LlmDefinition, ProviderClient};

const BASE_BOOLEAN_CLASSIFIER_PROMPT: &str = r#"You are answering a boolean question. The question will either be true/yes/affirmative, or false/no/negative. 
    IMPORTANT: If yes or true or affirmative, return '1'. If no or false or negative, return '0'."#;

pub async fn classify(
    llm_definition: &LlmDefinition,
    feature: Option<&str>,
    user_input: Option<&str>,
    prompt_template_path: Option<&str>,
    retry_after_fail_n_times: Option<u8>,
    best_of_n_tries: Option<u8>,
) -> Result<(bool, u8, u8), Box<dyn Error>> {
    if user_input.is_none() && feature.is_none() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::Other,
            "Either feature or user_input must be provided.",
        )));
    }
    let llm_client = ProviderClient::new(llm_definition, retry_after_fail_n_times).await;

    let mut best_of_n_tries = best_of_n_tries.unwrap_or(1);

    let logit_bias = create_boolean_classifier_logit_bias(&llm_client).await;

    let prompt = prompting::create_prompt_with_default_formatting(
        prompting::load_system_prompt_template(
            Some(BASE_BOOLEAN_CLASSIFIER_PROMPT),
            prompt_template_path,
        ),
        feature,
        user_input,
    );

    let mut fail_count = 0;
    let mut consensus_count;
    let mut results: Vec<bool> = vec![];
    let mut errors = vec![];
    let mut true_count;
    let mut false_count;
    let mut most_common = false;
    if best_of_n_tries > 1 && best_of_n_tries % 2 == 0 {
        best_of_n_tries += 1;
    }
    let mut batch_count = (best_of_n_tries + (best_of_n_tries % 2)) / 2;

    while fail_count < llm_client.retry_after_fail_n_times {
        if fail_count >= llm_client.retry_after_fail_n_times {
            break;
        }

        let responses = llm_client
            .make_boolean_decision(&prompt, &logit_bias, batch_count)
            .await;
        if let Err(error) = responses {
            eprintln!("Error: {}", error);
            errors.push(error);
            fail_count += 1;
            continue;
        }
        let responses = responses.unwrap();
        for resp in responses {
            if let Err(error) = boolean_classifier_validator(&resp) {
                eprintln!("Error: {}", error);
                errors.push(error.into());
                fail_count += 1;
                continue;
            }
            results.push(boolean_classifier_response_parser(&resp));
        }
        if results.is_empty() {
            fail_count += 1;
            continue;
        }
        true_count = 0;
        false_count = 0;
        for result in &results {
            if *result {
                true_count += 1;
            } else {
                false_count += 1;
            }
        }
        if true_count == false_count {
            consensus_count = true_count;
        } else {
            most_common = true_count > false_count;
            consensus_count = if most_common { true_count } else { false_count };
        }
        if consensus_count >= (best_of_n_tries + (best_of_n_tries % 2)) / 2 {
            println!(
                "Consensus reached of {:?}: with count of {}",
                most_common, consensus_count
            );
            return Ok((most_common, true_count, false_count));
        } else {
            batch_count =
                (best_of_n_tries - consensus_count + (best_of_n_tries - consensus_count) % 2) / 2;
        }
    }
    let error_message = format!(
        "Boolean classifier: Failed to get a valid response after {} retries with errors: {:?}.",
        fail_count, errors
    );

    Err(Box::new(io::Error::new(
        io::ErrorKind::Other,
        error_message,
    )))
}

async fn create_boolean_classifier_logit_bias(
    llm_client: &ProviderClient,
) -> HashMap<String, serde_json::Value> {
    let allowed_chars = vec!["0".to_string(), "1".to_string()];
    let logit_bias = prompting::logit_bias::generate_logit_bias_from_chars(
        llm_client,
        Some(allowed_chars),
        None,
    )
    .await;
    logit_bias.unwrap()
}

fn boolean_classifier_validator(response: &str) -> Result<(), String> {
    // Check if the response is a single character
    if text_utils::tiktoken_len(response) > 1 {
        return Err(format!("response '{}' should be a single token.", response));
    }

    // Check if the response is either "0" or "1"
    if response != "0" && response != "1" {
        return Err(format!(
            "response '{}' should be either '0' or '1'.",
            response
        ));
    }

    Ok(())
}

fn boolean_classifier_response_parser(response: &str) -> bool {
    response == "1"
}
