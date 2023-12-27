use crate::agents::prelude::boolean_classifier;
use crate::{prompting, text_utils, LlmDefinition, ProviderClient};
use std::collections::HashMap;
use std::error::Error;
use std::io;

const BASE_SPLIT_PROMPT: &str = "Split the feature into discrete features. A feature is unique if a journalist would write a story about it, or if it would have it's own Wikipedia page. Do not label the features. Do not title the features. Return only the summarized features.";
const SPLIT_BY_NUMBERED_LIST_PROMPT: &str = r#"Create a numbered newline separated list. Start each feature with a number like "n. feature". End each feature with a new line char: "\n"."#;
const SEPARATORS: [&str; 11] = [
    r"\d+\.\s",
    r"\d+:\s",
    r"\d+\)\s",
    r"\d+\s",
    "Feature \\d+:",
    "feature \\d+:",
    "- ",
    "[n]",
    "\n",
    "\n\n",
    "\\n",
];

pub async fn summarize(
    llm_definition: &LlmDefinition,
    feature: &str,
    prompt_template_path: Option<&str>,
    retry_after_fail_n_times: Option<u8>,
    model_token_utilization: Option<f32>,
) -> Result<(Vec<String>, String), Box<dyn Error>> {
    let llm_client = ProviderClient::new(llm_definition, retry_after_fail_n_times).await;
    let logit_bias = create_split_and_summarize_logit_bias(&llm_client).await;

    let feature = text_utils::clean_text_content(feature);

    let prompt = create_split_and_summarize_prompt(&feature, prompt_template_path);

    let mut fail_count = 1;
    let mut errors = vec![];

    while fail_count < llm_client.retry_after_fail_n_times {
        if fail_count >= llm_client.retry_after_fail_n_times {
            break;
        }

        let response = llm_client
            .generate_text(
                &prompt,
                &Some(logit_bias.clone()),
                model_token_utilization,
                None,
                None,
            )
            .await;
        if let Err(error) = response {
            fail_count += 1;
            errors.push(error);
            continue;
        }
        let unsplit_summarized_feature = response.unwrap().trim().to_string();

        let list_check = check_if_list(llm_definition, &unsplit_summarized_feature).await;
        if let Err(error) = list_check {
            fail_count += 1;
            errors.push(error);
            continue;
        } else if !list_check.unwrap() {
            fail_count += 1;
            errors.push("Not a list.".into());
            continue;
        }
        for sep in SEPARATORS.iter() {
            let sep_check = check_seperator(llm_definition, &unsplit_summarized_feature, sep).await;
            if let Err(error) = sep_check {
                fail_count += 1;
                errors.push(error);
                continue;
            }
            if !sep_check.unwrap() {
                continue;
            }

            let splits = text_utils::split_text_with_regex(&unsplit_summarized_feature, sep, false);
            if splits.len() > 1 {
                return Ok((splits, unsplit_summarized_feature));
            }
        }
        fail_count += 1;
    }

    let error_message = format!(
        "Failed to get a valid response after {} retries with errors: {:?}.",
        fail_count, errors
    );

    Err(Box::new(io::Error::new(
        io::ErrorKind::Other,
        error_message,
    )))
}

async fn create_split_and_summarize_logit_bias(
    llm_client: &ProviderClient,
) -> HashMap<String, serde_json::Value> {
    let mut logit_bias_tokens = vec![];

    logit_bias_tokens.append(&mut prompting::logit_bias::generate_whitespace_chars());
    logit_bias_tokens.append(&mut prompting::logit_bias::generate_bad_split_chars());

    prompting::logit_bias::generate_logit_bias_from_chars(llm_client, None, Some(logit_bias_tokens))
        .await
        .unwrap()
}

fn create_split_and_summarize_prompt(
    feature: &str,
    prompt_template_path: Option<&str>,
) -> HashMap<String, HashMap<String, String>> {
    let mut base_prompt = BASE_SPLIT_PROMPT.to_string();

    base_prompt += "\n";
    base_prompt += SPLIT_BY_NUMBERED_LIST_PROMPT;

    prompting::create_prompt_with_default_formatting(
        prompting::load_system_prompt_template(Some(&base_prompt), prompt_template_path),
        Some(feature),
        None,
    )
}

async fn check_if_list(
    llm_definition: &LlmDefinition,
    unsplit_summarized_feature: &str,
) -> Result<bool, Box<dyn Error>> {
    let (instruction_check, _, _) = boolean_classifier::classify(
        llm_definition,
        Some(unsplit_summarized_feature),
        Some("Is the attached feature a list of content split into discrete entries?"),
        None,
        None,
        Some(4),
    )
    .await?;
    Ok(instruction_check)
}

async fn check_seperator(
    llm_definition: &LlmDefinition,
    unsplit_summarized_feature: &str,
    sep: &str,
) -> Result<bool, Box<dyn Error>> {
    let sep_prompt = format!(
        r#"The attached feature is a list. We need to split it using regex.
            If we used '{}' as a seperator, would the list be properly split into discrete items?"#,
        sep
    );

    let response = boolean_classifier::classify(
        llm_definition,
        Some(unsplit_summarized_feature),
        Some(&sep_prompt),
        None,
        None,
        Some(4),
    )
    .await?;
    let (check, _, _) = response;
    Ok(check)
}
