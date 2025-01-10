use llm_client::prelude::*;

/// Extracts URLs from a given text based on the given instructions.
#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    // Using a preset model from Hugging Face
    let llm_client = LlmClient::anthropic().claude_3_haiku().init().unwrap();

    let response = llm_client.nlp().extract().urls()
        .set_instructions("Which of these is URLs would someone find useful documentation for working with the library and it's APIs? Things such as examples, guides, and the README.")
        .set_supporting_material("<a href='https://github.com/ShelbyJenkins/llm_client/blob/master/guides/nv-power-limit.md</a></div><div role='gridcell'>Guide for NVIDIA power limit configuration</div></div><div role='row'><div role='rowheader'><svg aria-label='File' class='octicon'></svg><a href='https://github.com/ShelbyJenkins/llm_client/blob/master/LICENSE'>LICENSE</a></div><div role='gridcell'>License information</div></div><div role='row'><div role='rowheader'><svg aria-label='File' class='octicon'></svg><a href='https://github.com/ShelbyJenkins/llm_client/blob/master/.gitignore'>.gitignore</a></div><div role='gridcell'>Git ignore file</div></div></div></div></div></div></body></html>").run_return_urls().await.unwrap();

    assert_eq!(
        response,
        Some(
            [url::Url::parse(
                "https://github.com/ShelbyJenkins/llm_client/blob/master/guides/nv-power-limit.md"
            )
            .unwrap(),]
            .to_vec()
        )
    );
}
