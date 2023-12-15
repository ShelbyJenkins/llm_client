use llm_client::providers::llama_cpp::{model_loader::download_model, models};
// cargo run -p llm_client --bin model_loader_cli --model_url "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q8_0.gguf"

#[tokio::main]
pub async fn main() {
    let matches = clap::Command::new("Model Downloader")
        .version("1.0")
        .about("Downloads and sets up models")
        .arg(
            clap::Arg::new("model_url")
                .help("The model URL")
                .long("model_url")
                .required(true),
        )
        .arg(
            clap::Arg::new("model_token")
                .help("HF token")
                .long("model_token")
                .required(false),
        )
        .get_matches();

    let model_url = matches.get_one::<String>("model_url").unwrap();
    let model_token: Option<&String> = matches.get_one::<String>("model_token");
    let (model_id, model_filename) = models::convert_url_to_hf_format(model_url);
    download_model(&model_id, &model_filename, model_token.cloned()).await;
}
