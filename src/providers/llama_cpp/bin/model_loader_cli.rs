use llm_client::providers::llama_cpp::model_loader::download_model;

// cargo run -p llm_client --bin model_loader_cli -- --model_id TheBloke/zephyr-7B-alpha-GGUF --model_filename zephyr-7b-alpha.Q2_K.gguf
// cargo run -p llm_client --bin model_loader_cli -- --model_id TheBloke/Mistral-7B-Instruct-v0.2-GGUF --model_filename mistral-7b-instruct-v0.2.Q8_0.gguf

#[tokio::main]
pub async fn main() {
    let matches = clap::Command::new("Model Downloader")
        .version("1.0")
        .about("Downloads and sets up models")
        .arg(
            clap::Arg::new("model_id")
                .help("The model URL")
                .long("model_id")
                .required(true),
        )
        .arg(
            clap::Arg::new("model_filename")
                .help("The model version")
                .long("model_filename")
                .required(true),
        )
        .arg(
            clap::Arg::new("model_token")
                .help("HF token")
                .long("model_token")
                .required(false),
        )
        .get_matches();

    let model_id = matches.get_one::<String>("model_id").unwrap();
    let model_filename = matches.get_one::<String>("model_filename").unwrap();
    let model_token: Option<&String> = matches.get_one::<String>("model_token");

    download_model(model_id, model_filename, model_token.cloned()).await;
}
