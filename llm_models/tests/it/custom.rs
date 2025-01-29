use llm_models::{GgufLoader, GgufLoaderTrait};

#[test]
fn load_local_basic() {
    let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
            .load()
           .unwrap();

    println!("{:#?}", model);
}

#[test]
fn load_local_with_config() {
    let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
        .local_config_path("/workspaces/test/llm_utils/src/models/local_model/gguf/preset/llama/llama3_1_8b_instruct/config.json")
            .load()
           .unwrap();

    println!("{:#?}", model);
}

#[test]
fn load_hf_basic() {
    let model = GgufLoader::default()
        .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
            .load()
            .unwrap();

    println!("{:#?}", model);
}

#[test]
fn load_hf_with_config() {
    let model = GgufLoader::default()
        .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
            .load()
            .unwrap();

    println!("{:#?}", model);
}
