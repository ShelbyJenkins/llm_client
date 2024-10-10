use gguf::GgufLoader;
use llm_models::local_model::*;

fn main() {
    // Using
    let _model = GgufLoader::default()
    .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
        .load()
       .unwrap();

    // By default we attempt to extract everything we need from the GGUF file.
    // If you need to specifiy the tokenizer or chat template to use, you can add a local_config_path.
    let _model = GgufLoader::default()
    .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
    .local_config_path("/workspaces/test/llm_utils/src/models/local_model/gguf/preset/llama/llama3_1_8b_instruct/config.json")
        .load()
       .unwrap();

    // model.local_model_path can now be used to load the model into the inference engine.
}
