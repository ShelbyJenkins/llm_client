use llm_models::*;

fn main() {
    // Using
    let _model = GgufLoader::default()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        .load()
        .unwrap();

    // By default we attempt to extract everything we need from the GGUF file.
    // If you need to specifiy the tokenizer or chat template to use, you can add a hf repo to load from.
    let _model = GgufLoader::default()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        .load()
        .unwrap();

    // model.local_model_path can now be used to load the model into the inference engine.
}
