use llm_models::*;

fn main() {
    // Using
    let _model: LocalLlmModel = GgufLoader::default()
       .llama_3_2_1b_instruct()
        .preset_with_memory_gb(48)
        .load()
        .unwrap();

    // model.local_model_path can now be used to load the model into the inference engine.
}
