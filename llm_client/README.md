# The Easiest Rust Interface for Local LLMs 

```toml
# For Mac (CPU and GPU), windows (CPU and CUDA), or linux (CPU and CUDA)
llm_client="*"
```

This will download and build [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md). See [build.md](../docs/build.md) for other features and backends like mistral.rs. 

```rust
use llm_client::prelude::*;

let llm_client = LlmClient::llama_cpp()
    .mistral7b_instruct_v0_3() // Uses a preset model
    .init() // Downloads model from hugging face and starts the inference interface
    .await?;
```

Several of the most common models are available as presets. Loading from local models is also fully supported. See [models.md](./docs/models.md) for more information.

# An Interface for Deterministic Signals from Probabilistic LLM Vibes

## Reasoning with Primitive Outcomes

A constraint enforced CoT process for reasoning. First, we get the LLM to 'justify' an answer in plain english. This allows the LLM to 'think' by outputting the stream of tokens required to come to an answer. Then we take that 'justification', and prompt the LLM to parse it for the answer. See [the workflow for implementation details](./src/workflows/reason/one_round.rs).

- Currently supporting returning booleans, u32s, and strings from a list of options

- Can be 'None' when ran with `return_optional_primitive()`

```rust
    // boolean outcome
    let reason_request = llm_client.reason().boolean();
    reason_request
        .instructions()
        .set_content("Does this email subject indicate that the email is spam?");
    reason_request
        .supporting_material()
        .set_content("You'll never believe these low, low prices 💲💲💲!!!");
    let res: bool = reason_request.return_primitive().await.unwrap();
    assert_eq!(res, true);

    // u32 outcome
    let reason_request = llm_client.reason().integer();
    reason_request.primitive.lower_bound(0).upper_bound(10000);
    reason_request
        .instructions()
        .set_content("How many times is the word 'llm' mentioned in these comments?");
    reason_request
        .supporting_material()
        .set_content(hacker_news_comment_section);
    // Can be None
    let response: Option<u32> = reason_request.return_optional_primitive().await.unwrap();
    assert!(res > Some(9000));

    // string from a list of options outcome
    let mut reason_request = llm_client.reason().exact_string();
    reason_request
        .instructions()
        .set_content("Based on this readme, what is the name of the creator of this project?");
    reason_request
        .supporting_material()
        .set_content(llm_client_readme);
    reason_request
        .primitive
        .add_strings_to_allowed(&["shelby", "jack", "camacho", "john"]);
    let response: String = reason_request.return_primitive().await.unwrap();
    assert_eq!(res, "shelby");
```

See [the reason example for more](./examples/reason.rs)

## Decisions with N number of Votes Across a Temperature Gradient 

Uses the same process as above N number of times where N is the number of times the process must be repeated to reach a consensus. We dynamically alter the temperature to ensure an accurate consensus. See [the workflow for implementation details](./src/workflows/reason/decision.rs).

- Supports primitives that implement the reasoning trait

- The consensus vote count can be set with `best_of_n_votes()`

- By default `dynamic_temperture` is enabled, and each 'vote' increases across a gradient

```rust
    // An integer decision request
    let decision_request = llm_client.reason().integer().decision();
    decision_request.best_of_n_votes(5); 
    decision_request
        .instructions()
        .set_content("How many fingers do you have?");
    let response = decision_request.return_primitive().await.unwrap();
    assert_eq!(response, 5);
```

See [the decision example for more](./examples/decision.rs)

## Structured Outputs and NLP

- Data extraction, summarization, and semantic splitting on text.

- Currently implemented NLP workflows are url extraction. 

See [the extract_urls example](./examples/extract_urls.rs)

## Basic Primitives

A generation where the output is constrained to one of the defined primitive types. See [the currently implemented primitive types](./src/primitives/mod.rs). These are used in other workflows, but only some are used as the output for specific workflows like reason and decision.

- These are fairly easy to add, so feel free to open an issue if you'd like one added.

See [the basic_primitive example](./examples/basic_primitive.rs)

## API LLMs

- Basic support for API based LLMs. Currently, anthropic, openai, perplexity

- Perplexity does not *currently* return documents, but it does create it's responses from live data

```rust
    let llm_client = LlmClient::perplexity().sonar_large().init();
    let mut basic_completion = llm_client.basic_completion();
    basic_completion
        .prompt()
        .add_user_message()
        .set_content("Can you help me use the llm_client rust crate? I'm having trouble getting cuda to work.");
    let response = basic_completion.run().await?;
```

See [the basic_completion example](./examples/basic_completion.rs)


## Configuring Requests

- All requests and workflows implement the `RequestConfigTrait` which gives access to the parameters sent to the LLM

- These settings are normalized across both local and API requests

```rust
    let llm_client = LlmClient::llama_cpp()
        .available_vram(48)
        .mistral7b_instruct_v0_3()
        .init()
        .await?;

    let basic_completion = llm_client.basic_completion();

    basic_completion
        .temperature(1.5)
        .frequency_penalty(0.9)
        .max_tokens(200);
```
See [See all the settings here](../llm_interface/src/requests/req_components.rs)


## More Resouces

See the [main repo](https://github.com/ShelbyJenkins/llm_client) for more documentation.

