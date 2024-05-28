<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->


# llm_client: structured text, decision making, and benchmarks. A user friendly interface to write once and run on any local or API model.

> LLMs aren't chat bots; They're information arbitrage machines and prompts are database queries.

* Structure the outputs of generated text, make decisions from novel inputs, and classify data.
* The easiest interface possible to deploy and test the same logic to various LLM backends.
* A local first and embedded model. Meant to be built and ran in-process with your business logic. No stand alone servers.

### LLMs as decision makers 🚦
- What previously took dozens, hundreds, or thousands of `if statements` for a specific requirement, can now be done with a few lines of code across novel inputs.

- llm_client uses what might be a novel process for LLM decision making. First, we get the LLM to 'justify' an answer in plain english. This allows the LLM to 'think' by outputting the stream of tokens required to come to an answer. Then we take that 'justification', and prompt the LLM to parse it for the answer.

```rust
    let res: bool = llm_client.decider().boolean()
        .system_content("Does this email subject indicate that the email is spam?")
        .user_content("You'll never believe these low, low prices 💲💲💲!!!")
        .run().await?;
    assert_eq!(res, true);

    let res: u16 = llm_client.decider().integer()
        .system_content("How many times is the word 'llm' mentioned in these comments?")
        .user_content(hacker_news_comment_section)
        .run().await?;
    assert!(res > 1);

    let res: String = llm_client.decider().custom()
        .system_content("Based on this resume, what is the users first name?")
        .user_content(shelby_resume)
        .add_choice("shelby")
        .add_choice("jack")
        .add_choice("camacho")
        .add_choice("john")
        .run().await?;
    assert!(res != "shelby");
```

### Structured text 📝
- 'Some people, when confronted with a problem, think "I know, I'll use regular expressions." Now they have two problems.' Using Regex to parse and structure the output of LLMs puts an exponent over this old joke.

- llm_client implements structuring text through logit_bias and grammars. Of the two, [grammars is the most powerful, and allows for very granular controls of text generation.](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) Logit bias, through having wider support, is less useful as it relies on adjusting the probabilities of individual tokens.

```rust
    let res: Vec<String> = llm_client.text().grammar_list()
        .system_content("ELI5 each topic in this text.")
        .user_content(wikipedia_article)
        .max_items(5)
        .min_items(3)
        .run().await?;
    assert_eq!(res.len() > 3);

    let res: String = llm_client.text().grammar_text()
        .system_content("Summarize this mathematical funtion in plain english. Do not use notation.")
        .user_content(wikipedia_article)
        .restrict_extended_punctuation()
        .run().await?;
    assert!(!res.contains('('));
    assert!(!res.contains('['));

    let res: String = llm_client.text().logit_bias_text()
        .system_content("Summarize this article")
        .user_content(wikipedia_article)
        .add_logit_bias_from_word("delve", -100.0);
        .run().await?;
    assert!(!res.contains("delve"));

```

### LLM -> LLMs 🤹
- The same code across multiple LLMs.

- This makes benchmarking multiple LLMs really easy. Checkout src/bechmark for an example.

```rust
    pub async fn chatbot(llm_client: &LlmClient, user_input: &str) -> Result<String> {
        llm_client.text().basic_text()
            .system_content("You're a kind robot.")
            .user_content(user_input)
            .temperature(0.5)
            .max_tokens(2)
            .run().await
    }

    let llm_client = LlmClient::llama_backend()
        .mistral_7b_instruct()
        .init()
        .await?;
    assert_eq!(chatbot(&llm_client, "What is the meaning of life?").await?, "42")

    let llm_client = LlmClient::llama_backend()
        .model_url("https://huggingface.co/your_cool_model_Q5_K.gguf")
        .init()
        .await?;
    assert_eq!(chatbot(&llm_client, "What is the meaning of life?").await?, "42")

    let llm_client = LlmClient::openai_backend().gpt_4_o().init()?;
    assert_eq!(chatbot(&llm_client, "What is the meaning of life?").await?, "42")

    let llm_client = LlmClient::anthropic_backend().claude_3_opus().init()?;
    assert_eq!(chatbot(&llm_client, "What is the meaning of life?").await?, "42")
```

## Minimal Example

```rust
use llm_client::LlmClient;

// Setting available_vram will load the largest quantized model that can fit the given vram.
let llm_client = LlmClient::llama_backend().available_vram(16).llama_3_8b_instruct().init().await?;

let res = llm_client.text().basic_text().user_content("Hello world?").run().await?;

assert_eq!(res, "Hello world!");
```

## Examples

* [A simple example.](./examples/llm_client.rs)

## Guides

* [Limiting power in Nvidia GPUs](./guides/nv-power-limit.md)


## Installation

llm_client currently relies on llama.cpp. As it's a c++ project, it's not bundled in the crate. In the near future, llm_client will support mistral-rs, an inference backend built in Candle and supporting great features like ISQ. Once integration is complete, llm_client will be pure Rust and can be installed as just a crate.

### If *only* using OpenAi and/or Anthropic
-  Add to cargo.toml:
```toml
[dependencies]
llm_client = "*"
```
- Add API key
    - Add `OPENAI_API_KEY=<key>` and/or `ANTHROPIC_API_KEY=<key>` to your `.env` file
    - Or use the `api_key` function in the backend builder functions


### If using Llama.cpp and/or external APIs
- Clone repo:
```cmd
git clone --recursive https://github.com/ShelbyJenkins/llm_client.git
cd llm_client
```
- Add to cargo.toml:
```toml
[dependencies]
llm_client = {path="../llm_client"}
```
- Optional: Build devcontainer from `llm_client/.devcontainer/devcontainer.json` This will build out a dev container with nvidia dependencies installed. 

- Build llama.cpp (<a href="https://github.com/ggerganov/llama.cpp">This is dependent on your hardware. Please see full instructions here</a>):
  ```cmd
  // Example nvidia gpu build
  cd llm_client/src/llm_backends/llama_cpp/llama_cpp
  make LLAMA_CUDA=1
  ```




## Roadmap

* Migrate from llama.cpp to <a href="https://github.com/EricLBuehler/mistral.rs">mistral-rs</a>. This would greatly simplify consuming as an embedded crate. It's currently a WIP. It may also end up that llama.cpp is behind a feature flag as a fallback.
* Additional deciders: Multiple reponse deciders.
* Classifer, summarizer, map reduce agents.
* Extend grammar support: Custom grammars, JSON support.
* More external APIs such as Google, AWS, Groq, and LLM aggregators and routers.
* Dream roadmap item: web ui for streaming output of multiple LLMs for a single prompt. Because we already do this with Claude and ChatGPT anyways don't we? 
  
### Dependencies 
<a href="https://github.com/64bit/async-openai">async-openai</a> is used to interact with the OpenAI API. A modifed version of the async-openai crate is used for the Llama.cpp server. If you just need an OpenAI API interface, I suggest using the async-openai crate.

<a href="https://github.com/mochi-neko/clust">clust</a> is used to interact with the Anthropic API. If you just need an Anthropic API interface, I suggest using the clust crate.

<a href="https://github.com/shelbyJenkins/llm_utils">llm_utils</a> is a sibling crate that was split from the llm_client. If you just need prompting, tokenization, model loading, etc, I suggest using the llm_utils crate on it's own.


## Contributing

This is my first Rust crate. All contributions or feedback is more than welcomed!

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Shelby Jenkins - Here or Linkedin 


<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ShelbyJenkins/llm_client.svg?style=for-the-badge
[contributors-url]: https://github.com/ShelbyJenkins/llm_client/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ShelbyJenkins/llm_client.svg?style=for-the-badge
[forks-url]: https://github.com/ShelbyJenkins/llm_client/network/members
[stars-shield]: https://img.shields.io/github/stars/ShelbyJenkins/llm_client.svg?style=for-the-badge
[stars-url]: https://github.com/ShelbyJenkins/llm_client/stargazers
[issues-shield]: https://img.shields.io/github/issues/ShelbyJenkins/llm_client.svg?style=for-the-badge
[issues-url]: https://github.com/ShelbyJenkins/llm_client/issues
[license-shield]: https://img.shields.io/github/license/ShelbyJenkins/llm_client.svg?style=for-the-badge
[license-url]: https://github.com/ShelbyJenkins/llm_client/blob/master/LICENSE.txt
<!-- [linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com -->

