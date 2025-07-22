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
[llm_client will return](#what-happened-to-llm_client)
<!-- cargo-rdme start -->

lmcpp – `llama.cpp`'s [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) for Rust
=============================================================================================================

## Fully Managed
- **Automated Toolchain** – Downloads, builds, and manages the `llama.cpp` toolchain with [`LmcppToolChain`].  
- **Supported Platforms** – Linux, macOS, and Windows with CPU, CUDA, and Metal support.  
- **Multiple Versions** – Each release tag and backend is cached separately, allowing you to install multiple versions of `llama.cpp`.

## Blazing Fast UDS
- **UDS IPC** – Integrates with `llama-server`’s Unix-domain-socket client on Linux, macOS, and Windows.  
- **Fast!** – Is it faster than HTTP? Yes. Is it *measurably* faster? Maybe.

## Fully Typed / Fully Documented
- **Server Args** – *All* `llama-server` arguments implemented by [`ServerArgs`].  
- **Endpoints** – Each endpoint has request and response types defined.
- **Good Docs** – Every parameter was researched to improve upon the original `llama-server` documentation.

## CLI Tools & Web UI
- **`lmcpp-toolchain-cli`** – Manage the `llama.cpp` toolchain: download, build, cache.  
- **`lmcpp-server-cli`**    – Start, stop, and list servers.  
- **Easy Web UI** – Use [`LmcppServerLauncher::webui`] to start with HTTP *and* the Web UI enabled.

---

```rust
use lmcpp::*;

fn main() -> LmcppResult<()> {
    let server = LmcppServerLauncher::builder()
        .server_args(
            ServerArgs::builder()
                .hf_repo("bartowski/google_gemma-3-1b-it-qat-GGUF")?
                .build(),
        )
        .load()?;

    let res = server.completion(
        CompletionRequest::builder()
            .prompt("Tell me a joke about Rust.")
            .n_predict(64),
    )?;

    println!("Completion response: {:#?}", res.content);
    Ok(())
}
```

```sh,no_run
cargo run --bin lmcpp-server-cli -- --webui
// Or with a specific model from URL:
cargo run --bin lmcpp-server-cli -- --webui -u https://huggingface.co/bartowski/google_gemma-3-1b-it-qat-GGUF/blob/main/google_gemma-3-1b-it-qat-Q4_K_M.gguf
// Or with a specific local model:
cargo run --bin lmcpp-server-cli -- --webui -l /path/to/local/model.gguf
```

---

## How It Works

```text
Your Rust App
      │
      ├─→ LmcppToolChain        (downloads / builds / caches)
      │         ↓
      ├─→ LmcppServerLauncher   (spawns & monitors)
      │         ↓
      └─→ LmcppServer           (typed handle over UDS*)
                │
                ├─→ completion()       → text generation
                └─→ other endpoints    → fill-in-the-middle
```
*Windows transparently swaps in a named pipe.*

---

### Endpoints ⇄ Typed Helpers
| HTTP Route          | Helper on `LmcppServer` | Request type            | Response type          |
|---------------------|-------------------------|-------------------------|------------------------|
| `POST /completion`  | `completion()`          | [`CompletionRequest`]   | [`CompletionResponse`] |
| `POST /infill`      | `infill()`              | [`InfillRequest`]       | [`CompletionResponse`] |
| `POST /embeddings`  | `embeddings()`          | [`EmbeddingsRequest`]   | [`EmbeddingsResponse`] |
| `POST /tokenize`    | `tokenize()`            | [`TokenizeRequest`]     | [`TokenizeResponse`]   |
| `POST /detokenize`  | `detokenize()`          | [`DetokenizeRequest`]   | [`DetokenizeResponse`] |
| `GET  /props`       | `props()`               | –                       | [`PropsResponse`]      |
| *custom*            | `status()` ¹            | –                       | [`ServerStatus`]       |
| *Open AI*           | `open_ai_v1_*()`        | – [`serde_json::Value`] | [`serde_json::Value`]  |

¹ Internal helper for server health.

---
## Supported Platforms
| Platform   | CPU | CUDA | Metal | Binary Sources       |
|------------|-----|------|-------|----------------------|
| Linux x64  | ✅ | ✅ | –  | Pre-built + Source |
| macOS ARM  | ✅ | –  | ✅ | Pre-built + Source |
| macOS x64  | ✅ | –  | ✅ | Pre-built + Source |
| Windows x64| ✅ | ✅ | –  | Pre-built + Source |

---

<!-- cargo-rdme end -->

## What happened to llm_client?

And `llm_devices`, `llm_testing`, `llm_prompt`, `llm_models`, and the other crates that used to be in this repo?

* I moved cross country and took a long time off.
* Supporting local *and* cloud models exploded complexity.
* I realized the goals of llm_client and the goals of most people did not overlap; most people just want an Open AI compatible endpoint. They didn't want a new DSL for building AI agents or low level workflow builders.

So, I decided to narrow my scope, and *start fresh*. The new goal of this project is to be the best Llama.cpp integration possible.

And so, this repo will stick to the barebones and low level LLM implementation details. Shortly I will rework `llm_prompt`, and `llm_models` towards this goal.

Any further tooling built on top of that, will be a different project, which I will link to here once published.

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

