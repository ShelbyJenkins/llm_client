# llm_interface: The Backend for the llm_client Crate

This crate contains the build.rs, data types, and behaviors for LLMs.

* Llama.cpp (through llama-server) 
* Various LLM APIs including support for generic OpenAI format LLMs

You can use this crate to run local LLMs and make requests to LLMs. It's set up to be easy to integrate into other projects. 

See the various `Builders` implemented in the [lib.rs](https://github.com/ShelbyJenkins/llm_client/llm_interface/src/lib.rs) file for an example of using this crate.

For a look at a higher level API and how it implements this crate, checkout the [llm_client](https://github.com/ShelbyJenkins/llm_client) crate and it's lib.rs file.

[API Docs](https://docs.rs/llm_interface/latest/llm_interface/)