### An example cascade: CoT reasoning

This is in progress. 

An example of a cascade workflow is the [one round reasoning workflow](./src/workflows/reason/one_round.rs).

First we insert a guidance round the the user message that defines the workflow, followed by a faux 'guidance' response assistant message.

```rust
flow.new_round(
"A request will be provided. Think out loud about the request. State the arguments before arriving at a conclusion with, 'Therefore, we can conclude:...', and finish with a solution by saying, 'Thus, the solution...'. With no yapping.").add_guidance_step(
&StepConfig {
    ..StepConfig::default()
},
"'no yapping' refers to a design principle or behavior where the AI model provides direct, concise responses without unnecessary verbosity or filler content. Therefore, we can conclude: The user would like to get straight to the point. Thus, the solution is to to resolve the request as efficiently as possible.",
);
```





Optionally, we add a guidance step that restates the instructions. Often LLMs suffer with following instructions in long context, so this restates what the outcome should be.

```rust
let step_config = StepConfig {
    step_prefix: None,
    grammar: SentencesPrimitive::default().grammar(),
    ..StepConfig::default()
};
flow.last_round()?
    .add_guidance_step(&step_config, format!("The user's original request was '{}'.", &instructions,));
```



In this example the work flow is run linearly as built, but it's also possible to run dynamic workflows where each step is ran one at a time and the behavior of the workflow can be dynamic based on the outcome of that step. See [extract_urls](./examples/extract_urls.rs) for an example of this.