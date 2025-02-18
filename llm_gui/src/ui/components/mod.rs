mod header;
mod models;
mod query;
mod response;

use dioxus::prelude::*;
pub use header::Header;
pub use models::*;
pub use query::Query;
pub use response::Response;

use crate::{
    api::{available_memory_bytes, get_models, LlmGuiLocalModel},
    ui::global_state::LlmGuiState,
};

pub fn llm_gui() -> Element {
    let get_models_fn: Resource<Vec<LlmGuiLocalModel>> =
        use_server_future(|| async move { get_models().await.unwrap() })?;
    let available_memory_bytes_fn: Resource<u64> =
        use_server_future(|| async move { available_memory_bytes().await.unwrap() })?;

    let _ = use_context_provider(|| LlmGuiState::new(get_models_fn, available_memory_bytes_fn));

    rsx! {
        Header {}
        div { id: "llm-gui-app", class: "h-screen w-full",
            div {
                id: "llm-gui-main",
                class: "mx-auto flex h-full gap-6 w-full max-w-3xl flex-1 flex-col relative",

                Query {}

                div { class: "relative w-full h-full -z-0",
                    div { class: "absolute inset-0 bottom-0 flex flex-1 flex-col gap-2 px-4 bg-base-300 rounded-t-2xl pt-2",
                        div { class: "sticky top-0 z-10 flex flex-col items-start",
                            div { class: "flex items-start w-auto overflow-x-visible",
                                ModelDropdown {}
                            }
                            div { class: "rounded my-1 h-[1px] w-full bg-linear-to-l from-accent/20 to-primary/5" }
                        }
                        Response {}
                    }
                }
            }
        }
    }
}
