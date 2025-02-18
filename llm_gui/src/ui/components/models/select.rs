use dioxus::prelude::*;

use crate::{
    api::{LlmGuiLocalModel, LoadState, ModelState},
    ui::LlmGuiState,
};

#[component]
pub fn ModelSelect(model: LlmGuiLocalModel) -> Element {
    let mut llm_gui_cx = use_context::<LlmGuiState>();
    let model_id = model.base.model_id.clone();
    rsx! {
        div { class: "text-xs flex flex-row justify-end items-center",
            if model.model_state == ModelState::Selected(LoadState::Loaded) {
                button { class: "flex flex-col justify-items-center justify-center items-center w-20 h-10 border-primary border-1 rounded-2xl gap-1 py-1 opacity-60",
                    svg { "viewBox": "0 0 24 24",
                        g {
                            fill: "var(--color-primary)",
                            stroke: "var(--color-primary)",
                            stroke_width: "2",
                            stroke_linejoin: "round",
                            stroke_linecap: "round",
                            path { d: "M6 3L20 12 6 21 6 3z" }
                        }
                    }
                    "loaded"
                }
            } else if model.model_state == ModelState::Selected(LoadState::Unloaded) {
                button { class: "flex flex-col justify-items-center justify-center items-center w-20 h-10 border-primary border-1 rounded-2xl gap-1 py-1 opacity-60",
                    svg { "viewBox": "0 0 24 24",
                        g {
                            fill: "var(--color-primary)",
                            stroke: "var(--color-primary)",
                            stroke_width: "2",
                            stroke_linejoin: "round",
                            stroke_linecap: "round",
                            path { d: "M6 3L20 12 6 21 6 3z" }
                        }
                    }
                    "selected"
                }
            } else if model.model_state == ModelState::Downloaded {
                button {
                    class: "flex flex-col justify-items-center justify-center items-center w-20 h-10 hover:border-primary border-base-200 border-1 rounded-2xl gap-1 py-1",
                    onclick: move |_| {
                        let model_id = model_id.clone();
                        spawn(async move {
                            llm_gui_cx.select_model(&model_id).await;
                        });
                    },
                    svg { "viewBox": "0 0 24 24",
                        g {
                            fill: "var(--color-accent)",
                            stroke: "var(--color-accent)",
                            stroke_width: "2",
                            stroke_linejoin: "round",
                            stroke_linecap: "round",
                            path { d: "M6 3L20 12 6 21 6 3z" }
                        }
                    }
                    "use"
                }
            } else {
                button {
                    class: "flex flex-col justify-items-center justify-center items-center w-20 h-10 hover:border-secondary border-base-200 border-1 rounded-2xl gap-1 py-1",
                    onclick: move |_| {
                        let model_id = model_id.clone();
                        spawn(async move {
                            llm_gui_cx.select_model(&model_id).await;
                        });
                    },
                    svg { "viewBox": "0 0 24 24",
                        g {
                            fill: "var(--color-secondary)",
                            stroke: "var(--color-secondary)",
                            stroke_width: "2",
                            stroke_linejoin: "round",
                            stroke_linecap: "round",
                            path { d: "M21 6L12 20 3 6 21 6z" }
                        }
                    }
                    "download"
                }
            }
        }
    }
}
