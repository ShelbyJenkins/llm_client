use dioxus::prelude::*;

use crate::{api::LlmGuiLocalModel, ui::components::models::select::ModelSelect};

#[component]
pub fn ModelCard(model: LlmGuiLocalModel) -> Element {
    let recommended_quantization_level = model.recommended_quantization_level();
    let estimated_memory_usage = model.estimated_memory_usage();
    rsx! {
        li { class: "list-row gap-2 flex flex-row justify-items-end! items-center! w-full rounded-lg! pt-0! pb-3!",
            ModelSelect { model: model.clone() }
            div {
                span { class: "text-sm font-bold", "{model.base.friendly_name}" }

                div {
                    span { class: "text-xs opacity-60", {model.number_of_parameters()} }
                    span { " - " }
                    span {
                        link {
                            class: "text-xs opacity-60 inline-block link link-primary",
                            href: model.model_repo_link(),
                            "{model.organization()} Model Repo"
                        }
                    }
                    span { " - " }

                    span { " " }
                    span {
                        link {
                            class: "text-xs opacity-60 inline-block link link-primary",
                            href: model.gguf_repo_link(),
                            "GGUF Repo"
                        }
                    }
                }
            
            }
            if let Some(recommended_quantization_level) = recommended_quantization_level {
                div { class: "text-xs opacity-60 flex flex-col list-col-grow items-start!",
                    span { "{recommended_quantization_level} bit quant" }
                    span { "{estimated_memory_usage} VRAM" }
                }
            } else {
                div { class: "text-xs opacity-60 flex flex-col list-col-grow items-start!",
                    span { "No available quant" }
                    span { "Smallest quant exceeds available memory" }
                }
            }
        }
    }
}
