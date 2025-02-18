use dioxus::prelude::*;

use crate::ui::LlmGuiState;

#[component]
pub fn Response() -> Element {
    let llm_gui_cx = use_context::<LlmGuiState>();

    rsx! {
        div { class: "leading-[1.65rem] overflow-y-scroll pb-2 pr-9",
            p {
                {(llm_gui_cx.response)()}
                if (llm_gui_cx.inference_in_progress)() {
                    span { class: "loading loading-dots loading-xs" }
                }
            }
        }
    }
}
