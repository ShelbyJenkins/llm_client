use dioxus::prelude::*;

use crate::ui::LlmGuiState;

#[component]
pub fn Query() -> Element {
    let mut llm_gui_cx = use_context::<LlmGuiState>();
    let mut in_progress_class = use_signal(String::new);

    use_effect(move || {
        if (llm_gui_cx.inference_in_progress)() {
            in_progress_class.set("inference-in-progress".to_string());
        } else {
            in_progress_class.set("".to_string());
        }
    });
    rsx! {
        div { class: "relative h-14 w-full top-0 mx-auto pb-6 z-[5]",
            div { class: "absolute inset-x-0 flex pl-5 pt-2 pr-2.5 gap-2 bg-base-200 z-10 rounded-b-2xl",
                div {
                    class: "mt-1 min-h-12 h-12 focus:h-auto focus:max-h-[70vh] top-0 w-full overflow-y-auto whitespace-break-spaces break-words outline-none cursor-text caret-accent",
                    contenteditable: true,
                    enterkeyhint: "enter",
                    tabindex: "0",
                    autofocus: true,
                    onkeydown: move |e| {
                        if e.modifiers().shift() {
                            return;
                        }
                        match e.key() {
                            Key::Enter => {
                                e.prevent_default();
                                spawn(async move {
                                    llm_gui_cx.submit_inference_request().await;
                                });
                            }
                            _ => {}
                        }
                    },
                    oninput: move |e| {
                        llm_gui_cx.query.set(e.value());
                    },
                }
                div {
                    id: "prompt-submit-button-outer",
                    class: "{in_progress_class}",
                    button {
                        onclick: move |_| {
                            async move {
                                llm_gui_cx.submit_inference_request().await;
                            }
                        },
                        id: "prompt-submit-button",
                    }
                }
            }
        }
    }
}
