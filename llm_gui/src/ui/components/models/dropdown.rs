use std::rc::Rc;

use dioxus::prelude::*;

use crate::ui::{components, LlmGuiState};

#[component]
pub fn ModelDropdown() -> Element {
    let llm_gui_cx = use_context::<LlmGuiState>();
    let mut search_query = use_signal(String::new);
    let mut input_element: Signal<Option<Rc<MountedData>>> = use_signal(|| None);
    let available_memory_gigabytes = llm_gui_cx.available_memory_gigabytes();
    let filtered_models = use_resource(move || async move {
        let query = ReadOnlySignal::new(search_query);
        llm_gui_cx.model_query(query()).unwrap()
    });

    rsx! {
        div {
            class: "dropdown",
            onclick: move |_| async move {
                if let Some(input) = input_element.cloned() {
                    let _ = input.set_focus(true);
                }
            },
            div {
                tabindex: "0",
                role: "button",
                class: "btn border-none! -mx-2! px-2! text-xs! opacity-40 rounded-xl! h-6! btn-ghost",
                {(llm_gui_cx.current_model)().base.friendly_name}
            }
            ul {
                tabindex: "0",
                class: "dropdown-content bg-base-200 rounded-box z-1 w-[55vw] shadow-sm rounded-t-2xl!",
                input {
                    r#type: "text",
                    placeholder: "Search",
                    class: "sticky top-2 input input-bordered! outline-none! ml-2 mt-2 border-3! rounded-xl!  z-50",
                    onmounted: move |cx| input_element.set(Some(cx.data())),
                    oninput: move |e| {
                        search_query.set(e.value());
                    },
                }
                span { class: "text-xs opacity-60 ml-10",
                    "Available Memory: {available_memory_gigabytes}"
                }
                div { class: "relative mt-2",
                    ul { class: "absolute inset! inset-t-0!  overflow-y-scroll  list bg-base-200 w-full rounded-box shadow-md px-2 gap-1 max-h-[70vh] rounded-b-2xl! divide-dotted! divide-y-1! divide-base-100!",
                        for model in filtered_models.cloned().unwrap_or_default() {
                            components::ModelCard { model }
                        }
                    }
                }
            }
        }
    }
}
