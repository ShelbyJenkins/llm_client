use dioxus::prelude::*;

#[component]
pub fn Header() -> Element {
    use document::{Link, Meta, Title};
    rsx!(
        Link {
            rel: "icon",
            href: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='80'>🫡</text></svg>",
        }
        Meta {
            name: "description",
            content: "llm_gui | A proof-of-concept GUI for the llm_client project. In 🦀 btw",
        }
        Title { "llm_gui" }
        document::Stylesheet { href: asset!("./public/tailwind.css") }
        document::Stylesheet { href: asset!("./public/style.css") }
    )
}
