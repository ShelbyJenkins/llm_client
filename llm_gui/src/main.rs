// Internal modules
mod api;
mod ui;

// Internal imports
#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};

#[allow(unused_imports)]
use dioxus::logger::tracing::{debug, error, info, span, trace, warn, Level};
use dioxus::prelude::*;

fn main() {
    dioxus::logger::init(Level::INFO).expect("failed to init logger");
    launch(crate::ui::llm_gui)
}
