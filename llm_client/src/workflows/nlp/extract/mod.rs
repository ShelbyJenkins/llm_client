use super::*;

pub mod urls;

pub struct Extract {
    pub base_req: CompletionRequest,
}

impl Extract {
    pub fn new(base_req: CompletionRequest) -> Self {
        Self { base_req }
    }

    pub fn urls(self) -> urls::ExtractUrls {
        urls::ExtractUrls::new(self.base_req)
    }
}
