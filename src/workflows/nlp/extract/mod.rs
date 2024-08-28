// pub mod _entities;
pub mod urls;

use crate::components::base_request::BaseLlmRequest;

pub struct Extract {
    pub base_req: BaseLlmRequest,
}

impl Extract {
    pub fn new(base_req: BaseLlmRequest) -> Self {
        Self { base_req }
    }

    pub fn urls(self) -> urls::ExtractUrls {
        urls::ExtractUrls::new(self.base_req)
    }

    // pub fn entities(self) -> _entities::ExtractEntities {
    //     _entities::ExtractEntities::new(self.base_req)
    // }
}
