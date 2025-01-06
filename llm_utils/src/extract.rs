use linkify::{LinkFinder, LinkKind};
use std::{collections::HashSet, str::FromStr};
use url::Url;

pub fn extract_urls<T: AsRef<str>>(input: T) -> Vec<Url> {
    let mut unique_urls = HashSet::new();

    LinkFinder::new()
        .kinds(&[LinkKind::Url])
        .links(input.as_ref())
        .filter_map(|link| Url::from_str(link.as_str()).ok())
        .filter(|url| unique_urls.insert(url.clone()))
        .collect()
}
