use super::*;
use std::collections::HashSet;

pub trait TestItem {
    fn test_level(&self) -> u8;
    fn is_optional(&self) -> bool;
}

pub struct BooleanTests {
    pub cases: Vec<BooleanTest>,
}

impl BooleanTests {
    pub fn check_results(&self) {
        self.cases.iter().for_each(|test| test.check_result());
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BooleanTest {
    pub question: String,
    pub result: Option<bool>,
    correct_answer: Option<bool>,
    test_level: u8,
}

impl BooleanTest {
    pub fn check_result(&self) {
        if self.result == self.correct_answer {
            println!(
                "\n\nBoolean Test\n{:>2}question: '{}'\n{:>2}🟢 correct response: '{:?}'",
                "", self.question, "", self.result
            );
        } else {
            println!(
                "\n\nBoolean Test\n{:>2}question: '{}'\n{:>2}🔴 incorrect response: '{:?}'",
                "", self.question, "", self.result
            );
        };
    }
}

impl TestItem for BooleanTest {
    fn test_level(&self) -> u8 {
        self.test_level
    }

    fn is_optional(&self) -> bool {
        self.correct_answer.is_none()
    }
}

pub struct IntegerTests {
    pub cases: Vec<IntegerTest>,
}

impl IntegerTests {
    pub fn check_results(&self) {
        self.cases.iter().for_each(|test| test.check_result());
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IntegerTest {
    pub question: String,
    pub result: Option<u32>,
    correct_answer: Option<u32>,
    test_level: u8,
}

impl IntegerTest {
    pub fn check_result(&self) {
        let outcome = if self.result == self.correct_answer {
            format!("🟢 correct response: '{:?}'", self.result)
        } else {
            format!("🔴 incorrect response: '{:?}'", self.result)
        };
        println!(
            "\n\nInteger Test\n{:>2}question: '{}'\n{:>2}correct answer: '{:?}'\n{:>2}outcome: {}",
            "", self.question, "", self.correct_answer, "", outcome
        );
    }
}

impl TestItem for IntegerTest {
    fn test_level(&self) -> u8 {
        self.test_level
    }

    fn is_optional(&self) -> bool {
        self.correct_answer.is_none()
    }
}

pub struct ExactStringTests {
    pub cases: Vec<ExactStringTest>,
}

impl ExactStringTests {
    pub fn check_results(&self) {
        self.cases.iter().for_each(|test| test.check_result());
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExactStringTest {
    pub question: String,
    pub result: Option<String>,
    correct_answer: Option<String>,
    pub options: Vec<String>,
    test_level: u8,
}

impl ExactStringTest {
    pub fn check_result(&self) {
        let outcome = if self.result == self.correct_answer {
            format!("🟢 correct response: '{:?}'", self.result)
        } else {
            format!("🔴 incorrect response: '{:?}'", self.result)
        };
        println!(
            "\n\nExactString Test\n{:>2}question: '{}'\n{:>2}correct answer: '{:?}'\n{:>2}outcome: {}",
            "", self.question, "", self.correct_answer, "", outcome
        );
    }
}

impl TestItem for ExactStringTest {
    fn test_level(&self) -> u8 {
        self.test_level
    }

    fn is_optional(&self) -> bool {
        self.correct_answer.is_none()
    }
}

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ExtractEntitiesTest {
//     pub supporting_material: String,
//     test_level: u8,
// }

// impl ExtractEntitiesTest {
//     pub fn check_result(
//         &self,
//         res: llm_client::workflows::nlp::extract::_entities::ExtractEntitiesResult,
//     ) {
//         let outcome = if res.results.is_some() {
//             format!("🟢\n{}", res)
//         } else {
//             format!("🔴\n{}", res)
//         };

//         // Trim supporting_material to first 25 chars and add ellipsis if longer
//         let trimmed_material = if self.supporting_material.len() > 25 {
//             format!("{}...", &self.supporting_material[..25])
//         } else {
//             self.supporting_material.clone()
//         };

//         println!(
//             "\n\nExtractEntities Test\nsupporting_material: '{}'\n{}",
//             trimmed_material, outcome
//         );
//     }
// }

// impl TestItem for ExtractEntitiesTest {
//     fn test_level(&self) -> u8 {
//         self.test_level
//     }

//     fn is_optional(&self) -> bool {
//         false
//     }
// }

#[derive(Debug, Deserialize, Serialize)]
pub struct ExtractUrlsTest {
    pub instructions: String,
    pub supporting_material: String,
    correct_answers: Option<Vec<String>>,
    test_level: u8,
}

impl ExtractUrlsTest {
    pub fn check_result(&self, res: llm_client::workflows::nlp::extract::urls::ExtractUrlResult) {
        let correct_urls: Option<HashSet<Url>> = self.correct_answers.as_ref().map(|urls| {
            urls.iter()
                .filter_map(|url_str| Url::parse(url_str).ok())
                .collect()
        });

        let res_set: Option<HashSet<Url>> =
            res.results.clone().map(|urls| urls.into_iter().collect());

        let outcome = if res_set == correct_urls {
            format!(
                "🟢 correct response:\n{}",
                Self::format_optional_urls(&res.results)
            )
        } else {
            format!(
                "🔴 incorrect response:\n{}",
                Self::format_optional_urls(&res.results)
            )
        };

        // Trim supporting_material to first 25 chars and add ellipsis if longer
        let trimmed_material = if self.supporting_material.len() > 25 {
            format!("{}...", &self.supporting_material[..25])
        } else {
            self.supporting_material.clone()
        };

        println!(
        "\n\nExtractUrls Test\ninstructions: '{}'\nsupporting_material: '{}'\ncorrect_answers:\n{}\noutcome: {}\ncriteria: {}\nduration: {:?}",
         self.instructions, trimmed_material, Self::format_optional_urls(&self.correct_answers), outcome, res.criteria, res.duration
    );
    }

    fn format_optional_urls(urls: &Option<Vec<impl AsRef<str>>>) -> String {
        match urls {
            Some(urls) if !urls.is_empty() => urls
                .iter()
                .map(|url| format!("  - {}", url.as_ref()))
                .collect::<Vec<_>>()
                .join("\n"),
            Some(_) => "  (empty list)".to_string(),
            None => "  (null)".to_string(),
        }
    }
}

impl TestItem for ExtractUrlsTest {
    fn test_level(&self) -> u8 {
        self.test_level
    }

    fn is_optional(&self) -> bool {
        self.correct_answers.is_none()
    }
}
