use std::{fs::File, io::Read, path::PathBuf};

use serde::Deserialize;

use crate::{BooleanTests, ExactStringTests, ExtractUrlsTest, IntegerTests, TestItem};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestLevel {
    #[default]
    IntegrationTest,
    One,
    Two,
    Three,
    All,
    Custom(usize),
}

#[derive(Default)]
pub struct TestSetsLoader {
    pub result_can_be_none: bool,
    pub test_level: TestLevel,
}

#[allow(dead_code)]
impl TestSetsLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_optional() -> Self {
        Self::default().optional(true)
    }

    pub fn optional(mut self, result_can_be_none: bool) -> Self {
        self.result_can_be_none = result_can_be_none;
        self
    }

    pub fn test_level_enum(mut self, test_level: &TestLevel) -> Self {
        self.test_level = test_level.clone();
        self
    }

    pub fn test_level_one(mut self) -> Self {
        self.test_level = TestLevel::One;
        self
    }

    pub fn test_level_two(mut self) -> Self {
        self.test_level = TestLevel::Two;
        self
    }

    pub fn test_level_three(mut self) -> Self {
        self.test_level = TestLevel::Three;
        self
    }

    pub fn test_level_all(mut self) -> Self {
        self.test_level = TestLevel::All;
        self
    }

    pub fn test_level_custom(mut self, level: usize) -> Self {
        self.test_level = TestLevel::Custom(level);
        self
    }

    pub fn boolean(&self) -> crate::Result<BooleanTests> {
        Ok(BooleanTests {
            cases: self.load_tests(self.test_file_path("boolean.json"))?,
        })
    }

    pub fn integer(&self) -> crate::Result<IntegerTests> {
        Ok(IntegerTests {
            cases: self.load_tests(self.test_file_path("integer.json"))?,
        })
    }

    pub fn exact_string(&self) -> crate::Result<ExactStringTests> {
        Ok(ExactStringTests {
            cases: self.load_tests(self.test_file_path("exact_string.json"))?,
        })
    }

    // pub fn extract_entities(&self) -> crate::Result<Vec<ExtractEntitiesTest>> {
    //     self.load_tests(self.test_file_path("extract_entities.json"))
    // }

    pub fn extract_urls(&self) -> crate::Result<Vec<ExtractUrlsTest>> {
        self.load_tests(self.test_file_path("extract_urls.json"))
    }

    fn load_tests<T>(&self, file_path: PathBuf) -> crate::Result<Vec<T>>
    where
        T: for<'de> Deserialize<'de> + TestItem,
    {
        let mut file = File::open(file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let tests: Vec<T> = serde_json::from_str(&contents)?;

        Ok(tests
            .into_iter()
            .filter(|test| {
                self.should_include_test(test.test_level())
                    && (self.result_can_be_none || !test.result_can_be_none())
            })
            .collect())
    }

    fn test_file_path(&self, file_name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_sets")
            .join(file_name)
    }

    fn should_include_test(&self, test_level: u8) -> bool {
        match self.test_level {
            TestLevel::IntegrationTest => test_level == 0,
            TestLevel::One => test_level == 1,
            TestLevel::Two => test_level == 2,
            TestLevel::Three => test_level == 3,
            TestLevel::All => true,
            TestLevel::Custom(level) => test_level == level as u8,
        }
    }
}
