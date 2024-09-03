use super::*;
use std::{fs::File, io::Read, path::PathBuf};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestLevel {
    #[default]
    Zero,
    One,
    Two,
    Three,
    All,
}

#[derive(Default)]
pub struct TestSetsLoader {
    pub optional: bool,
    pub test_level: TestLevel,
}

#[allow(dead_code)]
impl TestSetsLoader {
    pub fn optional(mut self, optional: bool) -> Self {
        self.optional = optional;
        self
    }

    pub fn test_level_zero(mut self) -> Self {
        self.test_level = TestLevel::Zero;
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

    pub fn boolean(&self) -> Result<BooleanTests> {
        Ok(BooleanTests {
            cases: self.load_tests(self.test_file_path("boolean.json"))?,
        })
    }

    pub fn integer(&self) -> Result<IntegerTests> {
        Ok(IntegerTests {
            cases: self.load_tests(self.test_file_path("integer.json"))?,
        })
    }

    pub fn exact_string(&self) -> Result<ExactStringTests> {
        Ok(ExactStringTests {
            cases: self.load_tests(self.test_file_path("exact_string.json"))?,
        })
    }

    // pub fn extract_entities(&self) -> Result<Vec<ExtractEntitiesTest>> {
    //     self.load_tests(self.test_file_path("extract_entities.json"))
    // }

    pub fn extract_urls(&self) -> Result<Vec<ExtractUrlsTest>> {
        self.load_tests(self.test_file_path("extract_urls.json"))
    }

    fn load_tests<T>(&self, file_path: PathBuf) -> Result<Vec<T>>
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
                    && (self.optional || !test.is_optional())
            })
            .collect())
    }

    fn test_file_path(&self, file_name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("common")
            .join("test_sets")
            .join(file_name)
    }

    fn should_include_test(&self, test_level: u8) -> bool {
        match self.test_level {
            TestLevel::Zero => test_level == 0,
            TestLevel::One => test_level == 1,
            TestLevel::Two => test_level == 2,
            TestLevel::Three => test_level == 3,
            TestLevel::All => true,
        }
    }
}
