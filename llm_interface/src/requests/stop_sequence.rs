#[derive(Debug, Clone)]
pub enum StoppingSequence {
    InferenceDone(String),
    NoResult(String),
}

impl PartialEq for StoppingSequence {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl StoppingSequence {
    pub fn as_str(&self) -> &str {
        match self {
            StoppingSequence::InferenceDone(s) => s,
            StoppingSequence::NoResult(s) => s,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct StopSequences {
    pub sequences: Vec<StoppingSequence>,
    pub required: bool,
}

impl StopSequences {
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            required: false,
        }
    }

    pub fn to_vec(&self) -> Vec<String> {
        self.sequences
            .iter()
            .map(|sw| sw.as_str().to_owned())
            .collect()
    }

    pub fn parse_string_response<T: AsRef<str>>(
        &self,
        response_stop_word: T,
    ) -> Option<StoppingSequence> {
        for stop_word in &self.sequences {
            if response_stop_word.as_ref() == stop_word.as_str() {
                return Some(stop_word.clone());
            }
        }
        None
    }

    pub fn parse_option_response<T: AsRef<str>>(
        &self,
        response_stop_word: &Option<T>,
    ) -> Option<StoppingSequence> {
        match response_stop_word {
            Some(stop_word) => self.parse_string_response(stop_word),
            None => None,
        }
    }

    pub fn error_on_required(&self) -> String {
        format!(
            "One of the sequences: {} is required, but response stopping_word is None.",
            self.sequences
                .iter()
                .map(|sw| sw.as_str())
                .collect::<Vec<&str>>()
                .join(", ")
        )
    }

    pub fn add_stop_word<T: AsRef<str>>(&self, stop_word: T) -> bool {
        self.sequences.is_empty()
            || !self
                .sequences
                .iter()
                .any(|s| s.as_str() == stop_word.as_ref())
    }

    pub fn set_stop_word_done<T: AsRef<str>>(&mut self, stop_word: T) -> &mut Self {
        if self.add_stop_word(&stop_word) {
            self.sequences.push(StoppingSequence::InferenceDone(
                stop_word.as_ref().to_owned(),
            ));
        }
        self
    }

    pub fn set_stop_word_no_result<T: AsRef<str>>(&mut self, stop_word: T) -> &mut Self {
        if self.add_stop_word(&stop_word) {
            self.sequences
                .push(StoppingSequence::NoResult(stop_word.as_ref().to_owned()));
        }
        self
    }
}
