mod dfs_chunker;
#[cfg(test)]
mod external_text_chunker;
mod linear_chunker;
mod overlap;

use crate::splitting::{Separator, SeparatorGroup, TextSplit, TextSplitter};

use anyhow::Result;
use dfs_chunker::DfsTextChunker;
use linear_chunker::LinearChunker;
use llm_models::tokenizer::LlmTokenizer;
use overlap::OverlapChunker;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// An easy alternative to the [`TextChunker`] struct.  
///
/// * `text` - The natural language text to chunk.
/// * `max_chunk_token_size` - The maxium token sized to be chunked to. Inclusive.
/// * `overlap_percent` - The percentage of overlap between chunks. Default is None.
pub fn chunk_text(
    text: &str,
    max_chunk_token_size: u32,
    overlap_percent: Option<f32>,
) -> Result<Option<Vec<String>>> {
    let mut splitter = TextChunker::new()?.max_chunk_token_size(max_chunk_token_size);
    if let Some(overlap_percent) = overlap_percent {
        splitter = splitter.overlap_percent(overlap_percent);
    }
    Ok(splitter.run(text))
}

const ABSOLUTE_LENGTH_MAX_DEFAULT: u32 = 1024;
const ABSOLUTE_LENGTH_MIN_DEFAULT_RATIO: f32 = 0.75;
const TOKENIZER_TIKTOKEN_DEFAULT: &str = "gpt-4";

/// Splits text by paragraphs, newlines, sentences, spaces, and finally graphemes, and builds chunks from the splits that are within the desired token ranges.
pub struct TextChunker {
    /// An atomic reference to the tokenizer. Defaults to the TikToken tokenizer.
    tokenizer: Arc<LlmTokenizer>,
    /// Inclusive hard limit.
    absolute_length_max: u32,
    /// This is used solely for the [`DfsTextChunker`] to determine the minimum chunk size. Default is 75% of the `absolute_length_max`.
    absolute_length_min: Option<u32>,
    /// The percentage of overlap between chunks. Default is None.
    overlap_percent: Option<f32>,
    /// Whether to use the DFS semantic splitter to attempt to build valid chunks. Default is true.
    use_dfs_semantic_splitter: bool,
}

impl TextChunker {
    /// Creates a new instance of the [`TextChunker`] struct using the default TikToken tokenizer.
    pub fn new() -> Result<Self> {
        Ok(Self {
            tokenizer: Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT)?),
            absolute_length_max: ABSOLUTE_LENGTH_MAX_DEFAULT,
            absolute_length_min: None,
            overlap_percent: None,
            use_dfs_semantic_splitter: true,
        })
    }
    /// Creates a new instance of the [`TextChunker`] struct using a custom tokenizer. For example a Hugging Face tokenizer.
    pub fn new_with_tokenizer(custom_tokenizer: &Arc<LlmTokenizer>) -> Self {
        Self {
            tokenizer: Arc::clone(custom_tokenizer),
            absolute_length_max: ABSOLUTE_LENGTH_MAX_DEFAULT,
            absolute_length_min: None,
            overlap_percent: None,
            use_dfs_semantic_splitter: true,
        }
    }

    /// Sets the maximum token size for the chunks. Default is 1024.
    ///
    /// * `max_chunk_token_size` - The maxium token sized to be chunked to. Inclusive.
    pub fn max_chunk_token_size(mut self, max_chunk_token_size: u32) -> Self {
        self.absolute_length_max = max_chunk_token_size;
        self
    }

    /// Sets the minimum token size for the chunks. Default is 75% of the `absolute_length_max`. Used solely for the [`DfsTextChunker`] to determine the minimum chunk size.
    ///
    /// * `min_chunk_token_size` - The minimum token sized to be chunked to..
    pub fn min_chunk_token_size(mut self, min_chunk_token_size: u32) -> Self {
        self.absolute_length_min = Some(min_chunk_token_size);
        self
    }

    /// The [`DfsTextChunker`] is faster is completely respective of semantic separators. However, it produces less balanced chunk sizes and will fail if the text cannot be split.
    /// By default the [`TextChunker`] attempts to chunk with the [`DfsTextChunker`] first, and if that fails, it will use the [`LinearChunker`].
    ///
    /// * `use_dfs_semantic_splitter` - Whether to use the DFS semantic splitter to attempt to build valid chunks. Default is true.
    pub fn use_dfs_semantic_splitter(mut self, use_dfs_semantic_splitter: bool) -> Self {
        self.use_dfs_semantic_splitter = use_dfs_semantic_splitter;
        self
    }

    /// Sets the percentage of overlap between chunks. Default is None.
    /// The full percentage is used foward for the first chunk, and backwards for the last chunk.
    /// Middle chunks evenly split the percentage between forward and backwards.
    ///
    /// * `overlap_percent` - The percentage of overlap between chunks. Minimum is 0.01, and maximum is 0.5. Default is None.
    pub fn overlap_percent(mut self, overlap_percent: f32) -> Self {
        self.overlap_percent = if !(0.01..=0.5).contains(&overlap_percent) {
            Some(0.10)
        } else {
            Some(overlap_percent)
        };
        self
    }

    /// Runs the [`TextChunker`] on the incoming text and returns the chunks as a vector of strings.
    ///
    /// * `incoming_text` - The natural language text to chunk.
    pub fn run(&self, incoming_text: &str) -> Option<Vec<String>> {
        Some(self.text_chunker(incoming_text)?.chunks_to_text())
    }

    /// Runs the [`TextChunker`] on the incoming text and returns the chunks as a [`ChunkerResult`].
    /// The [`ChunkerResult`] contains the incoming text, the initial separator used, the chunks, the tokenizer, and the chunking duration. Useful for testing, benching, and diagnostics.
    ///
    /// * `incoming_text` - The natural language text to chunk.
    pub fn run_return_result(&self, incoming_text: &str) -> Option<ChunkerResult> {
        self.text_chunker(incoming_text)
    }

    /// Backend runner for [`TextChunker`].
    /// Attempts to chunk the incoming text on all [`Separator`] first using the [`DfsTextChunker`] and then [`LinearChunker`].
    /// Returns whichever [`Separator`] chunking attempt was successful first, and if none are successful, returns None.
    /// If the incoming text is less than the `absolute_length_max`, it will return a single chunk.
    fn text_chunker(&self, incoming_text: &str) -> Option<ChunkerResult> {
        let chunking_start_time = std::time::Instant::now();
        // A flag to signal if chunks have been found, and for all other threads to stop searching.
        let chunks_found: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        // Parallize the search for the first successful chunking attempt.
        Separator::get_all().par_iter().find_map_any(|separator| {
            if chunks_found.load(Ordering::Relaxed) {
                return None;
            }
            let config = Arc::new(ChunkerConfig::new(
                &chunks_found,
                separator.clone(),
                incoming_text,
                self.absolute_length_max,
                self.absolute_length_min,
                self.overlap_percent,
                self.tokenizer(),
            )?);
            if chunks_found.load(Ordering::Relaxed) {
                return None;
            }
            // If the text is less than the absolute_length_max, `initial_separator` will be set to Separator::None, and we return a single chunk.
            if config.initial_separator == Separator::None {
                chunks_found.store(true, Ordering::Relaxed);
                return Some(ChunkerResult::new(incoming_text, &config, chunking_start_time, vec![Chunk::dummy_chunk(&config, incoming_text)]));
            };
            // println!("Found config for separator: {:#?}", separator);
            if config.initial_separator.group() == SeparatorGroup::Semantic && self.use_dfs_semantic_splitter {
                let chunks: Option<Vec<Chunk>> = DfsTextChunker::run(&config);
                if let Some(chunks) = chunks {
                    let chunks = OverlapChunker::run(&config, chunks);
                    match chunks {
                        Ok(chunk) => {
                        chunks_found.store(true, Ordering::Relaxed);
                        println!(
                            "\nSuccessfully Split with: DfsTextChunker on separator: {:#?}\ntotal chunking_duration: {:#?}.\n",
                            separator,
                            chunking_start_time.elapsed()
                        );
                        return Some(ChunkerResult::new(incoming_text, &config, chunking_start_time, chunk));
                        },
                        Err(e) => {
                            eprintln!("Error: {:#?}", e);
                        }
                    }
                }
            }
            let chunks = LinearChunker::run(&config)?;
            let chunks = OverlapChunker::run(&config, chunks);
            match chunks {
                Ok(chunks) => {
                    chunks_found.store(true, Ordering::Relaxed);
                    println!(
                        "\nSuccessfully Split with: LinearChunker on separator: {:#?}\ntotal chunking_duration: {:#?}.\n",
                        separator,
                        chunking_start_time.elapsed()
                    );
                    Some(ChunkerResult::new(incoming_text, &config, chunking_start_time, chunks))
                }
                Err(e) => {
                    eprintln!("Error: {:#?}", e);
                    None
                }
            }
        })
    }

    fn tokenizer(&self) -> Arc<LlmTokenizer> {
        Arc::clone(&self.tokenizer)
    }
}

/// Configuration used by the [`TextChunker`], [`DfsTextChunker`], [`LinearChunker`], and [`OverlapChunker`] to build chunks.
/// Instantiated by the [`TextChunker`] on each [`Separator`] and passed to the chunkers.
pub struct ChunkerConfig {
    chunks_found: Arc<AtomicBool>,
    absolute_length_max: u32,
    absolute_length_min: u32,
    length_max: f32,
    overlap_percent: Option<f32>,
    tokenizer: Arc<LlmTokenizer>,
    base_text: Arc<str>,
    initial_separator: Separator,
    initial_splits: VecDeque<TextSplit>,
}

impl ChunkerConfig {
    fn new(
        chunks_found: &Arc<AtomicBool>,
        separator: Separator,
        incoming_text: &str,
        absolute_length_max: u32,
        absolute_length_min: Option<u32>,
        overlap_percent: Option<f32>,
        tokenizer: Arc<LlmTokenizer>,
    ) -> Option<Self> {
        let length_max = if let Some(overlap_percent) = overlap_percent {
            (absolute_length_max as f32 - (absolute_length_max as f32 * overlap_percent)).floor()
        } else {
            absolute_length_max as f32
        };
        let absolute_length_min = if let Some(absolute_length_min) = absolute_length_min {
            absolute_length_min
        } else {
            (absolute_length_max as f32 * ABSOLUTE_LENGTH_MIN_DEFAULT_RATIO) as u32
        };
        if absolute_length_max <= absolute_length_min {
            panic!(
                "\nA combination absolute_length_max: {:#?} and overlap_percent: {:#?} is less than or equal to absolute_length_min: {:#?}.",
                absolute_length_max, overlap_percent, absolute_length_min
            );
        }

        let mut config = Self {
            chunks_found: Arc::clone(chunks_found),
            absolute_length_max,
            absolute_length_min,
            length_max,
            overlap_percent,
            tokenizer,
            base_text: Arc::from(separator.clean_text(incoming_text)),
            initial_separator: separator.clone(),
            initial_splits: VecDeque::new(),
        };

        let cleaned_text_token_count = config.tokenizer.count_tokens(&config.base_text);
        if cleaned_text_token_count <= absolute_length_max {
            config.initial_separator = Separator::None;
            return Some(config);
        }
        let splits = if let Some(mut splits) = TextSplitter::new()
            .recursive(false)
            .clean_text(false)
            .on_separator(&separator)
            .split_text(&config.base_text)
        {
            splits.iter_mut().for_each(|split| {
                config.set_split_token_count(split);
            });
            splits
        } else {
            // eprintln!("No splits found for separator: {:#?}", separator);
            return None;
        };
        let splits_token_count = config.estimate_splits_token_count(&splits);
        let chunk_count = (splits_token_count / config.length_max).ceil() as usize;
        if splits.len() < chunk_count {
            eprintln!(
                "\nChunking is impossible for separator: {:#?}. Splits count: {:#?} is less than the minimum chunk_count: {:#?}.",
                separator,
                splits.len(),
                chunk_count,
            );
            return None;
        };

        config.initial_splits = splits;
        Some(config)
    }

    /// Splits an existing [`TextSplit`] into multiple [`TextSplit`]s on the next [`Separator`].
    /// If no splits are found, at attempts split on the following [`Separator`].
    /// If it reaches the final [`Separator`] without successfully splitting, it returns None.
    fn split_split(&self, split: TextSplit) -> Option<VecDeque<TextSplit>> {
        let mut new_splits: VecDeque<TextSplit> = match split.split() {
            Some(splits) => splits,
            None => {
                // eprintln!(
                //     "No splits found for split: {:#?}",
                //     split.base_text.chars().take(50).collect::<String>()
                // );
                return None;
            }
        };
        new_splits.iter_mut().for_each(|split| {
            self.set_split_token_count(split);
        });
        Some(new_splits)
    }

    fn set_split_token_count(&self, split: &mut TextSplit) {
        if split.token_count.is_none() {
            let token_count = self.tokenizer.count_tokens(split.text());
            split.token_count = Some(token_count);
        }
    }

    /// Estimates the token count of the splits.
    /// This is used for estimating the remaining token count, and is also used to estimate the token count of chunks.
    /// It is somewhat accurate.
    fn estimate_splits_token_count(&self, splits: &VecDeque<TextSplit>) -> f32 {
        let mut last_separator = Separator::None;
        let mut total_tokens = 0.0;
        for split in splits {
            let split_tokens = match split.split_separator {
                Separator::GraphemesUnicode => match last_separator {
                    Separator::None | Separator::GraphemesUnicode => 0.55,
                    _ => 1.0,
                },
                _ => split.token_count.unwrap() as f32,
            };
            if last_separator != Separator::None {
                let white_space_ratio = match split.split_separator {
                    Separator::None => {
                        unreachable!()
                    }
                    Separator::TwoPlusEoL => 0.999,
                    Separator::SingleEol => 0.999,
                    Separator::SentencesRuleBased => 0.998,
                    Separator::SentencesUnicode => 0.998,
                    Separator::WordsUnicode => 0.89,
                    Separator::GraphemesUnicode => 1.0,
                };
                total_tokens += split_tokens * white_space_ratio;
            } else {
                total_tokens += split_tokens;
            }
            last_separator = split.split_separator.clone();
        }
        total_tokens
    }
}

#[derive(Clone)]
pub struct Chunk {
    text: Option<String>,
    used_splits: VecDeque<TextSplit>,
    token_count: Option<usize>,
    estimated_token_count: f32,
    config: Arc<ChunkerConfig>,
}

impl Chunk {
    fn new(config: &Arc<ChunkerConfig>) -> Self {
        Chunk {
            text: None,
            used_splits: VecDeque::new(),
            token_count: Some(0),
            estimated_token_count: 0.0,
            config: Arc::clone(config),
        }
    }

    fn dummy_chunk(config: &Arc<ChunkerConfig>, text: &str) -> Self {
        Chunk {
            text: Some(text.to_string()),
            used_splits: VecDeque::new(),
            token_count: Some(0),
            estimated_token_count: 0.0,
            config: Arc::clone(config),
        }
    }

    fn add_split(&mut self, split: TextSplit, backwards: bool) {
        if backwards {
            self.used_splits.push_front(split);
        } else {
            self.used_splits.push_back(split);
        }
        self.estimated_token_count = self.config.estimate_splits_token_count(&self.used_splits);
        self.token_count = None;
        self.text = None;
    }

    fn remove_split(&mut self, backwards: bool) -> TextSplit {
        let split = if backwards {
            self.used_splits.pop_front().unwrap()
        } else {
            self.used_splits.pop_back().unwrap()
        };
        self.estimated_token_count = self.config.estimate_splits_token_count(&self.used_splits);
        self.token_count = None;
        self.text = None;
        split
    }

    fn token_count(&mut self, estimated: bool) -> f32 {
        if let Some(token_count) = self.token_count {
            token_count as f32
        } else if estimated {
            self.estimated_token_count
        } else {
            let text = &self.text();
            let token_count = self.config.tokenizer.count_tokens(text) as usize;
            self.token_count = Some(token_count);
            self.estimated_token_count = token_count as f32;
            token_count as f32
        }
    }

    fn text(&mut self) -> String {
        if let Some(text) = &self.text {
            text.to_owned()
        } else {
            let text = TextSplitter::splits_to_text(&self.used_splits, false);
            self.text = Some(text.clone());
            text
        }
    }
}

pub struct ChunkerResult {
    incoming_text: Arc<str>,
    initial_separator: Separator,
    chunks: Vec<Chunk>,
    tokenizer: Arc<LlmTokenizer>,
    chunking_duration: std::time::Duration,
}

impl ChunkerResult {
    fn new(
        incoming_text: &str,
        config: &Arc<ChunkerConfig>,
        chunking_start_time: std::time::Instant,
        mut chunks: Vec<Chunk>,
    ) -> ChunkerResult {
        chunks.iter_mut().for_each(|chunk| {
            chunk.text();
        });
        ChunkerResult {
            incoming_text: Arc::from(incoming_text),
            initial_separator: config.initial_separator.clone(),
            chunks,
            tokenizer: Arc::clone(&config.tokenizer),
            chunking_duration: chunking_start_time.elapsed(),
        }
    }

    pub fn chunks_to_text(&mut self) -> Vec<String> {
        self.chunks.iter_mut().map(|chunk| chunk.text()).collect()
    }

    pub fn token_counts(&mut self) -> Vec<u32> {
        let mut token_counts: Vec<u32> = Vec::with_capacity(self.chunks.len());
        for chunk in &self.chunks {
            let chunk_text = if let Some(text) = &chunk.text {
                text.to_owned()
            } else {
                TextSplitter::splits_to_text(&chunk.used_splits, false)
            };
            token_counts.push(self.tokenizer.count_tokens(&chunk_text));
        }
        token_counts
    }
}

impl std::fmt::Debug for ChunkerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut chunk_token_sizes = Vec::with_capacity(self.chunks.len());
        let mut largest_token_size = 0;
        let mut smallest_token_size = u32::MAX;
        let mut all_chunks_token_count = 0;
        let mut chunk_char_sizes = Vec::with_capacity(self.chunks.len());
        let mut largest_char_size = 0;
        let mut smallest_char_size = u32::MAX;
        let mut all_chunks_char_count = 0;

        for chunk in &self.chunks {
            let chunk_text = if let Some(text) = &chunk.text {
                text.to_owned()
            } else {
                panic!("Chunk text not found.")
            };
            let token_count = self.tokenizer.count_tokens(&chunk_text);
            let char_count = u32::try_from(chunk_text.chars().count()).unwrap();
            chunk_token_sizes.push(token_count);
            chunk_char_sizes.push(char_count);
            all_chunks_token_count += token_count;
            all_chunks_char_count += char_count;
            if token_count > largest_token_size {
                largest_token_size = token_count;
            }
            if char_count > largest_char_size {
                largest_char_size = char_count;
            }
            if token_count < smallest_token_size {
                smallest_token_size = token_count;
            }
            if char_count < smallest_char_size {
                smallest_char_size = char_count;
            }
        }
        f.debug_struct("\nChunkerTestResult")
            .field("chunk_count", &self.chunks.len())
            .field("chunk_token_sizes", &chunk_token_sizes)
            .field(
                "avg_token_size",
                &(all_chunks_token_count / u32::try_from(self.chunks.len()).unwrap()),
            )
            .field("largest_token_size", &largest_token_size)
            .field("smallest_token_size", &smallest_token_size)
            .field(
                "incoming_text_token_count",
                &self.tokenizer.count_tokens(&self.incoming_text),
            )
            .field("all_chunks_token_count", &all_chunks_token_count)
            .field("chunk_char_sizes", &chunk_char_sizes)
            .field(
                "avg_char_size",
                &(all_chunks_char_count / u32::try_from(self.chunks.len()).unwrap()),
            )
            .field("largest_char_size", &largest_char_size)
            .field("smallest_char_size", &smallest_char_size)
            .field(
                "incoming_text_char_count",
                &self.incoming_text.chars().count(),
            )
            .field("all_chunks_char_count", &all_chunks_char_count)
            .field("chunking_duration", &self.chunking_duration)
            .field("initial_separator", &self.initial_separator)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_text::*;
    use llm_models::local_model::gguf::preset::LlmPreset;

    fn run_test(case: u32, content: &str, tokenizer: &Arc<LlmTokenizer>) -> Option<ChunkerResult> {
        TextChunker::new_with_tokenizer(tokenizer)
            .max_chunk_token_size(case)
            .run_return_result(content)
        // println!("{:#?}", res.as_ref().unwrap());
    }

    fn check_first(
        case: u32,
        res: &mut ChunkerResult,
        test_cases: &ChunkingTestCases,
    ) -> Result<()> {
        if res.chunks.len() == 1 {
            Ok(())
        } else {
            let test_case = test_cases.case(case);
            if !res
                .chunks
                .first_mut()
                .unwrap()
                .text()
                .contains(test_case.first())
            {
                panic!(
                    "First chunk does not match for case: {:#?}\nresult: {:#?}\ntest_case: {:#?}\n",
                    case,
                    res.chunks.first_mut().unwrap().text(),
                    test_case.first()
                );
            }
            Ok(())
        }
    }

    fn check_last(
        case: u32,
        res: &mut ChunkerResult,
        test_cases: &ChunkingTestCases,
    ) -> Result<()> {
        if res.chunks.len() == 1 {
            return Ok(());
        }
        let test_case = test_cases.case(case);
        if !res
            .chunks
            .last_mut()
            .unwrap()
            .text()
            .contains(test_case.last())
        {
            panic!(
                "Last chunk does not match for case: {:#?}\nresult: {:#?}\ntest_case: {:#?}\n",
                case,
                res.chunks.last_mut().unwrap().text(),
                test_case.last()
            );
        }
        Ok(())
    }

    fn tiktoken() -> Arc<LlmTokenizer> {
        Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap())
    }

    fn hf() -> Arc<LlmTokenizer> {
        LlmPreset::Llama3_1_8bInstruct
            .load()
            .unwrap()
            .model_base
            .tokenizer
    }

    #[test]
    fn tiny() {
        let content = &CHUNK_TESTS.chunking_tiny.content;
        let test_cases = &CHUNK_TESTS.chunking_tiny.test_cases;
        let cases = vec![64, 128, 256, 512];

        for case in cases {
            let mut res: ChunkerResult = run_test(case, content, &tiktoken()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            check_first(case, &mut res, test_cases).unwrap();
            check_last(case, &mut res, test_cases).unwrap();
            let mut res = run_test(case, content, &hf()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            check_first(case, &mut res, test_cases).unwrap();
            check_last(case, &mut res, test_cases).unwrap();
        }
    }

    #[test]
    fn small() {
        let content = &CHUNK_TESTS.chunking_small.content;
        let test_cases = &CHUNK_TESTS.chunking_small.test_cases;
        let cases = vec![64, 128, 256, 512, 768, 1536];

        for case in cases {
            let mut res = run_test(case, content, &tiktoken()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            check_first(case, &mut res, test_cases).unwrap();
            check_last(case, &mut res, test_cases).unwrap();
            let mut res = run_test(case, content, &hf()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            check_first(case, &mut res, test_cases).unwrap();
            check_last(case, &mut res, test_cases).unwrap();
        }
    }

    #[test]
    fn medium() {
        let content = &TEXT.medium.content;
        let cases = vec![256, 512, 768, 1536];

        for case in cases {
            let mut res = run_test(case, content, &tiktoken()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            let mut res = run_test(case, content, &hf()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
        }
    }

    #[test]
    fn long() {
        let content = &TEXT.long.content;
        let cases = vec![512, 1024, 1536, 2048];

        for case in cases {
            let mut res = run_test(case, content, &tiktoken()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            let mut res = run_test(case, content, &hf()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
        }
    }

    #[test]
    fn really_long() {
        let content = &TEXT.really_long.content;
        let cases = vec![512, 1024, 2048, 4096];

        for case in cases {
            let mut res = run_test(case, content, &tiktoken()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
            let mut res = run_test(case, content, &hf()).unwrap();
            assert!(res.token_counts().iter().all(|&x| x <= case));
        }
    }

    #[test]
    fn within_abs_max() {
        let res = TextChunker::new()
            .unwrap()
            .max_chunk_token_size(400)
            .run_return_result(&CHUNK_TESTS.chunking_tiny.content)
            .unwrap();
        assert_eq!(res.chunks.len(), 1);
    }
}
