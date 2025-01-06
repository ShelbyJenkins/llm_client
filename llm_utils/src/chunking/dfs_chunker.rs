use super::*;
use std::collections::HashMap;

/// A text chunking implementation that uses a depth-first search algorithm to build chunks from splits of a single [`Separator`].
/// Fast and fully respects the splits it's given. Ideal for when you want to preserve any semantic meaning how text is separated with whitespaces.
pub struct DfsTextChunker {
    splits: VecDeque<TextSplit>,
    config: Arc<ChunkerConfig>,
    remaining_token_count: f32,
    /// Memoization of valid split indices for a given start index. Avoids repeating searches.
    valid_split_indices_memo: HashMap<usize, Vec<usize>>,
}

impl DfsTextChunker {
    /// This is called from [`TextChunker`], but you can call it by creating a [`ChunkerConfig`] and passing it in.
    pub fn run(config: &Arc<ChunkerConfig>) -> Option<Vec<Chunk>> {
        let splits = config.initial_splits.clone();

        if splits
            .iter()
            .any(|split: &TextSplit| split.token_count.unwrap() as f32 > config.length_max)
        {
            eprintln!(
                "\nPure semantic chunking is impossible for separator: {:#?}.\nA splits token count is more than length_max: {:#?}.", config.initial_separator, config.length_max,
            );
            return None;
        };
        let mut chunker = DfsTextChunker {
            splits,
            config: Arc::clone(config),
            remaining_token_count: 0.0,
            valid_split_indices_memo: HashMap::new(),
        };

        chunker.remaining_token_count = chunker.config.estimate_splits_token_count(&chunker.splits);

        let chunk_split_indexes = chunker.find_valid_chunk_combinations()?;
        chunker.create_chunks(chunk_split_indexes)
    }

    /// Runs the recursive chunk combo finding process.
    fn find_valid_chunk_combinations(&mut self) -> Option<Vec<usize>> {
        let chunk_split_indexes = self.recursive_chunk_tester(0);
        if chunk_split_indexes.is_none() || chunk_split_indexes.as_ref().unwrap().len() == 1 {
            None
        } else {
            chunk_split_indexes
        }
    }

    /// When given a starting split, it finds all valid combinations of splits that can be made into chunks.
    /// It then recursively calls itself with each of those splits as the new starting split.
    /// It does this until it finds a path that reaches the final split.
    fn recursive_chunk_tester(&mut self, start: usize) -> Option<Vec<usize>> {
        if self.config.chunks_found.load(Ordering::Relaxed) {
            return None;
        }
        if self.valid_split_indices_memo.contains_key(&{ start }) {
            return None; // We've already seen this path.
        }
        let valid_split_indices = self.find_valid_split_indices_for_chunk(start)?;

        for &end_split in &valid_split_indices {
            // Successful exit condition
            if end_split + 1 == self.splits.len() {
                return Some(vec![end_split]);
            }
        }

        for &end_split in &valid_split_indices {
            // Recursive call with the next start
            let result = self.recursive_chunk_tester(end_split + 1);
            if let Some(mut result) = result {
                result.insert(0, end_split);
                return Some(result);
            }
        }

        None
    }

    /// Finds all valid splits that can be made into a chunk starting from a given split index.
    fn find_valid_split_indices_for_chunk(&mut self, start: usize) -> Option<Vec<usize>> {
        let mut valid_split_indices = Vec::new();
        let mut chunk = Chunk::new(&self.config);

        for (index, split) in self.splits.iter().enumerate().skip(start) {
            chunk.add_split(split.clone(), false);

            if chunk.estimated_token_count >= self.config.absolute_length_min as f32 {
                if chunk.estimated_token_count > self.config.length_max {
                    break;
                }
                valid_split_indices.push(index);
            }
        }

        if valid_split_indices.is_empty() {
            self.valid_split_indices_memo
                .insert(start, valid_split_indices);
            None
        } else {
            self.valid_split_indices_memo
                .insert(start, valid_split_indices.clone());
            Some(valid_split_indices)
        }
    }

    /// Creates chunks from a list of split indices.
    fn create_chunks(&self, chunk_split_indexes: Vec<usize>) -> Option<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
        chunk_ranges.push((0, chunk_split_indexes[0]));
        for (i, &split_index) in chunk_split_indexes.iter().enumerate() {
            if i + 1 == chunk_split_indexes.len() {
                break;
            }
            chunk_ranges.push((split_index + 1, chunk_split_indexes[i + 1]));
        }
        for (start_index, end_index) in chunk_ranges {
            let mut chunk = Chunk::new(&self.config);
            for i in start_index..=end_index {
                chunk.add_split(self.splits[i].clone(), false);
            }

            chunks.push(chunk);
        }
        Some(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_models::local_model::gguf::preset::LlmPreset;

    fn runner(
        tokenizer: &std::sync::Arc<LlmTokenizer>,
        separator: Separator,
    ) -> Option<Vec<Chunk>> {
        let chunks_found: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let incoming_text =
            "\n\nOne one one one.\n\nTwo two two two.\n\n\nThree three three three.\n\n";
        let absolute_length_max = 5;
        let config = Arc::new(ChunkerConfig::new(
            &chunks_found,
            separator.clone(),
            incoming_text,
            absolute_length_max,
            None,
            Some(0.0),
            Arc::clone(tokenizer),
        )?);

        DfsTextChunker::run(&config)
    }

    #[test]
    fn all() {
        let test_cases = [
            "One one one one.",
            "Two two two two.",
            "Three three three three.",
        ];
        let separators = vec![
            Separator::TwoPlusEoL,
            Separator::SingleEol,
            Separator::SentencesRuleBased,
            Separator::SentencesUnicode,
        ];

        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());

        for separator in separators.clone() {
            let mut chunks = runner(&tokenizer, separator).unwrap();
            let chunks_string: Vec<String> = chunks.iter_mut().map(|chunk| chunk.text()).collect();
            for (i, chunk) in chunks_string.into_iter().enumerate() {
                assert_eq!(chunk, test_cases[i]);
            }
        }
        let tokenizer = LlmPreset::Llama3_1_8bInstruct
            .load()
            .unwrap()
            .model_base
            .tokenizer;
        for separator in separators {
            let mut chunks = runner(&tokenizer, separator).unwrap();
            let chunks_string: Vec<String> = chunks.iter_mut().map(|chunk| chunk.text()).collect();
            for (i, chunk) in chunks_string.into_iter().enumerate() {
                assert_eq!(chunk, test_cases[i]);
            }
        }
    }
}
