use super::*;

/// Builds chunks in linear fashion.
/// Linear meanin that it keeps adding splits until the chunk is within the min and max length. If adding a split makes the chunk too long, it removes the split, and splits that split into multiple smaller splits. It then adds those smaller splits to the list of splits to be added to the chunk.
pub struct LinearChunker {
    /// A pool of available splits that can be added to the chunk.
    unused_splits: VecDeque<TextSplit>,
    chunks: Vec<Chunk>,
    config: Arc<ChunkerConfig>,
    /// The minimum length of the chunk. This is not the same as absolute_length_min.
    length_min: f32,
    /// The maximum length of the chunk. This is not necessarily the same as absolute_length_max.
    length_max: f32,
    /// Remaining number of chunks to be created.
    chunk_count: usize,
    /// Remaining number of tokens from the unused_splits.
    remaining_token_count: f32,
}

impl LinearChunker {
    /// This is called from [`TextChunker`], but you can call it by creating a [`ChunkerConfig`] and passing it in.
    pub fn run(config: &Arc<ChunkerConfig>) -> Option<Vec<Chunk>> {
        let mut chunker = Self {
            unused_splits: config.initial_splits.clone(),
            config: Arc::clone(config),
            chunks: Vec::new(),
            length_min: 0.0,
            length_max: 0.0,
            chunk_count: 0,
            remaining_token_count: 0.0,
        };
        // let mut chunk_times = Vec::new();
        chunker.update_estimates();
        while chunker.chunk_count > 0 {
            if chunker.config.chunks_found.load(Ordering::Relaxed) {
                return None;
            }
            // let start = std::time::Instant::now();
            let mut chunk = Chunk::new(&chunker.config);
            // Ran once with estimated 'synthetic' token counts.
            chunk = chunker.chunk_builder(chunk, true)?;
            // Ran again with actual token counts.
            chunk = chunker.chunk_builder(chunk, false)?;
            chunker.chunks.push(chunk);
            chunker.update_estimates();
            // chunk_times.push(start.elapsed());
        }
        // let avg_chunk_time =
        //     chunk_times.iter().sum::<std::time::Duration>() / chunk_times.len() as u32;
        // println!("Average chunk time: {:?}", avg_chunk_time);
        Some(chunker.chunks)
    }

    /// Ran for each [`Chunk`].
    /// Updates the length_min and length_max.
    fn update_estimates(&mut self) -> usize {
        if self.unused_splits.is_empty() {
            self.chunk_count = 0;
            return self.chunk_count;
        }
        self.remaining_token_count = self.config.estimate_splits_token_count(&self.unused_splits);

        if self.chunks.is_empty() {
            self.chunk_count =
                (self.remaining_token_count / self.config.length_max).ceil() as usize;
        } else {
            self.chunk_count -= 1;
        };

        // Estimates of tokens lost when splits are combined. Trial and error for the ratio.
        let modifier = self.config.length_max * 0.0005 * self.chunk_count as f32;
        let actual_token_count = self.remaining_token_count - modifier;
        if self.chunk_count == 1 {
            if self.remaining_token_count > self.config.length_max {
                eprintln!(
                    "Chunk count is 1 but remaining_token_count is greater than length_max: {:#?}",
                    self.remaining_token_count
                );
            }

            self.length_min = self.remaining_token_count;
            self.length_max = self.config.length_max;
        } else {
            self.length_min = actual_token_count / self.chunk_count as f32;
            self.length_max = self.remaining_token_count / self.chunk_count as f32;
        }
        if self.length_min < 1.0 {
            self.length_min = 1.0;
        }
        if self.length_min > self.config.length_max {
            self.length_min = self.config.length_max - 1.0;
        }
        if self.length_max > self.config.length_max {
            self.length_max = self.config.length_max;
        }

        self.chunk_count
    }

    /// Builds a chunk.
    /// 'estimated' calls the function with estimated token counts. This is much more effecient than calling it with actual token counts, and is used first to get the chunk close enough to the min and max length.
    /// With 'estimated' false, the function is called with actual token counts. This ensures that the chunk is within the min and max length.
    /// By calling the function first with estimated token counts, we avoid calling the tokenizer as much as possible.
    fn chunk_builder(&mut self, mut chunk: Chunk, estimated: bool) -> Option<Chunk> {
        let length_max = if estimated {
            // Add a small buffer to the length_max when using the estimated token counts.
            let length_max = self.length_max + (self.length_max * 0.001).ceil();
            if length_max > self.config.length_max {
                self.config.length_max
            } else {
                length_max
            }
        } else {
            self.config.length_max
        };
        let length_min = if estimated {
            // Subtract a small buffer from the length_min when using the estimated token counts.
            (self.length_min - (self.length_min * 0.001)).floor()
        } else if self.chunk_count == 1 {
            // A safeguard to ensure that we can succesfully build the chunk.
            // In practice this doesn't often happen, but it's possible that the chunk is too small to be built.
            self.config.length_max * 0.5
        } else {
            self.length_min
        };

        if self.chunk_count == 1 {
            while let Some(split) = self.unused_splits.pop_front() {
                chunk.add_split(split, false);
            }
            if estimated {
                return Some(chunk);
            }
            if chunk.token_count(estimated) >= length_min
                && chunk.token_count(estimated) <= length_max
            {
                return Some(chunk);
            } else {
                eprintln!(
                    "\nseparator: {:?}\nChunk count is 1 but final chunk token count is not within min and max.",
                    self.config.initial_separator
                );
                return None;
            }
        }
        loop {
            if self.config.chunks_found.load(Ordering::Relaxed) {
                return None;
            }
            while chunk.token_count(estimated) < length_min && !self.unused_splits.is_empty() {
                let split = self.unused_splits.pop_front().unwrap();
                chunk.add_split(split, false);
                continue;
            }
            while chunk.token_count(estimated) > length_max {
                let split = chunk.remove_split(false);
                self.unused_splits.push_front(split);
                continue;
            }
            if chunk.token_count(estimated) >= length_min
                && chunk.token_count(estimated) <= length_max
            {
                return Some(chunk);
            }
            if self.unused_splits.is_empty() {
                eprintln!(
                    "\nseparator: {:?}\nunused_splits is empty, but chunk token count is not within min and max.",
                    self.config.initial_separator
                );
                return None;
            }
            // Split the removed split, and push the new splits back into the pool of unused_splits.
            let split = self.unused_splits.pop_front().unwrap();
            self.config
                .split_split(split)?
                .into_iter()
                .rev()
                .for_each(|split| {
                    self.unused_splits.push_front(split);
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_text::*;
    use llm_models::local_model::gguf::preset::LlmPreset;

    fn runner(
        tokenizer: &Arc<LlmTokenizer>,
        separator: Separator,
        incoming_text: &str,
        absolute_length_max: u32,
    ) -> Option<Vec<Chunk>> {
        let chunks_found: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let config = Arc::new(ChunkerConfig::new(
            &chunks_found,
            separator.clone(),
            incoming_text,
            absolute_length_max,
            None,
            Some(0.0),
            Arc::clone(tokenizer),
        )?);

        LinearChunker::run(&config)
    }

    #[test]
    fn all() {
        let test_cases = [
            "One one one one.",
            "Two two two two.",
            "Three three three three.",
        ];
        let incoming_text =
            "\n\nOne one one one.\n\nTwo two two two.\n\n\nThree three three three.\n\n";
        let absolute_length_max = 5;

        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());

        for separator in Separator::get_all() {
            let mut chunks =
                runner(&tokenizer, separator, incoming_text, absolute_length_max).unwrap();
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
        for separator in Separator::get_all() {
            let mut chunks =
                runner(&tokenizer, separator, incoming_text, absolute_length_max).unwrap();
            let chunks_string: Vec<String> = chunks.iter_mut().map(|chunk| chunk.text()).collect();
            for (i, chunk) in chunks_string.into_iter().enumerate() {
                assert_eq!(chunk, test_cases[i]);
            }
        }
    }

    #[test]
    fn some() {
        let separators = [
            Separator::TwoPlusEoL,
            Separator::SingleEol,
            // Separator::SentencesRuleBased, // Because this can fail, we can't test it here.
            Separator::SentencesUnicode,
        ];

        let incoming_text = &TEXT.long.content;
        let absolute_length_max = 1024;

        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());

        for separator in separators.clone() {
            let _ = runner(&tokenizer, separator, incoming_text, absolute_length_max).unwrap();
        }
        let tokenizer = LlmPreset::Llama3_1_8bInstruct
            .load()
            .unwrap()
            .model_base
            .tokenizer;
        for separator in separators {
            let _ = runner(&tokenizer, separator, incoming_text, absolute_length_max).unwrap();
        }
    }

    #[test]
    fn switch() {
        let incoming_text = &TEXT.long.content;
        let absolute_length_max = 1024;
        let mut res = TextChunker::new()
            .unwrap()
            .max_chunk_token_size(absolute_length_max)
            .use_dfs_semantic_splitter(false)
            .run_return_result(incoming_text)
            .unwrap();
        assert!(res.token_counts().iter().all(|&x| x <= 1024));
    }
}
