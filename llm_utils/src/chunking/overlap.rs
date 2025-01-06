use super::*;
use anyhow::anyhow;

/// Adds overlap to chunks built by the [`TextChunker`].
pub struct OverlapChunker {
    config: Arc<ChunkerConfig>,
    chunks: Vec<Chunk>,
    // A copy of the original chunks to be used as a source of unprocessed splits.
    chunks_copy: Vec<Chunk>,
    overlap_percent: f32,
}

impl OverlapChunker {
    pub fn run(config: &Arc<ChunkerConfig>, chunks: Vec<Chunk>) -> Result<Vec<Chunk>> {
        if chunks.is_empty() {
            panic!("Chunks is empty. This should never happen.");
        }
        // If overlap_percent is None, 0.0, or the overlap would be less than 1 token, return the chunks as is.
        let overlap_percent = if let Some(overlap_percent) = config.overlap_percent {
            overlap_percent
        } else {
            return Ok(chunks);
        };
        if overlap_percent == 0.0
            || (config.absolute_length_max as f32
                - (config.absolute_length_max as f32 * overlap_percent))
                .floor()
                <= 0.0
        {
            return Ok(chunks);
        }

        let mut chunker = Self {
            config: Arc::clone(config),
            chunks_copy: chunks.clone(),
            chunks,
            overlap_percent,
        };

        for i in 0..chunker.chunks.len() {
            if chunker.config.chunks_found.load(Ordering::Relaxed) {
                return Err(anyhow!(
                    "OverlapChunker stopping early due to chunks_found in other thread."
                ));
            }
            let (back_min, back_max, for_min, for_max) = chunker.overlap_lengths(i);
            chunker.forward_overlap(i, for_min, for_max)?;
            chunker.backward_overlap(i, back_min, back_max)?;
        }

        Ok(chunker.chunks)
    }

    fn overlap_lengths(&mut self, chunk_index: usize) -> (f32, f32, f32, f32) {
        let chunk = self.chunks.get_mut(chunk_index).unwrap();

        let token_count = chunk.token_count(false);
        let overlap_length = ((token_count * self.overlap_percent) + token_count).ceil();

        if chunk_index == 0 {
            let overlap_length_min =
                if f32::abs(overlap_length - self.config.absolute_length_max as f32) < 1.0 {
                    overlap_length - 1.0
                } else {
                    overlap_length
                };
            (
                0.0,
                0.0,
                overlap_length_min,
                self.config.absolute_length_max as f32,
            )
        } else if chunk_index == self.chunks.len() - 1 {
            let overlap_length_min =
                if f32::abs(overlap_length - self.config.absolute_length_max as f32) < 1.0 {
                    overlap_length - 1.0
                } else {
                    overlap_length
                };
            (
                overlap_length_min,
                self.config.absolute_length_max as f32,
                0.0,
                0.0,
            )
        } else {
            let overlap_length_min =
                if f32::abs(overlap_length - self.config.absolute_length_max as f32) < 1.0 {
                    overlap_length - 2.0
                } else {
                    overlap_length
                };
            let headroom_min = self.config.absolute_length_max as f32 - overlap_length_min;
            let headroom_max = self.config.absolute_length_max as f32 - token_count;
            (
                (token_count + headroom_min).floor(),
                (token_count + headroom_max).ceil(),
                (token_count + (headroom_min / 2.0)).floor(),
                (token_count + headroom_max / 2.0).ceil(),
            )
        }
    }

    fn forward_overlap(
        &mut self,
        chunk_index: usize,
        length_min: f32,
        length_max: f32,
    ) -> Result<()> {
        if chunk_index == self.chunks.len() - 1 {
            return Ok(());
        };
        let mut splits = self.chunks_copy[chunk_index + 1].used_splits.clone();
        self.chunk_builder(
            chunk_index,
            &mut splits,
            length_min,
            length_max,
            true,
            false,
        )?;
        self.chunk_builder(
            chunk_index,
            &mut splits,
            length_min,
            length_max,
            false,
            false,
        )?;
        Ok(())
    }

    fn backward_overlap(
        &mut self,
        chunk_index: usize,
        length_min: f32,
        length_max: f32,
    ) -> Result<()> {
        if chunk_index == 0 {
            return Ok(());
        };

        let mut splits = self.chunks_copy[chunk_index - 1].used_splits.clone();
        self.chunk_builder(chunk_index, &mut splits, length_min, length_max, true, true)?;
        self.chunk_builder(
            chunk_index,
            &mut splits,
            length_min,
            length_max,
            false,
            true,
        )?;
        Ok(())
    }

    fn chunk_builder(
        &mut self,
        chunk_index: usize,
        splits: &mut VecDeque<TextSplit>,
        length_min: f32,
        length_max: f32,
        estimated: bool,
        backwards: bool,
    ) -> Result<()> {
        let chunk = &mut self.chunks[chunk_index];

        loop {
            if self.config.chunks_found.load(Ordering::Relaxed) {
                return Err(anyhow!(
                    "OverlapChunker stopping early due to chunks_found in other thread."
                ));
            }
            while chunk.token_count(estimated).ceil() <= length_min && !splits.is_empty() {
                let split = if backwards {
                    splits.pop_back().unwrap()
                } else {
                    splits.pop_front().unwrap()
                };
                chunk.add_split(split, backwards);
                continue;
            }
            while chunk.token_count(estimated).ceil() > length_max {
                let split = chunk.remove_split(backwards);
                if backwards {
                    splits.push_back(split);
                } else {
                    splits.push_front(split);
                }
                continue;
            }
            if chunk.token_count(estimated).ceil() >= length_min
                && chunk.token_count(estimated).ceil() <= length_max
            {
                return Ok(());
            }
            if splits.is_empty() {
                return Err(anyhow!("overlap error: splits is empty, but chunk token count is not within min and max."));
            }
            let split = if backwards {
                splits.pop_back().unwrap()
            } else {
                splits.pop_front().unwrap()
            };
            self.config
                .split_split(split)
                .ok_or_else(|| anyhow!("split_split returned None for while generating overlap."))?
                .into_iter()
                .rev()
                .for_each(|split| {
                    splits.push_front(split);
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_text;

    use super::*;
    use llm_models::local_model::gguf::preset::LlmPreset;

    #[test]
    fn test_overlap() {
        let test_cases = [
            "one one one one one one one one one one two two",
            "one two two two two two two two two two two three",
            "two two three three three three three three three three three three",
        ];
        let incoming_text =
            "\n\none one one one one one one one one one\n\ntwo two two two two two two two two two\n\n\nthree three three three three three three three three three\n\n";
        let absolute_length_max = 12;

        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());
        let mut res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .overlap_percent(0.166666)
            .run_return_result(incoming_text)
            .unwrap();
        let chunks_string: Vec<String> = res.chunks.iter_mut().map(|chunk| chunk.text()).collect();
        for (i, chunk) in chunks_string.into_iter().enumerate() {
            assert_eq!(chunk, test_cases[i]);
        }

        let tokenizer = LlmPreset::Llama3_1_8bInstruct
            .load()
            .unwrap()
            .model_base
            .tokenizer;
        let mut res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .overlap_percent(0.166666)
            .run_return_result(incoming_text)
            .unwrap();
        let chunks_string: Vec<String> = res.chunks.iter_mut().map(|chunk| chunk.text()).collect();
        for (i, chunk) in chunks_string.into_iter().enumerate() {
            assert_eq!(chunk, test_cases[i]);
        }
    }

    #[test]
    fn content() {
        let incoming_text = &test_text::TEXT.long.content;
        let absolute_length_max = 1024;

        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());
        let mut res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .overlap_percent(0.166666)
            .run_return_result(incoming_text)
            .unwrap();
        assert!(res.token_counts().iter().all(|&x| x <= absolute_length_max));

        let tokenizer = LlmPreset::Llama3_1_8bInstruct
            .load()
            .unwrap()
            .model_base
            .tokenizer;
        let mut res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .overlap_percent(0.166666)
            .run_return_result(incoming_text)
            .unwrap();
        assert!(res.token_counts().iter().all(|&x| x <= absolute_length_max));
    }
}
