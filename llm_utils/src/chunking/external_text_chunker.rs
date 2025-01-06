use std::{
    collections::VecDeque,
    sync::{atomic::AtomicBool, Arc},
};

use llm_models::tokenizer::LlmTokenizer;
use text_splitter::ChunkConfig;
use text_splitter::TextSplitter as TextSplitterExternal;
use tiktoken_rs::cl100k_base;

use crate::splitting::Separator;

use super::{Chunk, ChunkerConfig, ChunkerResult, ABSOLUTE_LENGTH_MIN_DEFAULT_RATIO};

/// Chunk incoming text using the [text-splitter](https://github.com/benbrandt/text-splitter) crate.
/// This is a dev-dependency for comparing the performance of the text-splitter crate with the TextChunker.
/// In the future we may integrate the text-splitter crate's markdown and code chunking capabilities into the [`TextChunker`].
pub fn chunk_text_with_text_splitter(
    incoming_text: &str,
    max_chunk_token_size: u32,
    overlap_percent: Option<f32>,
) -> Option<ChunkerResult> {
    let tiktoken_tokenizer = cl100k_base().unwrap();
    let chunking_start_time = std::time::Instant::now();
    let config = if let Some(overlap_percent) = overlap_percent {
        let overlap = (max_chunk_token_size as f32 * overlap_percent).floor() as u32;
        if overlap >= max_chunk_token_size {
            eprintln!("Overlap is greater than or equal to max_chunk_token_size");
            return None;
        }
        let max_chunk_token_size = max_chunk_token_size - overlap;
        ChunkConfig::new(max_chunk_token_size as usize)
            .with_trim(true)
            .with_sizer(tiktoken_tokenizer)
            .with_overlap(overlap as usize)
            .unwrap()
    } else {
        ChunkConfig::new(max_chunk_token_size as usize)
            .with_trim(true)
            .with_sizer(tiktoken_tokenizer)
    };
    let splitter = TextSplitterExternal::new(config);
    let text_chunks = splitter
        .chunks(incoming_text)
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let tokenizer = LlmTokenizer::new_tiktoken("gpt-4").unwrap();

    let dummy_config = Arc::new(ChunkerConfig {
        chunks_found: Arc::new(AtomicBool::new(true)),
        absolute_length_max: max_chunk_token_size,
        length_max: max_chunk_token_size as f32,
        absolute_length_min: (max_chunk_token_size as f32 * ABSOLUTE_LENGTH_MIN_DEFAULT_RATIO)
            as u32,
        overlap_percent,
        tokenizer: Arc::new(tokenizer),
        base_text: Arc::from(incoming_text),
        initial_separator: Separator::None,
        initial_splits: VecDeque::new(),
    });
    let mut chunks = Vec::new();
    for chunk in text_chunks.iter() {
        chunks.push(Chunk::dummy_chunk(&Arc::clone(&dummy_config), chunk));
    }
    Some(ChunkerResult::new(
        incoming_text,
        &dummy_config,
        chunking_start_time,
        chunks,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chunking::TOKENIZER_TIKTOKEN_DEFAULT, test_text::*, TextChunker};

    #[test]
    fn text_splitter() {
        let res = chunk_text_with_text_splitter(&TEXT.tiny.content, 128, None).unwrap();
        println!("{:#?}", res);
        let res = chunk_text_with_text_splitter(&TEXT.tiny.content, 128, Some(0.10)).unwrap();
        println!("{:#?}", res);

        let res = chunk_text_with_text_splitter(&TEXT.small.content, 256, None).unwrap();
        println!("{:#?}", res);
        let res = chunk_text_with_text_splitter(&TEXT.small.content, 256, Some(0.10)).unwrap();
        println!("{:#?}", res);

        let res = chunk_text_with_text_splitter(&TEXT.medium.content, 512, None).unwrap();
        println!("{:#?}", res);
        let res = chunk_text_with_text_splitter(&TEXT.medium.content, 512, Some(0.10)).unwrap();
        println!("{:#?}", res);

        let res = chunk_text_with_text_splitter(&TEXT.long.content, 1024, None).unwrap();
        println!("{:#?}", res);
        let res = chunk_text_with_text_splitter(&TEXT.long.content, 1024, Some(0.10)).unwrap();
        println!("{:#?}", res);

        let res = chunk_text_with_text_splitter(&TEXT.really_long.content, 1024, None).unwrap();
        println!("{:#?}", res);
        let res =
            chunk_text_with_text_splitter(&TEXT.really_long.content, 1024, Some(0.10)).unwrap();
        println!("{:#?}", res);
    }

    fn speed_comparison_runner(
        incoming_text: &str,
        absolute_length_max: u32,
        overlap_percent: f32,
    ) -> Vec<String> {
        let mut output_strings: Vec<String> = Vec::new();
        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());

        let title = format!(
            "Speed comparison for content: {}...\n content token_count: {}\nabsolute_length_max: {absolute_length_max}\noverlap_percent: {overlap_percent}",
            incoming_text.chars().take(22).collect::<String>(),
            tokenizer.count_tokens(incoming_text)
        );
        output_strings.push(title);

        let res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .run_return_result(incoming_text)
            .unwrap();
        let one = format!("\nduration without overlap: {:?}", res.chunking_duration);

        let res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .overlap_percent(overlap_percent)
            .run_return_result(incoming_text)
            .unwrap();
        let two = format!("\nduration with overlap: {:?}", res.chunking_duration);
        output_strings.push(format!("TextChunker Results:{one}{two}"));

        let res = chunk_text_with_text_splitter(incoming_text, absolute_length_max, None).unwrap();
        let one = format!("\nduration without overlap: {:?}", res.chunking_duration);

        let res = chunk_text_with_text_splitter(
            incoming_text,
            absolute_length_max,
            Some(overlap_percent),
        )
        .unwrap();
        let two = format!("\nduration with overlap: {:?}", res.chunking_duration);
        output_strings.push(format!("TextSplitter Results:{one}{two}"));

        output_strings
    }

    #[test]
    fn speed_comparison() {
        let mut output = Vec::new();

        let out = speed_comparison_runner(&TEXT.tiny.content, 128, 0.166666);
        output.extend(out);

        let out = speed_comparison_runner(&TEXT.small.content, 256, 0.166666);
        output.extend(out);

        let out = speed_comparison_runner(&TEXT.medium.content, 512, 0.166666);
        output.extend(out);

        let out = speed_comparison_runner(&TEXT.long.content, 1024, 0.166666);
        output.extend(out);

        let out = speed_comparison_runner(&TEXT.really_long.content, 2048, 0.166666);
        output.extend(out);

        for o in output {
            println!("\n{o}",);
        }
    }

    fn balance_formatter(mut res: ChunkerResult) -> String {
        let mut chunk_token_sizes = Vec::with_capacity(res.chunks.len());
        let mut largest_token_size = 0;
        let mut smallest_token_size = u32::MAX;
        let mut all_chunks_token_count = 0;

        let chunks = res.chunks_to_text();
        for chunk in &chunks {
            let token_count = res.tokenizer.count_tokens(chunk);
            chunk_token_sizes.push(token_count);
            all_chunks_token_count += token_count;

            if token_count > largest_token_size {
                largest_token_size = token_count;
            }

            if token_count < smallest_token_size {
                smallest_token_size = token_count;
            }
        }
        format!(
            "chunk_count: {}\navg_token_size: {}\nlargest_token_size: {}\nsmallest_token_size: {}",
            res.chunks.len(),
            all_chunks_token_count / u32::try_from(res.chunks.len()).unwrap(),
            largest_token_size,
            smallest_token_size,
        )
    }

    fn balance_comparison_runner(
        incoming_text: &str,
        absolute_length_max: u32,
        overlap_percent: f32,
    ) -> Vec<String> {
        let mut output_strings: Vec<String> = Vec::new();
        let tokenizer: Arc<LlmTokenizer> =
            Arc::new(LlmTokenizer::new_tiktoken(TOKENIZER_TIKTOKEN_DEFAULT).unwrap());

        let title = format!(
            "Balance comparison for content: {}...\n content token_count: {}\nabsolute_length_max: {absolute_length_max}\noverlap_percent: {overlap_percent}",
            incoming_text.chars().take(22).collect::<String>(),
            tokenizer.count_tokens(incoming_text)
        );
        output_strings.push(title);

        let res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .run_return_result(incoming_text)
            .unwrap();
        let one = balance_formatter(res);

        let res = TextChunker::new_with_tokenizer(&tokenizer)
            .max_chunk_token_size(absolute_length_max)
            .overlap_percent(overlap_percent)
            .run_return_result(incoming_text)
            .unwrap();
        let two = balance_formatter(res);
        output_strings.push(format!(
            "TextChunker result without overlap:\n{one}\n\nTextChunker result with overlap:\n{two}"
        ));

        let res = chunk_text_with_text_splitter(incoming_text, absolute_length_max, None).unwrap();
        let one = balance_formatter(res);

        let res = chunk_text_with_text_splitter(
            incoming_text,
            absolute_length_max,
            Some(overlap_percent),
        )
        .unwrap();
        let two = balance_formatter(res);
        output_strings.push(format!(
            "TextSplitter result without overlap:\n{one}\n\nTextSplitter result with overlap:\n{two}"
        ));
        output_strings
    }

    #[test]
    fn balance_comparison() {
        let mut output = Vec::new();

        let out = balance_comparison_runner(&TEXT.tiny.content, 128, 0.166666);
        output.extend(out);

        let out = balance_comparison_runner(&TEXT.small.content, 256, 0.166666);
        output.extend(out);

        let out = balance_comparison_runner(&TEXT.medium.content, 512, 0.166666);
        output.extend(out);

        let out = balance_comparison_runner(&TEXT.long.content, 1024, 0.166666);
        output.extend(out);

        let out = balance_comparison_runner(&TEXT.really_long.content, 2048, 0.166666);
        output.extend(out);

        for o in output {
            println!("\n{o}",);
        }
    }
}
