mod common;
use common::*;

// #[tokio::test]
// #[serial]
// async fn extract_entities_test() {
//     let llm_client = get_tiny_llm().await.unwrap();
//     let tests: Vec<ExtractEntitiesTest> = TestSetsLoader::default()
//         .optional(true)
//         .test_level_zero()
//         .extract_entities()
//         .unwrap();

//     let mut test_results: Vec<(
//         ExtractEntitiesTest,
//         llm_client::workflows::nlp::extract::_entities::ExtractEntitiesResult,
//     )> = Vec::new();
//     for test in tests {
//         let mut gen = llm_client.nlp().extract().entities().entity_type("topic");
//         gen.supporting_material()
//             .set_content(&test.supporting_material);
//         let result = gen.run_return_result().await.unwrap();
//         match (PRINT_PROMPT, PRINT_RESULT) {
//             (true, true) => {
//                 println!("{}", gen.base_req.instruct_prompt.prompt);
//                 println!("{result}");
//             }
//             (true, false) => {
//                 println!("{}", gen.base_req.instruct_prompt.prompt);
//             }
//             (false, true) => println!("{result}"),
//             (false, false) => (),
//         }
//         test_results.push((test, result));
//     }
//     test_results.into_iter().for_each(|(test, result)| {
//         test.check_result(result);
//     });
// }

#[tokio::test]
#[serial]
async fn extract_urls_test() {
    let llm_client = get_tiny_llm().await.unwrap();
    let tests = TestSetsLoader::default()
        .optional(true)
        .test_level_one()
        .extract_urls()
        .unwrap();

    let mut test_results: Vec<(
        ExtractUrlsTest,
        llm_client::workflows::nlp::extract::urls::ExtractUrlResult,
    )> = Vec::new();
    for test in tests {
        let mut gen = llm_client.nlp().extract().urls();
        gen.instructions().set_content(&test.instructions);
        gen.supporting_material()
            .set_content(&test.supporting_material);
        let result = gen.run_return_result().await.unwrap();
        if PRINT_PROMPT {
            println!("{}", gen.base_req.instruct_prompt.prompt);
        }
        if PRINT_WORKFLOW_RESULT {
            println!("{}", result);
        }

        test_results.push((test, result));
    }
    test_results.into_iter().for_each(|(test, result)| {
        test.check_result(result);
    });
}
