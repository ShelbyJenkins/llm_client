use super::*;

mod basic_primitive_unit_tests {
    use super::*;
    #[tokio::test]
    #[serial]
    #[ignore]
    async fn boolean() -> crate::Result<()> {
        let llm_client = default_tiny_llm().await?;
        boolean_integration_tester(&llm_client, &TestLevel::IntegrationTest).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    async fn boolean_optional() -> crate::Result<()> {
        let llm_client = default_tiny_llm().await?;
        boolean_optional_integration_tester(&llm_client, &TestLevel::IntegrationTest).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    async fn integer() -> crate::Result<()> {
        let llm_client = default_tiny_llm().await?;
        integer_integration_tester(&llm_client, &TestLevel::IntegrationTest).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    async fn integer_optional() -> crate::Result<()> {
        let llm_client = default_tiny_llm().await?;
        integer_optional_integration_tester(&llm_client, &TestLevel::IntegrationTest).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    async fn exact_string() -> crate::Result<()> {
        let llm_client = default_tiny_llm().await?;
        exact_string_integration_tester(&llm_client, &TestLevel::IntegrationTest).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    async fn exact_string_optional() -> crate::Result<()> {
        let llm_client = default_tiny_llm().await?;
        exact_string_optional_integration_tester(&llm_client, &TestLevel::IntegrationTest).await?;
        Ok(())
    }
}

pub(super) async fn run(llm_client: &LlmClient, test_level: &TestLevel) -> crate::Result<()> {
    boolean_integration_tester(&llm_client, &test_level).await?;
    integer_integration_tester(&llm_client, &test_level).await?;
    exact_string_integration_tester(&llm_client, &test_level).await?;
    Ok(())
}

pub(super) async fn run_optional(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    boolean_optional_integration_tester(&llm_client, &test_level).await?;
    integer_optional_integration_tester(&llm_client, &test_level).await?;
    exact_string_optional_integration_tester(&llm_client, &test_level).await?;
    Ok(())
}

pub(super) async fn boolean_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_primitive().boolean();

    let mut tests = TestSetsLoader::new()
        .test_level_enum(test_level)
        .boolean()?;

    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_primitive().await?;
        print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
        case.result = Some(res);
        gen.reset_request();
    }

    tests.check_results();
    Ok(())
}

pub(super) async fn boolean_optional_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_primitive().boolean();

    let mut tests = TestSetsLoader::new_optional()
        .test_level_enum(test_level)
        .boolean()?;

    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_optional_primitive().await?;
        print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
        case.result = res;
        gen.reset_request();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn integer_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_primitive().integer();

    let mut tests = TestSetsLoader::new()
        .test_level_enum(test_level)
        .integer()?;
    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_primitive().await?;
        print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
        case.result = Some(res);
        gen.reset_request();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn integer_optional_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_primitive().integer();
    let mut tests = TestSetsLoader::new_optional()
        .test_level_enum(test_level)
        .integer()?;
    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_optional_primitive().await?;
        case.result = res;
        print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
        gen.reset_request();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn exact_string_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_primitive().exact_string();
    let mut tests = TestSetsLoader::new()
        .test_level_enum(test_level)
        .exact_string()?;

    for case in &mut tests.cases {
        gen.primitive.add_strings_to_allowed(&case.options);
        gen.instructions().set_content(&case.question);
        let res = gen.return_primitive().await?;
        case.result = Some(res.clone());
        print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
        gen.primitive.clear_primitive();
        gen.reset_request();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn exact_string_optional_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_primitive().exact_string();
    let mut tests = TestSetsLoader::new_optional()
        .test_level_enum(test_level)
        .exact_string()?;

    for case in &mut tests.cases {
        gen.primitive.add_strings_to_allowed(&case.options);
        gen.instructions().set_content(&case.question);
        let res = gen.return_optional_primitive().await?;
        print_results(&gen.base_req.prompt, &None::<String>, &Some(res.clone()));
        gen.primitive.clear_primitive();
        case.result = res;
        gen.reset_request();
    }

    tests.check_results();
    Ok(())
}

// fn sentences_primitive_test_questions() -> Vec<String> {
//     vec![
//         "Do scientist believe the moon is made of cheese?".to_string(),
//         "Would most people say the sky is blue?".to_string(),
//         "Would most people say Italy is in Asia?".to_string(),
//         "In a fictional universe HAL, Skynet, and Agent Smith are real beings. They walk into a bar in this fictional universe. Based on their history, should the fictional bartender should be worried?".to_string(),
//         "I wonder if the Roman colesseum older than the Egyptian pyramids?".to_string(),
//         "Most people believe, correctly, that a tomato is a vegetable.".to_string(),

//     ]
// }

// #[tokio::test]
// #[serial]
// async fn sentences() {
//     let mut gen = get_tiny_llm().await.unwrap().basic_primitive().sentences();
//     gen.primitive.min_count(2).max_count(3);

//     let questions = sentences_primitive_test_questions();
//     for question in questions {
//         gen.instructions().set_content(question);
//         let res = gen.return_primitive().await?;
//         print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
//         gen.reset_request();
//     }
// }

// fn words_primitive_test_questions() -> Vec<(String, String)> {
//     vec![
//             (
//                 "Generate keywords for the main topics discussed in the given text.".to_string(),
//                 "Climate change is one of the most pressing issues of our time. It affects weather patterns, sea levels, and ecosystems worldwide. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause of global warming. Governments and individuals must take action to reduce carbon emissions and mitigate the effects of climate change.".to_string()
//             ),
//             (
//                 "List three keywords in the passage.".to_string(),
//                 "Regular exercise offers numerous benefits for both physical and mental health. It helps maintain a healthy weight, reduces the risk of chronic diseases such as heart disease and diabetes, and strengthens bones and muscles. Exercise also improves mood, reduces stress and anxiety, and enhances cognitive function and memory. Additionally, it can increase energy levels and promote better sleep quality.".to_string()
//             ),
//             (
//                 "Classify the primary character traits of the protagonist based on the given excerpt. Return only keywords.".to_string(),
//                 "As Sarah walked into the crowded room, she took a deep breath and squared her shoulders. Despite her nervousness, she was determined to make a good impression. She had spent weeks preparing for this presentation, and she wasn't about to let her fear of lic speaking hold her back. With a confident smile, she approached the podium and began her speech, her voice steady and clear.".to_string()
//             ),
//             (
//                 "List pros keywords and two cons keywords of social media usage mentioned in the text.".to_string(),
//                 "Social media has become an integral part of modern life, offering both benefits and drawbacks. On the positive side, it allows for instant communication and connection with friends and family across the globe, and provides a platform for sharing ideas and information. However, excessive use of social media can lead to addiction-like behavior, decreased productivity, and increased feelings of loneliness and depression. Additionally, the spread of misinformation and privacy concerns are significant issues associated with social media platforms.".to_string()
//             )
//         ]
// }

// #[tokio::test]
// #[serial]
// async fn words() {
//     let mut gen = get_tiny_llm().await.unwrap().basic_primitive().words();
//     gen.primitive.min_count(2).max_count(5).concatenator(", ");

//     let cases = words_primitive_test_questions();
//     for (instructions, supporting_material) in cases {
//         gen.instructions().set_content(instructions);
//         gen.supporting_material().set_content(supporting_material);
//         let res = gen.return_primitive().await?;
//         print_results(&gen.base_req.prompt, &None::<String>, &Some(res));
//         gen.reset_request();
//     }
// }
