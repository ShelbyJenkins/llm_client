mod common;
use common::*;

#[tokio::test]
#[serial]
pub async fn boolean() {
    let llm_client = get_tiny_llm().await.unwrap();
    let mut gen = llm_client.basic_primitive().boolean();

    let mut tests = primitive_tests(false).boolean().unwrap();
    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_primitive().await.unwrap();
        print_results(
            &gen.base_req.instruct_prompt.prompt,
            &None::<String>,
            &Some(res),
        );
        case.result = Some(res);
        gen.clear_request();
    }

    let mut tests_opt = primitive_tests(true).boolean().unwrap();
    for case in &mut tests_opt.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_optional_primitive().await.unwrap();
        print_results(
            &gen.base_req.instruct_prompt.prompt,
            &None::<String>,
            &Some(res),
        );
        case.result = res;
        gen.clear_request();
    }

    tests.check_results();
    tests_opt.check_results();
}

#[tokio::test]
#[serial]
pub async fn integer() {
    let mut gen = get_tiny_llm().await.unwrap().basic_primitive().integer();

    let mut tests = primitive_tests(false).integer().unwrap();
    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_primitive().await.unwrap();
        print_results(
            &gen.base_req.instruct_prompt.prompt,
            &None::<String>,
            &Some(res),
        );
        case.result = Some(res);
        gen.clear_request();
    }

    let mut tests_opt = primitive_tests(true).integer().unwrap();
    for case in &mut tests_opt.cases {
        gen.instructions().set_content(&case.question);
        let res = gen.return_optional_primitive().await.unwrap();
        case.result = res;
        print_results(
            &gen.base_req.instruct_prompt.prompt,
            &None::<String>,
            &Some(res),
        );
        gen.clear_request();
    }
    tests.check_results();
    tests_opt.check_results();
}

mod sentences_primitive_tests {
    use super::*;

    fn test_questions() -> Vec<String> {
        vec![
        "Do scientist believe the moon is made of cheese?".to_string(),
        "Would most people say the sky is blue?".to_string(),
        "Would most people say Italy is in Asia?".to_string(),
        "In a fictional universe HAL, Skynet, and Agent Smith are real beings. They walk into a bar in this fictional universe. Based on their history, should the fictional bartender should be worried?".to_string(),
        "I wonder if the Roman colesseum older than the Egyptian pyramids?".to_string(),
        "Most people believe, correctly, that a tomato is a vegetable.".to_string(),

    ]
    }

    #[tokio::test]
    #[serial]
    pub async fn generate_primitive() {
        let mut gen = get_tiny_llm().await.unwrap().basic_primitive().sentences();
        gen.primitive.min_count(2).max_count(3);

        let questions = test_questions();
        for question in questions {
            gen.instructions().set_content(question);
            let res = gen.return_primitive().await.unwrap();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &None::<String>,
                &Some(res),
            );
            gen.clear_request();
        }
    }
}

mod words_primitive_tests {
    use super::*;

    fn test_questions() -> Vec<(String, String)> {
        vec![
            (
                "Generate keywords for the main topics discussed in the given text.".to_string(),
                "Climate change is one of the most pressing issues of our time. It affects weather patterns, sea levels, and ecosystems worldwide. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause of global warming. Governments and individuals must take action to reduce carbon emissions and mitigate the effects of climate change.".to_string()
            ),
            (
                "List three keywords in the passage.".to_string(),
                "Regular exercise offers numerous benefits for both physical and mental health. It helps maintain a healthy weight, reduces the risk of chronic diseases such as heart disease and diabetes, and strengthens bones and muscles. Exercise also improves mood, reduces stress and anxiety, and enhances cognitive function and memory. Additionally, it can increase energy levels and promote better sleep quality.".to_string()
            ),
            (
                "Classify the primary character traits of the protagonist based on the given excerpt. Return only keywords.".to_string(),
                "As Sarah walked into the crowded room, she took a deep breath and squared her shoulders. Despite her nervousness, she was determined to make a good impression. She had spent weeks preparing for this presentation, and she wasn't about to let her fear of public speaking hold her back. With a confident smile, she approached the podium and began her speech, her voice steady and clear.".to_string()
            ),
            (
                "List pros keywords and two cons keywords of social media usage mentioned in the text.".to_string(),
                "Social media has become an integral part of modern life, offering both benefits and drawbacks. On the positive side, it allows for instant communication and connection with friends and family across the globe, and provides a platform for sharing ideas and information. However, excessive use of social media can lead to addiction-like behavior, decreased productivity, and increased feelings of loneliness and depression. Additionally, the spread of misinformation and privacy concerns are significant issues associated with social media platforms.".to_string()
            )
        ]
    }

    #[tokio::test]
    #[serial]
    pub async fn generate_primitive() {
        let mut gen = get_tiny_llm().await.unwrap().basic_primitive().words();
        gen.primitive.min_count(2).max_count(5).concatenator(", ");

        let cases = test_questions();
        for (instructions, supporting_material) in cases {
            gen.instructions().set_content(instructions);
            gen.supporting_material().set_content(supporting_material);
            let res = gen.return_primitive().await.unwrap();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &None::<String>,
                &Some(res),
            );
            gen.clear_request();
        }
    }
}

#[tokio::test]
#[serial]
pub async fn exact_string() {
    let mut gen = get_tiny_llm()
        .await
        .unwrap()
        .basic_primitive()
        .exact_string();

    let mut tests = primitive_tests(false).exact_string().unwrap();
    for case in &mut tests.cases {
        gen.primitive.add_strings_to_allowed(&case.options);
        gen.instructions().set_content(&case.question);
        let res = gen.return_primitive().await.unwrap();
        case.result = Some(res.clone());
        print_results(
            &gen.base_req.instruct_prompt.prompt,
            &None::<String>,
            &Some(res),
        );
        gen.primitive.clear_primitive();
        gen.clear_request();
    }

    let mut tests_opt = primitive_tests(true).exact_string().unwrap();
    for case in &mut tests_opt.cases {
        gen.primitive.add_strings_to_allowed(&case.options);
        gen.instructions().set_content(&case.question);
        let res = gen.return_optional_primitive().await.unwrap();
        print_results(
            &gen.base_req.instruct_prompt.prompt,
            &None::<String>,
            &Some(res.clone()),
        );
        gen.primitive.clear_primitive();
        case.result = res;
        gen.clear_request();
    }
    tests.check_results();
    tests_opt.check_results();
}
