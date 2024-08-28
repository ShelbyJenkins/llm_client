mod common;
use common::*;

mod boolean_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn reason_one_round() {
        let mut gen = get_tiny_llm().await.unwrap().reason().boolean();
        let mut tests = primitive_tests(false).boolean().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_result().await.unwrap();
            let res = gen.primitive.parse_reason_result(&result).unwrap();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &Some(result),
                &Some(res),
            );
            case.result = res;
        }
        tests.check_results();
    }

    #[tokio::test]
    #[serial]
    pub async fn reason_one_round_nullable() {
        let mut gen = get_tiny_llm().await.unwrap().reason().boolean();

        let mut tests = primitive_tests(true).boolean().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let res = gen.return_optional_primitive().await.unwrap();
            print_results(&gen.base_req.instruct_prompt.prompt, &None::<String>, &res);
            case.result = res;
        }
        tests.check_results();
    }
}

mod intenger_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn reason_one_round() {
        let mut gen = get_tiny_llm().await.unwrap().reason().integer();
        let mut tests = primitive_tests(false).integer().unwrap();
        for case in &mut tests.cases {
            gen.reasoning_sentences(6);
            gen.conclusion_sentences(4);
            gen.instructions().set_content(&case.question);
            let result = gen.return_result().await.unwrap();
            let res = gen.primitive.parse_reason_result(&result).unwrap();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &Some(result),
                &Some(res),
            );
            case.result = res;
            gen.clear_request();
        }
        tests.check_results();
    }

    #[tokio::test]
    #[serial]
    pub async fn reason_one_round_nullable() {
        let mut gen = get_tiny_llm().await.unwrap().reason().integer();

        let mut tests = primitive_tests(true).integer().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_optional_primitive().await.unwrap();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &None::<String>,
                &Some(result),
            );
            case.result = result;
            gen.clear_request();
        }
        tests.check_results();
    }
}

mod exact_string_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn reason_one_round() {
        let mut gen = get_tiny_llm().await.unwrap().reason().exact_string();
        let mut tests = primitive_tests(false).exact_string().unwrap();
        for case in &mut tests.cases {
            gen.primitive.add_strings_to_allowed(&case.options);
            gen.instructions().set_content(&case.question);
            let result = gen.return_primitive().await.unwrap();
            case.result = Some(result.clone());
            gen.primitive.clear_primitive();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &None::<String>,
                &Some(result),
            );
            gen.clear_request();
        }
        tests.check_results();
    }

    #[tokio::test]
    #[serial]
    pub async fn reason_one_round_nullable() {
        let mut gen = get_tiny_llm().await.unwrap().reason().exact_string();

        let mut tests = primitive_tests(false).exact_string().unwrap();
        for case in &mut tests.cases {
            gen.primitive.add_strings_to_allowed(&case.options);
            gen.instructions().set_content(&case.question);
            let result = gen.return_optional_primitive().await.unwrap();
            print_results(
                &gen.base_req.instruct_prompt.prompt,
                &None::<String>,
                &Some(result.clone()),
            );
            gen.primitive.clear_primitive();
            case.result = result;
            gen.clear_request();
        }
        tests.check_results();
    }
}
