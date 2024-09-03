mod common;
use common::*;

mod boolean_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round() {
        let llm_client = get_tiny_llm().await.unwrap();
        let mut reason_gen = llm_client.reason().boolean();
        reason_gen.reasoning_sentences(5).conclusion_sentences(4);
        let mut gen = reason_gen.decision();
        gen.best_of_n_votes(7);

        let mut tests = primitive_tests(false).boolean().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            gen.supporting_material().set_content("Consider that this is a common question, but may have small details is compared to how the question is normally phrased. Consider small details that might make this a 'trick' question before giving discussing the problem.");
            let result = gen.return_result().await.unwrap();
            let res = gen.parse_decision_result(&result).unwrap();
            print_results(
                &gen.reason.base_req.instruct_prompt.prompt,
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
    pub async fn decision_one_round_nullable() {
        let mut gen = get_tiny_llm().await.unwrap().reason().boolean().decision();
        let mut tests = primitive_tests(true).boolean().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_result().await.unwrap();
            let res = gen.parse_decision_result(&result).unwrap();
            print_results(
                &gen.reason.base_req.instruct_prompt.prompt,
                &Some(result),
                &Some(res),
            );
            case.result = res;
            gen.clear_request();
        }
        tests.check_results();
    }
}

mod integer_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round() {
        let llm_client = get_tiny_llm().await.unwrap();
        let mut reason_gen = llm_client.reason().integer();
        reason_gen.reasoning_sentences(5).conclusion_sentences(4);
        let mut gen = reason_gen.decision();
        gen.best_of_n_votes(7);

        let mut tests = primitive_tests(false).integer().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_result().await.unwrap();
            let res = gen.parse_decision_result(&result).unwrap();
            print_results(
                &gen.reason.base_req.instruct_prompt.prompt,
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
    pub async fn decision_one_round_nullable() {
        let mut gen = get_tiny_llm().await.unwrap().reason().integer().decision();
        let mut tests = primitive_tests(true).integer().unwrap();
        for case in &mut tests.cases {
            let result = gen.return_result().await.unwrap();
            let res = gen.parse_decision_result(&result).unwrap();
            print_results(
                &gen.reason.base_req.instruct_prompt.prompt,
                &Some(result),
                &Some(res),
            );
            case.result = res;
            gen.clear_request();
        }
        tests.check_results();
    }
}

mod exact_string_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round() {
        let llm_client = get_tiny_llm().await.unwrap();
        let mut reason_gen = llm_client.reason().exact_string();
        reason_gen.reasoning_sentences(5).conclusion_sentences(4);
        let mut gen = reason_gen.decision();
        gen.best_of_n_votes(7);

        let mut tests = primitive_tests(false).exact_string().unwrap();
        for case in &mut tests.cases {
            gen.reason.primitive.add_strings_to_allowed(&case.options);
            gen.instructions().set_content(&case.question);
            let result = gen.return_result().await.unwrap();
            let res = gen.parse_decision_result(&result).unwrap();
            case.result.clone_from(&res);
            print_results(
                &gen.reason.base_req.instruct_prompt.prompt,
                &Some(result),
                &Some(res),
            );
            gen.clear_request();
            gen.reason.primitive.clear_primitive();
        }
        tests.check_results();
    }

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round_nullable() {
        let mut gen = get_tiny_llm()
            .await
            .unwrap()
            .reason()
            .exact_string()
            .decision();
        gen.best_of_n_votes(3);
        let mut tests = primitive_tests(true).exact_string().unwrap();
        for case in &mut tests.cases {
            gen.reason.primitive.add_strings_to_allowed(&case.options);
            gen.instructions().set_content(&case.question);
            let result = gen.return_result().await.unwrap();
            let res = gen.parse_decision_result(&result).unwrap();
            case.result.clone_from(&res);
            print_results(
                &gen.reason.base_req.instruct_prompt.prompt,
                &Some(result),
                &Some(res),
            );
            gen.clear_request();
            gen.reason.primitive.clear_primitive();
        }
        tests.check_results();
    }
}
