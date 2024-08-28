mod common;
use common::*;

mod boolean_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round() {
        let mut gen = get_tiny_llm().await.unwrap().reason().boolean().decision();
        let mut tests = primitive_tests(false).boolean().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_primitive().await.unwrap();
            case.result = Some(result);
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
            let result = gen.return_optional_primitive().await.unwrap();
            case.result = result;
            gen.clear_request();
        }
        tests.check_results();
    }
}

mod intenger_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round() {
        let mut gen = get_tiny_llm().await.unwrap().reason().integer().decision();
        let mut tests = primitive_tests(false).integer().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_primitive().await.unwrap();
            case.result = Some(result);
        }
        tests.check_results();
    }

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round_nullable() {
        let mut gen = get_tiny_llm().await.unwrap().reason().integer().decision();
        let mut tests = primitive_tests(true).integer().unwrap();
        for case in &mut tests.cases {
            gen.instructions().set_content(&case.question);
            let result = gen.return_optional_primitive().await.unwrap();
            case.result = result;
        }
        tests.check_results();
    }
}

mod exact_string_reason_tests {
    use super::*;

    #[tokio::test]
    #[serial]
    pub async fn decision_one_round() {
        let mut gen = get_tiny_llm()
            .await
            .unwrap()
            .reason()
            .exact_string()
            .decision();
        gen.best_of_n_votes(3);
        let mut tests = primitive_tests(false).exact_string().unwrap();
        for case in &mut tests.cases {
            gen.reason.primitive.add_strings_to_allowed(&case.options);
            gen.instructions().set_content(&case.question);
            let result = gen.return_primitive().await.unwrap();
            case.result = Some(result.clone());
            gen.reason.primitive.clear_primitive();
            gen.clear_request();
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
            let result = gen.return_optional_primitive().await.unwrap();
            gen.reason.primitive.clear_primitive();
            case.result = result;
            gen.clear_request();
        }
        tests.check_results();
    }
}
