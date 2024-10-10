use super::*;

mod decision_unit_tests {
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
    let mut gen = llm_client.reason().boolean().decision();

    let mut tests = TestSetsLoader::new()
        .test_level_enum(test_level)
        .boolean()?;

    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let result = gen.return_result().await?;
        let res = gen.parse_decision_result(&result)?;
        print_results(&gen.reason.base_req.prompt, &Some(result), &Some(res));
        case.result = res;
        gen.reset_request();
    }

    tests.check_results();
    Ok(())
}

pub(super) async fn boolean_optional_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.reason().boolean().decision();

    let mut tests = TestSetsLoader::new_optional()
        .test_level_enum(test_level)
        .boolean()?;

    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let result = gen.return_optional_result().await?;
        let res = gen.parse_decision_result(&result)?;
        print_results(&gen.reason.base_req.prompt, &Some(result), &Some(res));
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
    let mut gen = llm_client.reason().integer().decision();

    let mut tests = TestSetsLoader::new()
        .test_level_enum(test_level)
        .integer()?;
    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let result = gen.return_result().await?;
        let res = gen.parse_decision_result(&result)?;
        print_results(&gen.reason.base_req.prompt, &Some(result), &Some(res));
        case.result = res;
        gen.reset_request();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn integer_optional_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.reason().integer().decision();
    let mut tests = TestSetsLoader::new_optional()
        .test_level_enum(test_level)
        .integer()?;
    for case in &mut tests.cases {
        gen.instructions().set_content(&case.question);
        let result = gen.return_optional_result().await?;
        let res = gen.parse_decision_result(&result)?;
        print_results(&gen.reason.base_req.prompt, &Some(result), &Some(res));
        case.result = res;
        gen.reset_request();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn exact_string_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.reason().exact_string().decision();
    let mut tests = TestSetsLoader::new()
        .test_level_enum(test_level)
        .exact_string()?;

    for case in &mut tests.cases {
        gen.reason.primitive.add_strings_to_allowed(&case.options);
        gen.instructions().set_content(&case.question);
        let result = gen.return_result().await?;
        let res = gen.parse_decision_result(&result)?;
        case.result.clone_from(&res);
        print_results(&gen.reason.base_req.prompt, &Some(result), &Some(res));
        gen.reset_request();
        gen.reason.primitive.clear_primitive();
    }
    tests.check_results();
    Ok(())
}

pub(super) async fn exact_string_optional_integration_tester(
    llm_client: &LlmClient,
    test_level: &TestLevel,
) -> crate::Result<()> {
    let mut gen = llm_client.reason().exact_string().decision();
    let mut tests = TestSetsLoader::new_optional()
        .test_level_enum(test_level)
        .exact_string()?;

    for case in &mut tests.cases {
        gen.reason.primitive.add_strings_to_allowed(&case.options);
        gen.instructions().set_content(&case.question);
        let result = gen.return_optional_result().await?;
        let res = gen.parse_decision_result(&result)?;
        case.result.clone_from(&res);
        print_results(&gen.reason.base_req.prompt, &Some(result), &Some(res));
        gen.reset_request();
        gen.reason.primitive.clear_primitive();
    }

    tests.check_results();
    Ok(())
}
