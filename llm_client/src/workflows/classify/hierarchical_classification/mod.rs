mod hierarchical_tagger;
mod tag;
mod tag_collection;
mod tag_describer;

use super::*;
pub use hierarchical_tagger::{Critera, HierarchicalEntityTagger};
pub use tag::Tag;
pub use tag_collection::TagCollection;

#[cfg(test)]
mod test {
    use crate::LlmClient;
    use llm_interface::requests::CompletionRequest;
    use llm_models::GgufPresetTrait;
    use workflows::classify::subject_of_text::ClassifySubjectOfText;

    use super::*;
    use crate::*;
    const INSTRUCTIONS: &str = "Apply classfication labels to the source of a microbial sample. The source is a place or a thing that the microbe was collected from. Our goal is to apply labels using specifically mentioned details from the input text. A category should apply to details specified about what or where the collection source. Classify the source of the microbial sample.";

    const ENTITY_DEFINITION: &str = "The sample microbe was collected from";

    #[tokio::test]
    #[ignore]
    pub async fn test_one() -> crate::Result<()> {
        // let llm_client: LlmClient = LlmClient::llama_cpp().meta_llama_3_1_8b_instruct().init().await?;
        let llm_client: LlmClient = LlmClient::llama_cpp()
            .mistral_nemo_instruct_2407()
            .init()
            .await?;

        let mut tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();

        tag_collection
            .populate_descriptions(&llm_client, INSTRUCTIONS)
            .await?;

        let critera = Critera {
            entity_definition: ENTITY_DEFINITION.to_owned(),
            instructions: INSTRUCTIONS.to_owned(),
        };

        let mut req = HierarchicalEntityTagger::new(
            &llm_client,
            "Gryllus bimaculatus",
            "Edible insect Gryllus bimaculatus (Pet Feed Store)",
            &critera,
            tag_collection.get_root_tag()?,
        );
        req.run().await?;
        println!("{}", req);

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    pub async fn test_full_workflow() -> crate::Result<()> {
        let llm_client: LlmClient = LlmClient::llama_cpp()
            .llama_3_1_nemotron_51b_instruct()
            .init()
            .await?;

        let input_text = "Edible insect Gryllus bimaculatus (Pet Feed Store)";
        let mut tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();

        tag_collection
            .populate_descriptions(&llm_client, INSTRUCTIONS)
            .await?;

        let critera = Critera {
            entity_definition: ENTITY_DEFINITION.to_owned(),
            instructions: INSTRUCTIONS.to_owned(),
        };

        let entity = ClassifySubjectOfText::new(
            CompletionRequest::new(llm_client.backend.clone()),
            input_text,
        )
        .run()
        .await?
        .subject
        .ok_or_else(|| anyhow::anyhow!("Entity not classified from the text."))?;

        let mut req = HierarchicalEntityTagger::new(
            &llm_client,
            &entity,
            input_text,
            &critera,
            tag_collection.get_root_tag()?,
        );
        req.run().await?;

        println!("{}", req);

        Ok(())
    }

    const CASES: &[(&str, &str)] = &[
        ("Ciliate: Metopus sp. strain SALT15A", "ciliate"),
        ("Coastal soil sample", "soil"),
        (
            "Edible insect Gryllus bimaculatus (Pet Feed Store)",
            "insect",
        ),
        ("Public spring water", "water"),
        ("River snow from South Saskatchewan River", "snow"),
        ("A green turtle on a log in a mountain lake.", "turtle"),
        (
            "Green turtle on log\nSunlight warms her emerald shell\nStillness all around",
            "turtle",
        ),
    ];

    #[tokio::test]
    #[ignore]
    pub async fn test_cases() -> crate::Result<()> {
        let llm_client: LlmClient = LlmClient::llama_cpp()
            .llama_3_1_8b_instruct()
            .init()
            .await?;

        let mut tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();

        tag_collection
            .populate_descriptions(&llm_client, INSTRUCTIONS)
            .await?;

        let critera = Critera {
            entity_definition: ENTITY_DEFINITION.to_owned(),
            instructions: INSTRUCTIONS.to_owned(),
        };
        let mut results = vec![];
        for (input_text, _) in CASES {
            let entity = ClassifySubjectOfText::new(
                CompletionRequest::new(llm_client.backend.clone()),
                input_text,
            )
            .run()
            .await?
            .subject
            .ok_or_else(|| anyhow::anyhow!("Entity not classified from the text."))?;

            let mut req = HierarchicalEntityTagger::new(
                &llm_client,
                &entity,
                input_text,
                &critera,
                tag_collection.get_root_tag()?,
            );
            req.run().await?;
            results.push(req);
        }
        for res in results {
            println!("{}", res);
        }

        Ok(())
    }
}
