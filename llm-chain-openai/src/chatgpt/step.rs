use std::fmt;
use std::str::FromStr;

use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionRequestArgs};
#[cfg(feature = "serialization")]
use llm_chain::serialization::StorableEntity;
use llm_chain::{Parameters, traits};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;
use super::prompt::ChatPromptTemplate;

/// The `Model` enum represents the available OpenAI chat models. These models have different capabilities and performance characteristics, allowing you to choose the one that best suits your needs.
///
/// The `Other(String)` variant lets you use any model id that OpenAI serves,
/// so newly released models are always usable without waiting for a new
/// release of this crate.
///
/// # Example
///
/// ```
/// use llm_chain_openai::chatgpt::Model;
///
/// let flagship = Model::Gpt5;
/// let cheap_and_fast = Model::Gpt5Mini;
/// let custom_model: Model = "my-fine-tuned-model".parse().unwrap();
/// assert_eq!(Model::Gpt5.to_string(), "gpt-5");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Model {
    /// The flagship GPT-5 model.
    Gpt5,
    /// A faster, cheaper GPT-5 for well-defined tasks. A great default.
    #[default]
    Gpt5Mini,
    /// The smallest, fastest GPT-5.
    Gpt5Nano,
    /// GPT-4.1, strong at coding and long contexts (1M tokens).
    Gpt41,
    /// GPT-4.1 mini.
    Gpt41Mini,
    /// GPT-4.1 nano.
    Gpt41Nano,
    /// GPT-4o, the omni model.
    Gpt4o,
    /// GPT-4o mini.
    Gpt4oMini,
    /// The o3 reasoning model.
    O3,
    /// The o4-mini reasoning model.
    O4Mini,
    /// Any other model id, e.g. a fine-tune or a model released after this crate.
    Other(String),
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Gpt5 => "gpt-5",
            Self::Gpt5Mini => "gpt-5-mini",
            Self::Gpt5Nano => "gpt-5-nano",
            Self::Gpt41 => "gpt-4.1",
            Self::Gpt41Mini => "gpt-4.1-mini",
            Self::Gpt41Nano => "gpt-4.1-nano",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::O3 => "o3",
            Self::O4Mini => "o4-mini",
            Self::Other(model) => model,
        };
        f.write_str(s)
    }
}

impl FromStr for Model {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "gpt-5" => Self::Gpt5,
            "gpt-5-mini" => Self::Gpt5Mini,
            "gpt-5-nano" => Self::Gpt5Nano,
            "gpt-4.1" => Self::Gpt41,
            "gpt-4.1-mini" => Self::Gpt41Mini,
            "gpt-4.1-nano" => Self::Gpt41Nano,
            "gpt-4o" => Self::Gpt4o,
            "gpt-4o-mini" => Self::Gpt4oMini,
            "o3" => Self::O3,
            "o4-mini" => Self::O4Mini,
            other => Self::Other(other.to_string()),
        })
    }
}

// Models serialize as their OpenAI model id, e.g. `gpt-5-mini`.
#[cfg(feature = "serialization")]
impl Serialize for Model {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(feature = "serialization")]
impl<'de> Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(s.parse().expect("infallible"))
    }
}

/// The `Step` struct represents an individual step within a chain for OpenAI chat models. It is responsible for configuring the input parameters for the model and providing the prompt.
///
/// By creating a `Step`, you can customize the model and prompt used for a particular stage within an `llm-chain` workflow. This allows for granular control over the text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_openai::chatgpt::{Step, Model, ChatPromptTemplate};
/// let model = Model::default();
/// let prompt = ChatPromptTemplate::system_and_user("You are an assistant that speaks like Shakespeare.", "tell me a joke");
///
/// let step = Step::new(model, prompt);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Step {
    model: Model,
    prompt: ChatPromptTemplate,
}
impl Step {
    /// Creates a new step for the given model and prompt.
    pub fn new<P: Into<ChatPromptTemplate>>(model: Model, prompt: P) -> Step {
        let prompt = prompt.into();
        Step { model, prompt }
    }
}

impl traits::Step for Step {
    type Output = CreateChatCompletionRequest;
    type Error = FormatError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        let messages = self.prompt.format(parameters)?;
        Ok(CreateChatCompletionRequestArgs::default()
            .model(self.model.to_string())
            .messages(messages)
            .build()?)
    }
}

#[cfg(feature = "serialization")]
impl StorableEntity for Step {
    fn get_metadata() -> Vec<(String, String)> {
        vec![
            (
                "step-type".to_string(),
                "llm-chain-openai::chatgpt::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-openai::chatgpt::ChatPromptTemplate".to_string(),
            ),
        ]
    }
}
