use std::fmt;
use std::str::FromStr;

use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionRequestArgs};
#[cfg(feature = "serialization")]
use llm_chain::serialization::StorableEntity;
use llm_chain::{Parameters, traits};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;
use super::options::Options;
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
/// use llm_chain_openai::chat::Model;
///
/// let flagship = Model::Gpt56Sol;
/// let balanced = Model::default(); // gpt-5.6-terra
/// let high_volume = Model::Gpt56Luna;
/// let custom_model: Model = "my-fine-tuned-model".parse().unwrap();
/// assert_eq!(Model::Gpt56Sol.to_string(), "gpt-5.6-sol");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Model {
    // -- GPT-5.6 family: the current generation --
    /// The `gpt-5.6` alias; OpenAI routes it to `gpt-5.6-sol`.
    Gpt56,
    /// GPT-5.6 Sol: flagship capability.
    Gpt56Sol,
    /// GPT-5.6 Terra: strong performance at a lower price. A great default.
    #[default]
    Gpt56Terra,
    /// GPT-5.6 Luna: efficient, high-volume workloads.
    Gpt56Luna,

    // -- GPT-5.x --
    /// GPT-5.4.
    Gpt54,
    /// GPT-5.4 mini.
    Gpt54Mini,
    /// GPT-5.4 nano.
    Gpt54Nano,
    /// GPT-5.2.
    Gpt52,
    /// GPT-5.2 Pro: more compute for harder problems.
    Gpt52Pro,
    /// GPT-5.1.
    Gpt51,
    /// GPT-5.1 mini.
    Gpt51Mini,
    /// GPT-5.1 Codex, tuned for coding agents.
    Gpt51Codex,
    /// The original GPT-5.
    Gpt5,
    /// GPT-5 mini.
    Gpt5Mini,
    /// GPT-5 nano.
    Gpt5Nano,

    // -- Previous generations --
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
            Self::Gpt56 => "gpt-5.6",
            Self::Gpt56Sol => "gpt-5.6-sol",
            Self::Gpt56Terra => "gpt-5.6-terra",
            Self::Gpt56Luna => "gpt-5.6-luna",
            Self::Gpt54 => "gpt-5.4",
            Self::Gpt54Mini => "gpt-5.4-mini",
            Self::Gpt54Nano => "gpt-5.4-nano",
            Self::Gpt52 => "gpt-5.2",
            Self::Gpt52Pro => "gpt-5.2-pro",
            Self::Gpt51 => "gpt-5.1",
            Self::Gpt51Mini => "gpt-5.1-mini",
            Self::Gpt51Codex => "gpt-5.1-codex",
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
            "gpt-5.6" => Self::Gpt56,
            "gpt-5.6-sol" => Self::Gpt56Sol,
            "gpt-5.6-terra" => Self::Gpt56Terra,
            "gpt-5.6-luna" => Self::Gpt56Luna,
            "gpt-5.4" => Self::Gpt54,
            "gpt-5.4-mini" => Self::Gpt54Mini,
            "gpt-5.4-nano" => Self::Gpt54Nano,
            "gpt-5.2" => Self::Gpt52,
            "gpt-5.2-pro" => Self::Gpt52Pro,
            "gpt-5.1" => Self::Gpt51,
            "gpt-5.1-mini" => Self::Gpt51Mini,
            "gpt-5.1-codex" => Self::Gpt51Codex,
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

// Models serialize as their OpenAI model id, e.g. `gpt-5.6-terra`.
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
/// By creating a `Step`, you can customize the model, prompt and request options used for a
/// particular stage within an `llm-chain` workflow. This allows for granular control over the
/// text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_openai::chat::{Step, Model, ChatPromptTemplate, Options, ReasoningEffort};
/// let model = Model::default();
/// let prompt = ChatPromptTemplate::system_and_user("You are an assistant that speaks like Shakespeare.", "tell me a joke");
///
/// let step = Step::new(model, prompt)
///     .with_options(Options::new().with_reasoning_effort(ReasoningEffort::Low));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Step {
    model: Model,
    prompt: ChatPromptTemplate,
    #[cfg_attr(
        feature = "serialization",
        serde(default, skip_serializing_if = "Options::is_default")
    )]
    options: Options,
}
impl Step {
    /// Creates a new step for the given model and prompt, with default request options.
    pub fn new<P: Into<ChatPromptTemplate>>(model: Model, prompt: P) -> Step {
        let prompt = prompt.into();
        Step {
            model,
            prompt,
            options: Options::default(),
        }
    }
    /// Sets the request options for this step.
    pub fn with_options(mut self, options: Options) -> Step {
        self.options = options;
        self
    }
}

impl traits::Step for Step {
    type Output = CreateChatCompletionRequest;
    type Error = FormatError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        let messages = self.prompt.format(parameters)?;
        let mut args = CreateChatCompletionRequestArgs::default();
        args.model(self.model.to_string()).messages(messages);
        self.options.apply(&mut args);
        Ok(args.build()?)
    }
}

#[cfg(feature = "serialization")]
impl StorableEntity for Step {
    fn get_metadata() -> Vec<(String, String)> {
        vec![
            (
                "step-type".to_string(),
                "llm-chain-openai::chat::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-openai::chat::ChatPromptTemplate".to_string(),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::Role;
    use llm_chain::traits::Step as _;

    #[test]
    fn model_ids_round_trip() {
        let models = [
            Model::Gpt56,
            Model::Gpt56Sol,
            Model::Gpt56Terra,
            Model::Gpt56Luna,
            Model::Gpt54,
            Model::Gpt54Mini,
            Model::Gpt54Nano,
            Model::Gpt52,
            Model::Gpt52Pro,
            Model::Gpt51,
            Model::Gpt51Mini,
            Model::Gpt51Codex,
            Model::Gpt5,
            Model::Gpt5Mini,
            Model::Gpt5Nano,
            Model::Gpt41,
            Model::Gpt41Mini,
            Model::Gpt41Nano,
            Model::Gpt4o,
            Model::Gpt4oMini,
            Model::O3,
            Model::O4Mini,
            Model::Other("my-fine-tune".to_string()),
        ];
        for model in models {
            let parsed: Model = model.to_string().parse().unwrap();
            assert_eq!(parsed, model);
        }
    }

    #[test]
    fn format_builds_a_request_with_options() {
        let step = Step::new(
            Model::default(),
            [(Role::System, "be brief"), (Role::User, "hi {}")],
        )
        .with_options(Options::new().with_temperature(0.1));
        let request = step.format(&Parameters::new_with_text("there")).unwrap();
        assert_eq!(request.model, "gpt-5.6-terra");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.temperature, Some(0.1));
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new(
            Model::Gpt56Luna,
            [(Role::Developer, "be brief"), (Role::User, "hi")],
        )
        .with_options(Options::new().with_max_completion_tokens(64));
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.model, Model::Gpt56Luna);
        assert_eq!(parsed.options, step.options);
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn steps_without_options_still_deserialize() {
        // YAML written before `options` existed must keep loading.
        let yaml = "model: gpt-5-mini\nprompt:\n  messages:\n  - role: system\n    content:\n      template: hello\n";
        let parsed: Step = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(parsed.model, Model::Gpt5Mini);
        assert!(parsed.options.is_default());
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn default_options_are_not_serialized() {
        let step = Step::new(Model::default(), [(Role::User, "hi")]);
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        assert!(!yaml.contains("options"));
    }
}
