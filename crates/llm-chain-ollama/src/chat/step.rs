use std::fmt;
use std::str::FromStr;

#[cfg(feature = "serialization")]
use llm_chain::serialization::StorableEntity;
use llm_chain::{Parameters, traits};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;
use super::options::Options;
use super::prompt::ChatPromptTemplate;
use super::types::{ChatRequest, ModelOptions};

/// The default model, used by [`Model::default`].
pub const DEFAULT_MODEL: &str = "qwen3";

/// The `Model` struct names an Ollama model.
///
/// Unlike hosted APIs, Ollama has no fixed model lineup: any `name:tag` you
/// have pulled (or any cloud model) is valid, so this is a transparent wrapper
/// around the model name rather than an enum. Popular choices include `qwen3`,
/// `llama3.3`, `gemma3`, `deepseek-r1` and `gpt-oss`; cloud-hosted variants
/// use a `-cloud` suffix, e.g. `gpt-oss:120b-cloud`.
///
/// # Example
///
/// ```
/// use llm_chain_ollama::chat::Model;
///
/// let default = Model::default(); // qwen3
/// let pinned = Model::new("llama3.3:70b");
/// let parsed: Model = "deepseek-r1:8b".parse().unwrap();
/// assert_eq!(pinned.to_string(), "llama3.3:70b");
/// assert_eq!(parsed.as_str(), "deepseek-r1:8b");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Model(String);

impl Model {
    /// Creates a model from a name, optionally with a tag (e.g. `qwen3:32b`).
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The model name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for Model {
    fn default() -> Self {
        Self(DEFAULT_MODEL.to_string())
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for Model {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

impl From<&str> for Model {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for Model {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// Models serialize as their Ollama name, e.g. `qwen3:32b`.
llm_chain::impl_model_id_serde!(Model);

/// The `Step` struct represents an individual step within a chain for Ollama-served models. It is responsible for configuring the input parameters for the model and providing the prompt.
///
/// By creating a `Step`, you can customize the model, prompt and request options used for a
/// particular stage within an `llm-chain` workflow. This allows for granular control over the
/// text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_ollama::chat::{Step, Model, ChatPromptTemplate, Options, Think};
/// let model = Model::default();
/// let prompt = ChatPromptTemplate::system_and_user("You are an assistant that speaks like Shakespeare.", "tell me a joke");
///
/// let step = Step::new(model, prompt)
///     .with_options(Options::new().with_temperature(0.7).with_think(Think::Disabled));
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
    pub fn new<M: Into<Model>, P: Into<ChatPromptTemplate>>(model: M, prompt: P) -> Step {
        Step {
            model: model.into(),
            prompt: prompt.into(),
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
    type Output = ChatRequest;
    type Error = FormatError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        let mut request = ChatRequest {
            model: self.model.to_string(),
            messages: self.prompt.format(parameters)?,
            stream: false,
            think: None,
            format: None,
            keep_alive: None,
            options: ModelOptions::default(),
        };
        self.options.apply(&mut request);
        Ok(request)
    }
}

#[cfg(feature = "serialization")]
impl StorableEntity for Step {
    fn get_metadata() -> Vec<(String, String)> {
        vec![
            (
                "step-type".to_string(),
                "llm-chain-ollama::chat::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-ollama::chat::ChatPromptTemplate".to_string(),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::Role;
    use super::*;
    use llm_chain::traits::Step as _;

    #[test]
    fn model_names_round_trip() {
        for name in ["qwen3", "llama3.3:70b", "gpt-oss:120b-cloud"] {
            let model: Model = name.parse().unwrap();
            assert_eq!(model.to_string(), name);
            assert_eq!(model.as_str(), name);
        }
        assert_eq!(Model::default().as_str(), DEFAULT_MODEL);
    }

    #[test]
    fn format_builds_a_request_with_options() {
        let step = Step::new(
            Model::default(),
            [(Role::System, "be brief"), (Role::User, "hi {}")],
        )
        .with_options(Options::new().with_temperature(0.1).with_num_predict(64));
        let request = step.format(&Parameters::new_with_text("there")).unwrap();
        assert_eq!(request.model, "qwen3");
        assert!(!request.stream);
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[1].content, "hi there");
        assert_eq!(request.options.temperature, Some(0.1));
        assert_eq!(request.options.num_predict, Some(64));
    }

    #[test]
    fn steps_accept_plain_strings_as_models() {
        let step = Step::new("deepseek-r1:8b", [(Role::User, "hi")]);
        let request = step.format(&Parameters::new()).unwrap();
        assert_eq!(request.model, "deepseek-r1:8b");
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new(
            Model::new("llama3.3"),
            [(Role::System, "be brief"), (Role::User, "hi")],
        )
        .with_options(Options::new().with_num_predict(64));
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        assert!(yaml.contains("model: llama3.3"));
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.model, Model::new("llama3.3"));
        assert_eq!(parsed.options, step.options);
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn default_options_are_not_serialized() {
        let step = Step::new(Model::default(), [(Role::User, "hi")]);
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        assert!(!yaml.contains("options"));
    }
}
