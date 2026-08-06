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
use super::types::ChatRequest;

/// The default model, used by [`Model::default`]: Lovable's default chat model.
pub const DEFAULT_MODEL: &str = "google/gemini-3.6-flash";

/// Well-known model ids from the Lovable model catalog.
///
/// The catalog is an allowlist maintained by Lovable — every id sent to the
/// gateway must match it exactly, vendor prefix included, or the request is
/// rejected with a 400. These constants cover common picks; any other
/// catalog id works via [`Model::new`].
pub mod models {
    /// Google Gemini 3.6 Flash — Lovable's default chat model.
    pub const GEMINI_3_6_FLASH: &str = "google/gemini-3.6-flash";
    /// OpenAI GPT-5.5 on the chat completions surface.
    pub const GPT_5_5: &str = "openai/gpt-5.5";
}

/// The `Model` struct names a model from the Lovable model catalog.
///
/// Ids are vendor-prefixed strings (`vendor/model`), and the catalog is the
/// source of truth — this is a transparent wrapper around the id rather than
/// an enum, so new catalog entries need no crate update. Switching vendors is
/// a one-string change.
///
/// # Example
///
/// ```
/// use llm_chain_lovable::chat::Model;
///
/// let default = Model::default(); // google/gemini-3.6-flash
/// let pinned = Model::new("openai/gpt-5.5");
/// let parsed: Model = "google/gemini-3.6-flash".parse().unwrap();
/// assert_eq!(pinned.to_string(), "openai/gpt-5.5");
/// assert_eq!(parsed.as_str(), "google/gemini-3.6-flash");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Model(String);

impl Model {
    /// Creates a model from a vendor-prefixed catalog id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The model id as a string slice.
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

// Models serialize as their catalog id, e.g. `google/gemini-3.6-flash`.
llm_chain::impl_model_id_serde!(Model);

/// The `Step` struct represents an individual step within a chain for models
/// served by the Lovable AI Gateway. It is responsible for configuring the
/// input parameters for the model and providing the prompt.
///
/// By creating a `Step`, you can customize the model, prompt and request options used for a
/// particular stage within an `llm-chain` workflow. This allows for granular control over the
/// text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_lovable::chat::{Step, Model, ChatPromptTemplate, Options};
/// let model = Model::default();
/// let prompt = ChatPromptTemplate::system_and_user("You are an assistant that speaks like Shakespeare.", "tell me a joke");
///
/// let step = Step::new(model, prompt)
///     .with_options(Options::new().with_temperature(0.7).with_max_tokens(256));
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
        let mut request = ChatRequest::new(self.model.to_string(), self.prompt.format(parameters)?);
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
                "llm-chain-lovable::chat::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-lovable::chat::ChatPromptTemplate".to_string(),
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
    fn model_ids_round_trip() {
        for id in [models::GEMINI_3_6_FLASH, models::GPT_5_5] {
            let model: Model = id.parse().unwrap();
            assert_eq!(model.to_string(), id);
            assert_eq!(model.as_str(), id);
        }
        assert_eq!(Model::default().as_str(), DEFAULT_MODEL);
    }

    #[test]
    fn format_builds_a_request_with_options() {
        let step = Step::new(
            Model::default(),
            [(Role::System, "be brief"), (Role::User, "hi {}")],
        )
        .with_options(Options::new().with_temperature(0.1).with_max_tokens(64));
        let request = step.format(&Parameters::new_with_text("there")).unwrap();
        assert_eq!(request.model, "google/gemini-3.6-flash");
        assert!(!request.stream);
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[1].content.as_deref(), Some("hi there"));
        assert_eq!(request.temperature, Some(0.1));
        assert_eq!(request.max_tokens, Some(64));
    }

    #[test]
    fn steps_accept_plain_strings_as_models() {
        let step = Step::new("openai/gpt-5.5", [(Role::User, "hi")]);
        let request = step.format(&Parameters::new()).unwrap();
        assert_eq!(request.model, "openai/gpt-5.5");
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new(
            Model::new("openai/gpt-5.5"),
            [(Role::System, "be brief"), (Role::User, "hi")],
        )
        .with_options(Options::new().with_max_tokens(64));
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        assert!(yaml.contains("model: openai/gpt-5.5"));
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.model, Model::new("openai/gpt-5.5"));
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
