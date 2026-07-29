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
use super::types::{Content, GenerateContentRequest};

/// The `Model` enum represents the available Gemini models.
///
/// Google ships dateless model ids (e.g. `gemini-2.5-flash`) that point at the
/// latest stable snapshot of that model, plus `-preview` ids for models still
/// in preview. The `Other(String)` variant lets you pin a dated snapshot or
/// use a model released after this crate.
///
/// # Example
///
/// ```
/// use llm_chain_gemini::generate_content::Model;
///
/// let balanced = Model::default(); // gemini-3.6-flash
/// let strongest = Model::Gemini31ProPreview;
/// let high_volume = Model::Gemini31FlashLite;
/// let pinned: Model = "gemini-2.5-flash-preview-05-20".parse().unwrap();
/// assert_eq!(Model::Gemini25Pro.to_string(), "gemini-2.5-pro");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Model {
    // -- Gemini 3 family: the current generation --
    /// Gemini 3.6 Flash: the latest-generation Flash model for fast coding,
    /// reasoning, and agentic workflows. A great default.
    #[default]
    Gemini36Flash,
    /// Gemini 3.5 Flash: high-efficiency previous Flash iteration.
    Gemini35Flash,
    /// Gemini 3.1 Pro (preview): the strongest Gemini reasoning model.
    Gemini31ProPreview,
    /// Gemini 3.1 Flash-Lite: the most cost-efficient Gemini 3 model for
    /// high-volume classification, summarization, and extraction.
    Gemini31FlashLite,
    /// Gemini 3 Flash (preview): the first Gemini 3 Flash release.
    Gemini3FlashPreview,

    // -- Gemini 2.5 family: the previous stable generation --
    /// Gemini 2.5 Pro: strong multimodal and complex reasoning.
    Gemini25Pro,
    /// Gemini 2.5 Flash: balanced cost and latency.
    Gemini25Flash,
    /// Gemini 2.5 Flash-Lite: cheapest and fastest 2.5 model.
    Gemini25FlashLite,

    /// Any other model id, e.g. a dated snapshot or a model released after this crate.
    Other(String),
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Gemini36Flash => "gemini-3.6-flash",
            Self::Gemini35Flash => "gemini-3.5-flash",
            Self::Gemini31ProPreview => "gemini-3.1-pro-preview",
            Self::Gemini31FlashLite => "gemini-3.1-flash-lite",
            Self::Gemini3FlashPreview => "gemini-3-flash-preview",
            Self::Gemini25Pro => "gemini-2.5-pro",
            Self::Gemini25Flash => "gemini-2.5-flash",
            Self::Gemini25FlashLite => "gemini-2.5-flash-lite",
            Self::Other(model) => model,
        };
        f.write_str(s)
    }
}

impl FromStr for Model {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "gemini-3.6-flash" => Self::Gemini36Flash,
            "gemini-3.5-flash" => Self::Gemini35Flash,
            "gemini-3.1-pro-preview" => Self::Gemini31ProPreview,
            "gemini-3.1-flash-lite" => Self::Gemini31FlashLite,
            "gemini-3-flash-preview" => Self::Gemini3FlashPreview,
            "gemini-2.5-pro" => Self::Gemini25Pro,
            "gemini-2.5-flash" => Self::Gemini25Flash,
            "gemini-2.5-flash-lite" => Self::Gemini25FlashLite,
            other => Self::Other(other.to_string()),
        })
    }
}

// Models serialize as their Gemini model id, e.g. `gemini-3.6-flash`.
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

/// The `Step` struct represents an individual step within a chain for Gemini models. It is responsible for configuring the input parameters for the model and providing the prompt.
///
/// By creating a `Step`, you can customize the model, prompt and request options used for a
/// particular stage within an `llm-chain` workflow. This allows for granular control over the
/// text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_gemini::generate_content::{Step, Model, ChatPromptTemplate, Options};
/// let model = Model::default();
/// let prompt = ChatPromptTemplate::system_and_user("You are an assistant that speaks like Shakespeare.", "tell me a joke");
///
/// let step = Step::new(model, prompt)
///     .with_options(Options::new().with_temperature(0.7));
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
    /// Sets the system instructions for this step's prompt.
    pub fn with_system<S: Into<llm_chain::PromptTemplate>>(mut self, system: S) -> Step {
        self.prompt = self.prompt.with_system(system);
        self
    }
    /// Sets the request options for this step.
    pub fn with_options(mut self, options: Options) -> Step {
        self.options = options;
        self
    }
}

impl traits::Step for Step {
    type Output = GenerateContentRequest;
    type Error = FormatError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        let mut request = GenerateContentRequest {
            model: self.model.to_string(),
            contents: self.prompt.format(parameters)?,
            system_instruction: self
                .prompt
                .format_system(parameters)?
                .map(Content::system),
            generation_config: None,
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
                "llm-chain-gemini::generate_content::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-gemini::generate_content::ChatPromptTemplate".to_string(),
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
        let models = [
            Model::Gemini36Flash,
            Model::Gemini35Flash,
            Model::Gemini31ProPreview,
            Model::Gemini31FlashLite,
            Model::Gemini3FlashPreview,
            Model::Gemini25Pro,
            Model::Gemini25Flash,
            Model::Gemini25FlashLite,
            Model::Other("gemini-2.5-flash-preview-05-20".to_string()),
        ];
        for model in models {
            let parsed: Model = model.to_string().parse().unwrap();
            assert_eq!(parsed, model);
        }
    }

    #[test]
    fn format_builds_a_request_with_options() {
        let step = Step::new(Model::default(), [(Role::User, "hi {}")])
            .with_system("be brief")
            .with_options(
                Options::new()
                    .with_temperature(0.1)
                    .with_max_output_tokens(64),
            );
        let request = step.format(&Parameters::new_with_text("there")).unwrap();
        assert_eq!(request.model, "gemini-3.6-flash");
        assert_eq!(
            request
                .system_instruction
                .as_ref()
                .map(|system| system.text_parts()),
            Some("be brief".to_string())
        );
        assert_eq!(request.contents.len(), 1);
        assert_eq!(request.contents[0].text_parts(), "hi there");
        let config = request.generation_config.expect("config set");
        assert_eq!(config.temperature, Some(0.1));
        assert_eq!(config.max_output_tokens, Some(64));
    }

    #[test]
    fn format_without_options_sends_no_generation_config() {
        let step = Step::new(Model::Gemini31FlashLite, [(Role::User, "hi")]);
        let request = step.format(&Parameters::new()).unwrap();
        assert_eq!(request.generation_config, None);
        assert_eq!(request.system_instruction, None);
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new(Model::Gemini25Pro, [(Role::User, "hi")])
            .with_system("be brief")
            .with_options(Options::new().with_max_output_tokens(64));
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.model, Model::Gemini25Pro);
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
