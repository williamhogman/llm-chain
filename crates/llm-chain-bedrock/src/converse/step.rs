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
use super::types::{ConverseRequest, SystemContentBlock};

/// The default model, used by [`Model::default`]: Claude Sonnet 5 via the
/// `global.` cross-region inference profile.
pub const DEFAULT_MODEL: &str = models::CLAUDE_SONNET_5;

/// Well-known Bedrock model and inference-profile ids.
///
/// Bedrock hosts many model families and adds new ones continuously, so
/// [`Model`] accepts any id; these constants cover the popular choices. Most
/// current frontier models must be invoked through a cross-region inference
/// profile — an id prefixed with `us.`, `eu.`, `apac.`, or `global.` — rather
/// than the bare model id.
pub mod models {
    /// Claude Sonnet 5: the best speed/intelligence balance (global profile).
    pub const CLAUDE_SONNET_5: &str = "global.anthropic.claude-sonnet-5-v1:0";
    /// Claude Opus 5: strongest reasoning for complex work (global profile).
    pub const CLAUDE_OPUS_5: &str = "global.anthropic.claude-opus-5-v1:0";
    /// Claude Haiku 4.5: fast and cost-efficient (global profile).
    pub const CLAUDE_HAIKU_4_5: &str = "global.anthropic.claude-haiku-4-5-v1:0";
    /// Amazon Nova Premier: Amazon's most capable model (US profile).
    pub const NOVA_PREMIER: &str = "us.amazon.nova-premier-v1:0";
    /// Amazon Nova Pro: capable multimodal model.
    pub const NOVA_PRO: &str = "amazon.nova-pro-v1:0";
    /// Amazon Nova Lite: low-cost multimodal model.
    pub const NOVA_LITE: &str = "amazon.nova-lite-v1:0";
    /// Amazon Nova Micro: text-only, lowest latency and cost.
    pub const NOVA_MICRO: &str = "amazon.nova-micro-v1:0";
    /// Llama 4 Maverick: Meta's general-purpose flagship.
    pub const LLAMA4_MAVERICK: &str = "meta.llama4-maverick-17b-instruct-v1:0";
    /// Llama 4 Scout: Meta's efficient long-context model.
    pub const LLAMA4_SCOUT: &str = "meta.llama4-scout-17b-instruct-v1:0";
}

/// The `Model` struct names a Bedrock model or inference profile.
///
/// Bedrock has no fixed model lineup: any hosted model id, cross-region
/// inference-profile id (`us.`/`eu.`/`apac.`/`global.` prefix), or
/// inference-profile ARN is valid, so this is a transparent wrapper around the
/// id rather than an enum. See [`models`] for well-known ids.
///
/// # Example
///
/// ```
/// use llm_chain_bedrock::converse::{Model, models};
///
/// let default = Model::default(); // Claude Sonnet 5, global profile
/// let nova = Model::new(models::NOVA_PRO);
/// let pinned: Model = "us.meta.llama4-scout-17b-instruct-v1:0".parse().unwrap();
/// assert_eq!(nova.to_string(), "amazon.nova-pro-v1:0");
/// assert_eq!(pinned.as_str(), "us.meta.llama4-scout-17b-instruct-v1:0");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Model(String);

impl Model {
    /// Creates a model from a Bedrock model id, inference-profile id, or ARN.
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

// Models serialize as their Bedrock id, e.g. `amazon.nova-pro-v1:0`.
llm_chain::impl_model_id_serde!(Model);

/// The `Step` struct represents an individual step within a chain for Bedrock-hosted models. It is responsible for configuring the input parameters for the model and providing the prompt.
///
/// By creating a `Step`, you can customize the model, prompt and request options used for a
/// particular stage within an `llm-chain` workflow. This allows for granular control over the
/// text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_bedrock::converse::{Step, Model, ChatPromptTemplate, Options};
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
    pub fn new<M: Into<Model>, P: Into<ChatPromptTemplate>>(model: M, prompt: P) -> Step {
        Step {
            model: model.into(),
            prompt: prompt.into(),
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
    type Output = ConverseRequest;
    type Error = FormatError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        let mut request = ConverseRequest {
            model_id: self.model.to_string(),
            messages: self.prompt.format(parameters)?,
            system: self
                .prompt
                .format_system(parameters)?
                .map(|text| vec![SystemContentBlock { text }]),
            inference_config: None,
            additional_model_request_fields: None,
            tool_config: None,
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
                "llm-chain-bedrock::converse::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-bedrock::converse::ChatPromptTemplate".to_string(),
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
        for id in [
            models::CLAUDE_SONNET_5,
            models::NOVA_PRO,
            "us.meta.llama4-scout-17b-instruct-v1:0",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0",
        ] {
            let model: Model = id.parse().unwrap();
            assert_eq!(model.to_string(), id);
            assert_eq!(model.as_str(), id);
        }
        assert_eq!(Model::default().as_str(), DEFAULT_MODEL);
    }

    #[test]
    fn format_builds_a_request_with_options() {
        let step = Step::new(Model::default(), [(Role::User, "hi {}")])
            .with_system("be brief")
            .with_options(Options::new().with_temperature(0.1).with_max_tokens(64));
        let request = step.format(&Parameters::new_with_text("there")).unwrap();
        assert_eq!(request.model_id, DEFAULT_MODEL);
        assert_eq!(
            request.system,
            Some(vec![SystemContentBlock {
                text: "be brief".to_string()
            }])
        );
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].text_blocks(), "hi there");
        let config = request.inference_config.expect("config set");
        assert_eq!(config.temperature, Some(0.1));
        assert_eq!(config.max_tokens, Some(64));
    }

    #[test]
    fn format_without_options_sends_no_inference_config() {
        let step = Step::new(models::NOVA_MICRO, [(Role::User, "hi")]);
        let request = step.format(&Parameters::new()).unwrap();
        assert_eq!(request.inference_config, None);
        assert_eq!(request.system, None);
    }

    #[test]
    fn steps_accept_plain_strings_as_models() {
        let step = Step::new("mistral.mistral-large-3-v1:0", [(Role::User, "hi")]);
        let request = step.format(&Parameters::new()).unwrap();
        assert_eq!(request.model_id, "mistral.mistral-large-3-v1:0");
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new(Model::new(models::NOVA_PRO), [(Role::User, "hi")])
            .with_system("be brief")
            .with_options(Options::new().with_max_tokens(64));
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        assert!(yaml.contains("model: amazon.nova-pro-v1:0"));
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.model, Model::new(models::NOVA_PRO));
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
