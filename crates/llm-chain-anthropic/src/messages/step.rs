#[cfg(feature = "serialization")]
use llm_chain::serialization::StorableEntity;
use llm_chain::{Parameters, traits};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;
use super::options::{DEFAULT_MAX_TOKENS, Options};
use super::prompt::ChatPromptTemplate;
use super::types::MessagesRequest;

/// The `Model` enum represents the available Claude models.
///
/// From the 4.6 generation onward, Anthropic model ids are dateless and
/// permanently pinned (e.g. `claude-opus-5` never changes what it points at);
/// earlier generations use dateless aliases that resolve to the latest dated
/// snapshot. The `Other(String)` variant lets you pin a dated snapshot (e.g.
/// `claude-sonnet-4-5-20250929`) or use a model released after this crate.
///
/// # Example
///
/// ```
/// use llm_chain_anthropic::messages::Model;
///
/// let frontier = Model::ClaudeFable5;
/// let balanced = Model::default(); // claude-sonnet-5
/// let high_volume = Model::ClaudeHaiku45;
/// let pinned: Model = "claude-sonnet-4-5-20250929".parse().unwrap();
/// assert_eq!(Model::ClaudeOpus5.to_string(), "claude-opus-5");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Model {
    // -- Claude 5 family: the current generation --
    /// Claude Fable 5: the frontier model for the hardest, long-running agentic work.
    ClaudeFable5,
    /// Claude Opus 5: strongest reasoning for complex coding and enterprise tasks.
    ClaudeOpus5,
    /// Claude Sonnet 5: the best speed/intelligence balance. A great default.
    #[default]
    ClaudeSonnet5,
    /// Claude Haiku 4.5: fast and cost-efficient for high-volume workloads.
    ClaudeHaiku45,

    // -- Previous generations (pinned ids from 4.6 onward) --
    /// Claude Opus 4.8.
    ClaudeOpus48,
    /// Claude Opus 4.7.
    ClaudeOpus47,
    /// Claude Opus 4.6.
    ClaudeOpus46,
    /// Claude Sonnet 4.6.
    ClaudeSonnet46,
    /// Claude Opus 4.5.
    ClaudeOpus45,
    /// Claude Sonnet 4.5.
    ClaudeSonnet45,

    /// Any other model id, e.g. a dated snapshot or a model released after this crate.
    Other(String),
}

// One table drives Display, FromStr, KNOWN_IDS and the id-string serde impls.
llm_chain::impl_model_id! {
    Model {
        ClaudeFable5 => "claude-fable-5",
        ClaudeOpus5 => "claude-opus-5",
        ClaudeSonnet5 => "claude-sonnet-5",
        ClaudeHaiku45 => "claude-haiku-4-5",
        ClaudeOpus48 => "claude-opus-4-8",
        ClaudeOpus47 => "claude-opus-4-7",
        ClaudeOpus46 => "claude-opus-4-6",
        ClaudeSonnet46 => "claude-sonnet-4-6",
        ClaudeOpus45 => "claude-opus-4-5",
        ClaudeSonnet45 => "claude-sonnet-4-5",
    }
    other: Other
}

/// The `Step` struct represents an individual step within a chain for Claude models. It is responsible for configuring the input parameters for the model and providing the prompt.
///
/// By creating a `Step`, you can customize the model, prompt and request options used for a
/// particular stage within an `llm-chain` workflow. This allows for granular control over the
/// text generation process, enabling you to create sophisticated multi-step chains.
///
/// # Example
///
/// ```
/// use llm_chain_anthropic::messages::{Step, Model, ChatPromptTemplate, Options};
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
    type Output = MessagesRequest;
    type Error = FormatError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        let mut request = MessagesRequest {
            model: self.model.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            system: self.prompt.format_system(parameters)?,
            messages: self.prompt.format(parameters)?,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            effort: None,
            tools: None,
            tool_choice: None,
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
                "llm-chain-anthropic::messages::Step".to_string(),
            ),
            (
                "prompt".to_string(),
                "llm-chain-anthropic::messages::ChatPromptTemplate".to_string(),
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
        for id in Model::KNOWN_IDS {
            let model: Model = id.parse().unwrap();
            assert!(!matches!(model, Model::Other(_)), "{id} parsed as Other");
            assert_eq!(model.to_string(), *id);
        }
        let pinned: Model = "claude-sonnet-4-5-20250929".parse().unwrap();
        assert_eq!(
            pinned,
            Model::Other("claude-sonnet-4-5-20250929".to_string())
        );
        assert_eq!(pinned.to_string(), "claude-sonnet-4-5-20250929");
    }

    #[test]
    fn format_builds_a_request_with_options() {
        let step = Step::new(Model::default(), [(Role::User, "hi {}")])
            .with_system("be brief")
            .with_options(Options::new().with_temperature(0.1).with_max_tokens(64));
        let request = step.format(&Parameters::new_with_text("there")).unwrap();
        assert_eq!(request.model, "claude-sonnet-5");
        assert_eq!(request.system.as_deref(), Some("be brief"));
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].content.as_text(), Some("hi there"));
        assert_eq!(request.temperature, Some(0.1));
        assert_eq!(request.max_tokens, 64);
    }

    #[test]
    fn format_defaults_max_tokens() {
        let step = Step::new(Model::ClaudeHaiku45, [(Role::User, "hi")]);
        let request = step.format(&Parameters::new()).unwrap();
        assert_eq!(request.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(request.system, None);
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new(Model::ClaudeOpus5, [(Role::User, "hi")])
            .with_system("be brief")
            .with_options(Options::new().with_max_tokens(64));
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.model, Model::ClaudeOpus5);
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
