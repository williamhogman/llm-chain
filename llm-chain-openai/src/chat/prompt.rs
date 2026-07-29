use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, Role,
};
use llm_chain::{Parameters, PromptTemplate};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;

/// A message prompt template consists of a role and a content. The role is either `User`, `System` or `Assistant`, and the content is a prompt template.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct MessagePromptTemplate {
    role: Role,
    content: PromptTemplate,
}

impl<T: Into<PromptTemplate>> From<(Role, T)> for MessagePromptTemplate {
    fn from((role, content): (Role, T)) -> Self {
        let content: PromptTemplate = content.into();
        Self { role, content }
    }
}

impl MessagePromptTemplate {
    /// Creates a new message template with the given role and content template.
    pub fn new(role: Role, content: PromptTemplate) -> MessagePromptTemplate {
        MessagePromptTemplate { role, content }
    }
    /// Formats the template into a concrete OpenAI chat message.
    pub fn format(
        &self,
        parameters: &Parameters,
    ) -> Result<ChatCompletionRequestMessage, FormatError> {
        let content = self.content.format(parameters)?;
        let message = match self.role {
            Role::System => ChatCompletionRequestSystemMessage {
                content: content.into(),
                ..Default::default()
            }
            .into(),
            Role::User => ChatCompletionRequestUserMessage {
                content: content.into(),
                ..Default::default()
            }
            .into(),
            Role::Assistant => ChatCompletionRequestAssistantMessage {
                content: Some(content.into()),
                ..Default::default()
            }
            .into(),
            role => return Err(FormatError::UnsupportedRole(role)),
        };
        Ok(message)
    }
}

// From any list of things that can become messages we can create prompt templates.
impl<T: Into<MessagePromptTemplate>, L: IntoIterator<Item = T>> From<L> for ChatPromptTemplate {
    fn from(messages: L) -> Self {
        Self::new(messages.into_iter().map(|message| message.into()).collect())
    }
}

/// The `ChatPromptTemplate` struct represents a conversational template for generating prompts with LLMs. It consists of a list of messages that form the structure of the conversation.
///
/// Typically, a `ChatPromptTemplate` starts with a system message to set the context, followed by user messages and potential assistant messages. This design makes it easy to create dynamic and engaging conversational prompts for chat models.
///
/// # Example
///
/// ```
/// use llm_chain_openai::chatgpt::{ChatPromptTemplate, MessagePromptTemplate, Role};
///
/// let system_msg = MessagePromptTemplate::new(Role::System, "You are an assistant that speaks like Shakespeare.".into());
/// let user_msg = MessagePromptTemplate::new(Role::User, "tell me a joke".into());
///
/// let chat_template = ChatPromptTemplate::new(vec![system_msg, user_msg]);
/// ```
/// Or simply
/// ```
/// use llm_chain_openai::chatgpt::{ChatPromptTemplate, Role};
/// let chat_template: ChatPromptTemplate = vec![
///   (Role::System, "You are an assistant that speaks like Shakespeare."),
///   (Role::User, "tell me a joke"),
/// ].into();
/// ```
/// And for the truly lazy
/// ```
/// use llm_chain_openai::chatgpt::ChatPromptTemplate;
/// let chat_template = ChatPromptTemplate::system_and_user(
///   "You are an assistant that speaks like Shakespeare.",
///   "tell me a joke",
/// );
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct ChatPromptTemplate {
    messages: Vec<MessagePromptTemplate>,
}

impl ChatPromptTemplate {
    /// Creates a new chat template from a list of message templates.
    pub fn new(messages: Vec<MessagePromptTemplate>) -> ChatPromptTemplate {
        ChatPromptTemplate { messages }
    }
    /// Creates a chat template with a system message followed by a user message.
    pub fn system_and_user<S: Into<PromptTemplate>, U: Into<PromptTemplate>>(
        system: S,
        user: U,
    ) -> ChatPromptTemplate {
        ChatPromptTemplate {
            messages: vec![
                MessagePromptTemplate::new(Role::System, system.into()),
                MessagePromptTemplate::new(Role::User, user.into()),
            ],
        }
    }
    /// Formats every message in the template into concrete OpenAI chat messages.
    pub fn format(
        &self,
        parameters: &Parameters,
    ) -> Result<Vec<ChatCompletionRequestMessage>, FormatError> {
        self.messages
            .iter()
            .map(|message| message.format(parameters))
            .collect()
    }

    /// Appends a message template to the conversation.
    pub fn add<T: Into<MessagePromptTemplate>>(&mut self, message: T) {
        self.messages.push(message.into());
    }
}
