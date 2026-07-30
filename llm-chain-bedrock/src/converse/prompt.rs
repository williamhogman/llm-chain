use llm_chain::{Parameters, PromptTemplate};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;
use super::types::{Message, Role};

/// A message prompt template consists of a role and a content. The role is either
/// [`Role::User`] or [`Role::Assistant`] — the Converse API keeps system
/// instructions out of the conversation (see [`ChatPromptTemplate::with_system`]).
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
    /// Formats the template into a concrete message.
    pub fn format(&self, parameters: &Parameters) -> Result<Message, FormatError> {
        Ok(Message::text(self.role, self.content.format(parameters)?))
    }
}

// From any list of things that can become messages we can create prompt templates.
impl<T: Into<MessagePromptTemplate>, L: IntoIterator<Item = T>> From<L> for ChatPromptTemplate {
    fn from(messages: L) -> Self {
        Self::new(messages.into_iter().map(|message| message.into()).collect())
    }
}

/// A conversational template for Bedrock's Converse API: optional system
/// instructions plus a list of user/assistant message templates.
///
/// # Example
///
/// ```
/// use llm_chain_bedrock::converse::{ChatPromptTemplate, Role};
///
/// let chat_template: ChatPromptTemplate = vec![
///   (Role::User, "tell me a joke"),
/// ]
/// .into();
/// let chat_template = chat_template
///     .with_system("You are an assistant that speaks like Shakespeare.");
/// ```
/// Or, for the common case:
/// ```
/// use llm_chain_bedrock::converse::ChatPromptTemplate;
/// let chat_template = ChatPromptTemplate::system_and_user(
///   "You are an assistant that speaks like Shakespeare.",
///   "tell me a joke",
/// );
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct ChatPromptTemplate {
    #[cfg_attr(
        feature = "serialization",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    system: Option<PromptTemplate>,
    messages: Vec<MessagePromptTemplate>,
}

impl ChatPromptTemplate {
    /// Creates a new chat template from a list of message templates, without
    /// system instructions.
    pub fn new(messages: Vec<MessagePromptTemplate>) -> ChatPromptTemplate {
        ChatPromptTemplate {
            system: None,
            messages,
        }
    }
    /// Sets the system instructions template.
    pub fn with_system<S: Into<PromptTemplate>>(mut self, system: S) -> ChatPromptTemplate {
        self.system = Some(system.into());
        self
    }
    /// Creates a chat template with system instructions and a single user message.
    pub fn system_and_user<S: Into<PromptTemplate>, U: Into<PromptTemplate>>(
        system: S,
        user: U,
    ) -> ChatPromptTemplate {
        ChatPromptTemplate {
            system: Some(system.into()),
            messages: vec![MessagePromptTemplate::new(Role::User, user.into())],
        }
    }
    /// Formats the system instructions, if any.
    pub fn format_system(&self, parameters: &Parameters) -> Result<Option<String>, FormatError> {
        self.system
            .as_ref()
            .map(|system| system.format(parameters).map_err(FormatError::from))
            .transpose()
    }
    /// Formats every message template into concrete messages.
    pub fn format(&self, parameters: &Parameters) -> Result<Vec<Message>, FormatError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_system_and_messages() {
        let template = ChatPromptTemplate::system_and_user("be {}", "hello {}");
        let parameters = Parameters::new_with_text("brief");
        assert_eq!(
            template.format_system(&parameters).unwrap().as_deref(),
            Some("be brief")
        );
        let messages = template.format(&parameters).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].text_blocks(), "hello brief");
    }

    #[test]
    fn templates_without_system_format_none() {
        let template: ChatPromptTemplate = vec![(Role::User, "hi")].into();
        assert_eq!(
            template.format_system(&Parameters::new()).unwrap(),
            None::<String>
        );
    }
}
