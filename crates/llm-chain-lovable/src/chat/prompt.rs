use llm_chain::{Parameters, PromptTemplate};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::error::FormatError;
use super::types::{Message, Role};

/// A message prompt template consists of a role and a content. The role is one of
/// [`Role::System`], [`Role::User`] or [`Role::Assistant`] — the gateway takes
/// system instructions inline in the message list, OpenAI-style.
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
        Ok(Message::new(self.role, self.content.format(parameters)?))
    }
}

// From any list of things that can become messages we can create prompt templates.
impl<T: Into<MessagePromptTemplate>, L: IntoIterator<Item = T>> From<L> for ChatPromptTemplate {
    fn from(messages: L) -> Self {
        Self::new(messages.into_iter().map(|message| message.into()).collect())
    }
}

/// A conversational template for the gateway's chat completions API: a list
/// of system/user/assistant message templates.
///
/// # Example
///
/// ```
/// use llm_chain_lovable::chat::{ChatPromptTemplate, Role};
///
/// let chat_template: ChatPromptTemplate = vec![
///   (Role::System, "You are an assistant that speaks like Shakespeare."),
///   (Role::User, "tell me a joke"),
/// ]
/// .into();
/// ```
/// Or, for the common case:
/// ```
/// use llm_chain_lovable::chat::ChatPromptTemplate;
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
    fn formats_every_role() {
        let template: ChatPromptTemplate = vec![
            (Role::System, "system {}"),
            (Role::User, "user {}"),
            (Role::Assistant, "assistant {}"),
        ]
        .into();
        let messages = template.format(&Parameters::new_with_text("x")).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content.as_deref(), Some("system x"));
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].content.as_deref(), Some("assistant x"));
    }

    #[test]
    fn system_and_user_builds_two_messages() {
        let template = ChatPromptTemplate::system_and_user("be {}", "hello {}");
        let messages = template
            .format(&Parameters::new_with_text("brief"))
            .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content.as_deref(), Some("be brief"));
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content.as_deref(), Some("hello brief"));
    }
}
