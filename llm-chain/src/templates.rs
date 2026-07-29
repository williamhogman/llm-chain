//! Prompt templates: strings with `{}` and `{name}` placeholders that are filled
//! in from [`Parameters`] before being sent to an LLM.

use crate::Parameters;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An error that occurred while formatting a [`PromptTemplate`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PromptTemplateError {
    /// The template referenced a parameter that was not supplied.
    #[error("missing parameter: `{0}`")]
    MissingParameter(String),
    /// The template contained unbalanced or unclosed braces.
    ///
    /// Use `{{` and `}}` to include literal braces in a template.
    #[error("malformed template: {0}")]
    Malformed(String),
}

fn apply_formatting(
    template: &str,
    parameters: &Parameters,
) -> Result<String, PromptTemplateError> {
    let mut output = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    output.push('{');
                    continue;
                }
                let mut key = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(k) => key.push(k),
                        None => {
                            return Err(PromptTemplateError::Malformed(
                                "unclosed `{` placeholder".to_string(),
                            ));
                        }
                    }
                }
                let key = key.trim();
                let key = if key.is_empty() {
                    crate::parameters::TEXT_KEY
                } else {
                    key
                };
                let value = parameters
                    .get(key)
                    .ok_or_else(|| PromptTemplateError::MissingParameter(key.to_string()))?;
                output.push_str(value);
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    output.push('}');
                } else {
                    return Err(PromptTemplateError::Malformed(
                        "unmatched `}` — use `}}` for a literal brace".to_string(),
                    ));
                }
            }
            _ => output.push(c),
        }
    }
    Ok(output)
}

/// A template for a prompt. This is a string that can be formatted with a set of parameters.
///
/// Placeholders use curly braces: `{}` refers to the default `text` parameter and
/// `{name}` refers to a named parameter. Literal braces are written `{{` and `}}`.
///
/// # Examples
/// **Using the default key**
/// ```
/// use llm_chain::{PromptTemplate, Parameters};
/// let template: PromptTemplate = "Hello {}!".into();
/// let parameters: Parameters = "World".into();
/// assert_eq!(template.format(&parameters).unwrap(), "Hello World!");
/// ```
/// **Using a custom key**
/// ```
/// use llm_chain::{PromptTemplate, Parameters};
/// let template: PromptTemplate = "Hello {name}!".into();
/// let parameters: Parameters = vec![("name", "World")].into();
/// assert_eq!(template.format(&parameters).unwrap(), "Hello World!");
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    /// Create a new prompt template from a string.
    pub fn new(template: String) -> PromptTemplate {
        PromptTemplate { template }
    }
    /// Format the template with the given parameters.
    pub fn format(&self, parameters: &Parameters) -> Result<String, PromptTemplateError> {
        apply_formatting(&self.template, parameters)
    }
}

impl<T: Into<String>> From<T> for PromptTemplate {
    fn from(template: T) -> Self {
        Self::new(template.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{Parameters, PromptTemplate, PromptTemplateError, apply_formatting};

    #[test]
    fn test_apply_formatting() {
        let template = "Hello {name}!";
        let parameters = vec![("name", "World")].into();
        assert_eq!(
            apply_formatting(template, &parameters).unwrap(),
            "Hello World!".to_string()
        );
    }

    #[test]
    fn test_prompt_template_format() {
        let template: PromptTemplate = "Hello {name}!".into();
        let parameters = vec![("name", "World")].into();
        assert_eq!(template.format(&parameters).unwrap(), "Hello World!");
    }

    #[test]
    fn test_prompt_template_format_with_default_key() {
        let template: PromptTemplate = "Hello {}!".into();
        let parameters: Parameters = "World".into();
        assert_eq!(template.format(&parameters).unwrap(), "Hello World!");
    }

    #[test]
    fn test_prompt_template_format_with_multiple_keys() {
        let template: PromptTemplate = "Hello {name}, you are {age} years old.".into();
        let parameters: Parameters = vec![("name", "John"), ("age", "30")].into();
        assert_eq!(
            template.format(&parameters).unwrap(),
            "Hello John, you are 30 years old."
        );
    }

    #[test]
    fn test_escaped_braces() {
        let template: PromptTemplate = "{{literal}} and {name}".into();
        let parameters: Parameters = vec![("name", "value")].into();
        assert_eq!(template.format(&parameters).unwrap(), "{literal} and value");
    }

    #[test]
    fn test_missing_parameter() {
        let template: PromptTemplate = "Hello {name}!".into();
        let parameters = Parameters::new();
        assert_eq!(
            template.format(&parameters),
            Err(PromptTemplateError::MissingParameter("name".to_string()))
        );
    }

    #[test]
    fn test_unclosed_brace() {
        let template: PromptTemplate = "Hello {name".into();
        let parameters: Parameters = vec![("name", "World")].into();
        assert!(matches!(
            template.format(&parameters),
            Err(PromptTemplateError::Malformed(_))
        ));
    }

    #[test]
    fn test_whitespace_in_key() {
        let template: PromptTemplate = "Hello { name }!".into();
        let parameters: Parameters = vec![("name", "World")].into();
        assert_eq!(template.format(&parameters).unwrap(), "Hello World!");
    }
}
