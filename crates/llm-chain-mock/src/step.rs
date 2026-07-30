#[cfg(feature = "serialization")]
use llm_chain::serialization::StorableEntity;
use llm_chain::{Parameters, PromptTemplate, PromptTemplateError, traits};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

/// A single mock step: formats a [`PromptTemplate`] with the current
/// [`Parameters`] and hands the resulting text to the [`Executor`](crate::Executor).
///
/// # Example
///
/// ```
/// use llm_chain::{Parameters, traits::Step as _};
/// use llm_chain_mock::Step;
///
/// let step = Step::new("Hello {name}!");
/// let prompt = step.format(&vec![("name", "world")].into()).unwrap();
/// assert_eq!(prompt, "Hello world!");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Step {
    prompt: PromptTemplate,
}

impl Step {
    /// Creates a new step from anything that converts into a [`PromptTemplate`].
    pub fn new(prompt: impl Into<PromptTemplate>) -> Self {
        Self {
            prompt: prompt.into(),
        }
    }

    /// The step's prompt template.
    pub fn prompt(&self) -> &PromptTemplate {
        &self.prompt
    }
}

impl traits::Step for Step {
    type Output = String;
    type Error = PromptTemplateError;
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        self.prompt.format(parameters)
    }
}

#[cfg(feature = "serialization")]
impl StorableEntity for Step {
    fn get_metadata() -> Vec<(String, String)> {
        vec![("step-type".to_string(), "llm-chain-mock::Step".to_string())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_chain::traits::Step as _;

    #[test]
    fn format_fills_placeholders() {
        let step = Step::new("{a} and {b}");
        let out = step
            .format(&vec![("a", "one"), ("b", "two")].into())
            .unwrap();
        assert_eq!(out, "one and two");
    }

    #[test]
    fn format_surfaces_missing_parameters() {
        let step = Step::new("{missing}");
        assert!(step.format(&Parameters::new()).is_err());
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn step_round_trips_through_yaml() {
        let step = Step::new("Hello {name}!");
        let yaml = serde_yaml_ng::to_string(&step).unwrap();
        let parsed: Step = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, step);
    }
}
