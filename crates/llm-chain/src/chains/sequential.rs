//! Sequential chains run their steps one after another, feeding each step's
//! output into the next step's parameters.

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serialization")]
use crate::serialization::StorableEntity;
use crate::{
    Parameters,
    chains::ChainError,
    traits::{Executor, Step},
};

/// A sequential chain is a chain where each step is executed in order, with the output of the previous being available to the next.
pub struct Chain<S: Step> {
    steps: Vec<S>,
}

impl<S: Step> Chain<S> {
    /// Creates a new chain from the given steps.
    pub fn new(steps: Vec<S>) -> Chain<S> {
        Chain { steps }
    }
    /// Creates a chain that consists of a single step.
    pub fn of_one(step: S) -> Chain<S> {
        Chain { steps: vec![step] }
    }
    /// Appends a step to the end of the chain.
    pub fn push(&mut self, step: S) {
        self.steps.push(step);
    }
    /// Returns the number of steps in the chain.
    pub fn len(&self) -> usize {
        self.steps.len()
    }
    /// Returns `true` if the chain has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
    /// The steps of the chain, in execution order.
    pub fn steps(&self) -> &[S] {
        &self.steps
    }

    /// Runs the chain, returning the output of the final step.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::Empty`] if the chain has no steps, and
    /// [`ChainError::Format`] or [`ChainError::Execute`] if a step fails.
    pub async fn run<L: Executor<Step = S>>(
        &self,
        parameters: Parameters,
        executor: &L,
    ) -> Result<L::Output, ChainError<S::Error, L::Error>> {
        let mut current_params = parameters;
        let mut output: Option<L::Output> = None;
        for step in self.steps.iter() {
            let formatted = step.format(&current_params).map_err(ChainError::Format)?;
            let res = executor
                .execute(formatted)
                .await
                .map_err(ChainError::Execute)?;
            current_params = L::apply_output_to_parameters(current_params, &res);
            output = Some(res);
        }
        output.ok_or(ChainError::Empty)
    }
}

/// Collects steps into a chain, in iteration order.
impl<S: Step> FromIterator<S> for Chain<S> {
    fn from_iter<T: IntoIterator<Item = S>>(iter: T) -> Self {
        Chain {
            steps: iter.into_iter().collect(),
        }
    }
}

/// Appends steps to the end of the chain.
impl<S: Step> Extend<S> for Chain<S> {
    fn extend<T: IntoIterator<Item = S>>(&mut self, iter: T) {
        self.steps.extend(iter);
    }
}

/// Consumes the chain, yielding its steps in execution order.
impl<S: Step> IntoIterator for Chain<S> {
    type Item = S;
    type IntoIter = std::vec::IntoIter<S>;
    fn into_iter(self) -> Self::IntoIter {
        self.steps.into_iter()
    }
}

#[cfg(feature = "serialization")]
impl<S: Step + Serialize> Serialize for Chain<S> {
    fn serialize<SER>(&self, serializer: SER) -> Result<SER::Ok, SER::Error>
    where
        SER: serde::Serializer,
    {
        Serialize::serialize(&self.steps, serializer)
    }
}

#[cfg(feature = "serialization")]
impl<'de, S: Step + Deserialize<'de>> Deserialize<'de> for Chain<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer).map(|steps| Chain { steps })
    }
}

#[cfg(feature = "serialization")]
impl<S: Step + StorableEntity> StorableEntity for Chain<S> {
    fn get_metadata() -> Vec<(String, String)> {
        let mut base = vec![(
            "chain-type".to_string(),
            "llm-chain::chains::sequential::Chain".to_string(),
        )];
        base.append(&mut S::get_metadata());
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoStep(&'static str);

    impl Step for EchoStep {
        type Output = String;
        type Error = std::convert::Infallible;
        fn format(&self, _parameters: &Parameters) -> Result<String, Self::Error> {
            Ok(self.0.to_string())
        }
    }

    #[test]
    fn chains_compose_from_iterators() {
        let mut chain: Chain<EchoStep> = [EchoStep("a"), EchoStep("b")].into_iter().collect();
        assert_eq!(chain.len(), 2);
        chain.push(EchoStep("c"));
        chain.extend([EchoStep("d")]);
        assert_eq!(chain.len(), 4);
        assert!(!chain.is_empty());
        let formatted: Vec<String> = chain
            .into_iter()
            .map(|step| step.format(&Parameters::new()).unwrap())
            .collect();
        assert_eq!(formatted, ["a", "b", "c", "d"]);
    }

    #[test]
    fn empty_chains_report_empty() {
        let chain: Chain<EchoStep> = Chain::new(vec![]);
        assert!(chain.is_empty());
        assert_eq!(chain.steps().len(), 0);
        assert_eq!(chain.len(), 0);
    }
}
