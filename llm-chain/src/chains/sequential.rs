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
