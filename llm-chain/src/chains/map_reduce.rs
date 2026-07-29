//! Map-reduce chains apply a `map` step to every document in parallel, combine
//! the outputs, and then run a `reduce` step over the combined result.

use futures::future::join_all;
#[cfg(feature = "serialization")]
use serde::{
    Deserialize, Serialize,
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
};

#[cfg(feature = "serialization")]
use crate::serialization::StorableEntity;
use crate::{
    Parameters,
    chains::ChainError,
    traits::{Executor, Step},
};

/// A map-reduce chain: applies `map` to each document, combines the results and runs `reduce` over the combination.
pub struct Chain<S: Step> {
    map: S,
    reduce: S,
}

impl<S: Step> Chain<S> {
    /// Creates a new map-reduce chain from a map step and a reduce step.
    pub fn new(map: S, reduce: S) -> Chain<S> {
        Chain { map, reduce }
    }

    /// Runs the chain over the given documents.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::Empty`] if `documents` is empty, and
    /// [`ChainError::Format`] or [`ChainError::Execute`] if a step fails.
    pub async fn run<L: Executor<Step = S>>(
        &self,
        documents: Vec<Parameters>,
        base_parameters: Parameters,
        executor: &L,
    ) -> Result<L::Output, ChainError<S::Error, L::Error>> {
        if documents.is_empty() {
            return Err(ChainError::Empty);
        }
        // TODO: We need to do this recursively for really big documents
        let formatted_documents = documents
            .iter()
            .map(|doc| self.map.format(&base_parameters.combine(doc)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ChainError::Format)?;

        let mapped_documents = join_all(
            formatted_documents
                .into_iter()
                .map(|formatted| executor.execute(formatted)),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(ChainError::Execute)?;

        let combined_output = mapped_documents
            .into_iter()
            .reduce(|a, b| L::combine_outputs(&a, &b))
            .ok_or(ChainError::Empty)?;

        let combined_parameters = L::apply_output_to_parameters(base_parameters, &combined_output);

        let formatted = self
            .reduce
            .format(&combined_parameters)
            .map_err(ChainError::Format)?;
        executor
            .execute(formatted)
            .await
            .map_err(ChainError::Execute)
    }
}

#[cfg(feature = "serialization")]
impl<S: Step + Serialize> Serialize for Chain<S> {
    fn serialize<SER>(&self, serializer: SER) -> Result<SER::Ok, SER::Error>
    where
        SER: serde::Serializer,
    {
        let mut strct = serializer.serialize_struct("Chain", 2)?;
        strct.serialize_field("map", &self.map)?;
        strct.serialize_field("reduce", &self.reduce)?;
        strct.end()
    }
}

#[cfg(feature = "serialization")]
impl<'de, S: Step + Deserialize<'de>> Deserialize<'de> for Chain<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ChainVisitor<S>(std::marker::PhantomData<S>);

        impl<'de, S: Step + Deserialize<'de>> Visitor<'de> for ChainVisitor<S> {
            type Value = Chain<S>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object with fields `map` and `reduce`")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut map_value: Option<S> = None;
                let mut reduce_value: Option<S> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "map" => {
                            if map_value.is_some() {
                                return Err(serde::de::Error::duplicate_field("map"));
                            }
                            map_value = Some(map.next_value()?);
                        }
                        "reduce" => {
                            if reduce_value.is_some() {
                                return Err(serde::de::Error::duplicate_field("reduce"));
                            }
                            reduce_value = Some(map.next_value()?);
                        }
                        _ => (),
                    }
                }

                let map = map_value.ok_or_else(|| serde::de::Error::missing_field("map"))?;
                let reduce =
                    reduce_value.ok_or_else(|| serde::de::Error::missing_field("reduce"))?;
                Ok(Chain { map, reduce })
            }
        }

        deserializer.deserialize_struct(
            "Chain",
            &["map", "reduce"],
            ChainVisitor(std::marker::PhantomData),
        )
    }
}

#[cfg(feature = "serialization")]
impl<S> StorableEntity for Chain<S>
where
    S: Step + StorableEntity,
{
    fn get_metadata() -> Vec<(String, String)> {
        let mut base = vec![(
            "chain-type".to_string(),
            "llm-chain::chains::map_reduce::Chain".to_string(),
        )];
        base.append(&mut S::get_metadata());
        base
    }
}
