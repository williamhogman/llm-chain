//! Macros for model-id types in driver crates.
//!
//! Every driver crate exposes a `Model` type that maps between Rust values and
//! provider model-id strings (`gpt-5.6-terra`, `claude-sonnet-5`, …), and that
//! serializes as that plain string in YAML chains. The macros in this module
//! generate those conversions from a single id table, so a driver adds a new
//! model by adding exactly one line.
//!
//! - [`impl_model_id!`](crate::impl_model_id) — for enums with a catch-all
//!   string variant: generates `Display`, `FromStr`, a `KNOWN_IDS` constant and
//!   the serde impls.
//! - [`impl_model_id_serde!`](crate::impl_model_id_serde) — for types that
//!   already have `Display`/`FromStr` (e.g. newtypes around `String`):
//!   generates only the serde impls.

/// Implements `Display`, `FromStr`, `KNOWN_IDS` and string-based serde for a
/// model enum with a catch-all variant, from a single variant-to-id table.
///
/// The enum must have one unit variant per known model plus a tuple variant
/// holding a `String` for everything else. `FromStr` is infallible: unknown
/// ids land in the catch-all variant, so newly released models are always
/// usable.
///
/// The serde impls are gated on a `serialization` cargo feature in the calling
/// crate (the convention for every llm-chain driver) and serialize the model
/// as its id string.
///
/// # Example
///
/// ```
/// #[derive(Debug, Clone, PartialEq, Eq, Default)]
/// pub enum Model {
///     #[default]
///     Balanced,
///     Flagship,
///     Other(String),
/// }
///
/// llm_chain::impl_model_id! {
///     Model {
///         Balanced => "provider-balanced",
///         Flagship => "provider-flagship",
///     }
///     other: Other
/// }
///
/// assert_eq!(Model::Flagship.to_string(), "provider-flagship");
/// assert_eq!("provider-balanced".parse::<Model>().unwrap(), Model::Balanced);
/// assert_eq!(
///     "future-model".parse::<Model>().unwrap(),
///     Model::Other("future-model".into())
/// );
/// assert_eq!(Model::KNOWN_IDS.len(), 2);
/// ```
#[macro_export]
macro_rules! impl_model_id {
    (
        $model:ident {
            $($variant:ident => $id:literal),+ $(,)?
        }
        other: $other:ident
    ) => {
        impl $model {
            /// Every model id this crate knows by name — i.e. every variant
            /// except the catch-all string variant.
            pub const KNOWN_IDS: &'static [&'static str] = &[$($id),+];
        }

        impl ::core::fmt::Display for $model {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str($id),)+
                    Self::$other(model) => f.write_str(model),
                }
            }
        }

        impl ::core::str::FromStr for $model {
            type Err = ::core::convert::Infallible;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ok(match s {
                    $($id => Self::$variant,)+
                    other => Self::$other(other.to_string()),
                })
            }
        }

        $crate::impl_model_id_serde!($model);
    };
}

/// Implements string-based serde for a model type via its `Display` and
/// `FromStr` impls.
///
/// Use this for model types that are transparent wrappers around a string
/// (Ollama names, Bedrock ids); [`impl_model_id!`](crate::impl_model_id)
/// invokes it automatically for enum models. The generated impls are gated on
/// a `serialization` cargo feature in the calling crate, which must depend on
/// `serde` (at least when that feature is enabled).
#[macro_export]
macro_rules! impl_model_id_serde {
    ($model:ty) => {
        #[cfg(feature = "serialization")]
        impl ::serde::Serialize for $model {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.to_string())
            }
        }

        #[cfg(feature = "serialization")]
        impl<'de> ::serde::Deserialize<'de> for $model {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                let id = <String as ::serde::Deserialize>::deserialize(deserializer)?;
                id.parse::<$model>().map_err(::serde::de::Error::custom)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    enum Model {
        #[default]
        Balanced,
        Flagship,
        Other(String),
    }

    crate::impl_model_id! {
        Model {
            Balanced => "test-balanced",
            Flagship => "test-flagship",
        }
        other: Other
    }

    #[test]
    fn known_ids_round_trip() {
        for id in Model::KNOWN_IDS {
            let model: Model = id.parse().unwrap();
            assert!(!matches!(model, Model::Other(_)), "{id} parsed as Other");
            assert_eq!(model.to_string(), *id);
        }
    }

    #[test]
    fn unknown_ids_fall_back_to_the_catch_all() {
        let model: Model = "test-next-gen".parse().unwrap();
        assert_eq!(model, Model::Other("test-next-gen".to_string()));
        assert_eq!(model.to_string(), "test-next-gen");
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn models_serialize_as_plain_id_strings() {
        let yaml = serde_yaml_ng::to_string(&Model::Flagship).unwrap();
        assert_eq!(yaml.trim(), "test-flagship");
        let parsed: Model = serde_yaml_ng::from_str("test-balanced").unwrap();
        assert_eq!(parsed, Model::Balanced);
        let parsed: Model = serde_yaml_ng::from_str("something-new").unwrap();
        assert_eq!(parsed, Model::Other("something-new".to_string()));
    }
}
