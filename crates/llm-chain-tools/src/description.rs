use serde::{Serialize, Serializer, ser::SerializeMap};

#[derive(Clone)]
/// A description of a parameter for a tool.
pub struct FormatPart {
    key: String,
    purpose: String,
}

impl FormatPart {
    pub fn new(key: &str, purpose: &str) -> Self {
        FormatPart {
            key: key.to_string(),
            purpose: purpose.to_string(),
        }
    }

    /// The parameter name.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// What the parameter is for.
    pub fn purpose(&self) -> &str {
        &self.purpose
    }
}

impl<K: Into<String>, P: Into<String>> From<(K, P)> for FormatPart {
    fn from((k, p): (K, P)) -> Self {
        FormatPart::new(&k.into(), &p.into())
    }
}

pub struct Format {
    parts: Vec<FormatPart>,
}

impl Format {
    pub fn new(parts: Vec<FormatPart>) -> Self {
        Format { parts }
    }

    /// The parameters making up this format.
    pub fn parts(&self) -> &[FormatPart] {
        &self.parts
    }
}

impl<T: AsRef<[FormatPart]>> From<T> for Format {
    fn from(parts: T) -> Self {
        Format::new(parts.as_ref().to_vec())
    }
}

impl Serialize for Format {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let n = self.parts.len();
        let mut map = serializer.serialize_map(Some(n))?;
        for part in &self.parts {
            map.serialize_entry(&part.key, &part.purpose)?;
        }
        map.end()
    }
}

pub trait Describe {
    fn describe() -> Format;
}

#[derive(Serialize)]
/// A description of a tool, used to prompt the model
pub struct ToolDescription {
    pub(crate) name: String,
    description: String,
    description_context: String,
    input_format: Format,
    #[serde(skip)]
    #[allow(dead_code)]
    output_format: Format,
}

impl ToolDescription {
    pub fn new(
        name: &str,
        description: &str,
        description_context: &str,
        input_format: Format,
        output_format: Format,
    ) -> Self {
        ToolDescription {
            name: name.to_string(),
            description: description.to_string(),
            description_context: description_context.to_string(),
            input_format,
            output_format,
        }
    }

    /// The name the tool is invoked by.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What the tool does.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// When the tool should be used.
    pub fn description_context(&self) -> &str {
        &self.description_context
    }

    /// The description and its usage context joined into one sentence,
    /// suitable for a native tool definition.
    pub fn full_description(&self) -> String {
        if self.description_context.is_empty() {
            self.description.clone()
        } else {
            format!("{} {}", self.description, self.description_context)
        }
    }

    /// The tool's input format as a [JSON Schema](https://json-schema.org/)
    /// object, suitable for a native tool definition.
    ///
    /// Every parameter maps to a required string property, matching how tools
    /// in this crate describe their inputs. The schema deliberately omits
    /// `additionalProperties` so it is accepted verbatim by every provider
    /// (some reject unknown schema keywords).
    pub fn input_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for part in self.input_format.parts() {
            properties.insert(
                part.key().to_string(),
                serde_json::json!({"type": "string", "description": part.purpose()}),
            );
            required.push(serde_json::Value::String(part.key().to_string()));
        }
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }
}
