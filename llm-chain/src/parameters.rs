use std::collections::HashMap;

/// Parameters define the parameters sent into each step. The parameters are used to fill in the prompt template, and are also filled in by the output of the previous step. Parameters have a special key, `text`, which is used as a default key for simple use cases.
///
/// Parameters also implement a few convenience conversion traits to make it easier to work with them.
///
/// # Examples
///
/// **Creating a default parameter from a string**
/// ```
/// use llm_chain::Parameters;
/// let p: Parameters = "Hello world!".into();
/// assert_eq!(p.get("text"), Some("Hello world!"));
/// ```
/// **Creating a list of parameters from a list of pairs**
/// ```
/// use llm_chain::Parameters;
/// let p: Parameters = vec![("text", "Hello world!"), ("name", "John Doe")].into();
/// assert_eq!(p.get("text"), Some("Hello world!"));
/// assert_eq!(p.get("name"), Some("John Doe"));
/// ```
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Parameters(HashMap<String, String>);

pub(crate) const TEXT_KEY: &str = "text";

impl Parameters {
    /// Creates a new empty set of parameters.
    pub fn new() -> Parameters {
        Parameters(HashMap::new())
    }
    /// Creates a new set of parameters with a single key, `text`, set to the given value.
    pub fn new_with_text<T: Into<String>>(text: T) -> Parameters {
        let mut map = HashMap::new();
        map.insert(TEXT_KEY.to_string(), text.into());
        Parameters(map)
    }
    /// Copies the parameters and adds a new key-value pair.
    #[must_use]
    pub fn with<K: Into<String>, V: Into<String>>(&self, key: K, value: V) -> Parameters {
        let mut copy = self.clone();
        copy.0.insert(key.into(), value.into());
        copy
    }
    /// Copies the parameters and adds a new key-value pair with the key `text`, which is the default key.
    #[must_use]
    pub fn with_text<K: Into<String>>(&self, text: K) -> Parameters {
        self.with(TEXT_KEY, text)
    }
    /// Combines two sets of parameters, returning a new set of parameters with all the keys from both sets.
    #[must_use]
    pub fn combine(&self, other: &Parameters) -> Parameters {
        let mut copy = self.clone();
        copy.0.extend(other.0.clone());
        copy
    }
    /// Inserts a key-value pair in place.
    pub fn insert<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.0.insert(key.into(), value.into());
    }
    /// Returns the value of the given key, or `None` if the key does not exist.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
    /// Returns the value of the default `text` key, or `None` if it is not set.
    pub fn get_text(&self) -> Option<&str> {
        self.get(TEXT_KEY)
    }
    /// Returns `true` if there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Returns the number of parameters.
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Iterates over the key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

impl From<String> for Parameters {
    fn from(text: String) -> Self {
        Parameters::new_with_text(text)
    }
}

impl From<&str> for Parameters {
    fn from(text: &str) -> Self {
        Parameters::new_with_text(text)
    }
}

impl From<HashMap<String, String>> for Parameters {
    fn from(map: HashMap<String, String>) -> Self {
        Parameters(map)
    }
}

impl From<Vec<(String, String)>> for Parameters {
    fn from(data: Vec<(String, String)>) -> Self {
        let map: HashMap<String, String> = data.into_iter().collect();
        Parameters(map)
    }
}

impl From<Vec<(&str, &str)>> for Parameters {
    fn from(data: Vec<(&str, &str)>) -> Self {
        let map: HashMap<String, String> = data
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Parameters(map)
    }
}

impl FromIterator<(String, String)> for Parameters {
    fn from_iter<T: IntoIterator<Item = (String, String)>>(iter: T) -> Self {
        Parameters(iter.into_iter().collect())
    }
}
