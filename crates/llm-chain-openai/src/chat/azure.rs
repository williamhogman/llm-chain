//! Azure OpenAI (Azure AI Foundry) support via the OpenAI-compatible **v1** surface.
//!
//! Azure's v1 API (`https://{resource}.openai.azure.com/openai/v1/…`) speaks the
//! same wire format as `api.openai.com`: the model (deployment) name goes in the
//! request body and no `api-version` query parameter is needed. What differs is
//! authentication — Azure API keys travel in an `api-key` header, and
//! `Authorization: Bearer` is reserved for Microsoft Entra ID tokens.
//! [`AzureV1Config`] captures exactly that contract for the `async_openai` client.

use async_openai::config::Config;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use secrecy::{ExposeSecret, SecretString};

/// The header carrying an Azure OpenAI API key.
pub const AZURE_API_KEY_HEADER: &str = "api-key";
/// The v1 path prefix appended to an Azure resource endpoint.
pub const AZURE_V1_PATH: &str = "/openai/v1";

/// How requests to Azure authenticate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AzureAuth {
    /// An Azure API key, sent in the `api-key` header.
    ApiKey,
    /// A Microsoft Entra ID access token, sent as `Authorization: Bearer`.
    EntraToken,
}

/// An `async_openai` client configuration for Azure OpenAI's v1 surface.
///
/// Use it through the [`AzureExecutor`](super::AzureExecutor) constructors, or
/// directly with [`async_openai::Client::with_config`] for anything custom.
///
/// # Examples
///
/// ```
/// use llm_chain_openai::chat::AzureV1Config;
///
/// // A bare resource name expands to https://{resource}.openai.azure.com/openai/v1
/// let config = AzureV1Config::new("my-resource", "azure-key");
/// // Full endpoints work too:
/// let config = AzureV1Config::new("https://my-resource.openai.azure.com", "azure-key");
/// // Production deployments authenticate with Microsoft Entra ID tokens:
/// let config = AzureV1Config::with_entra_token("my-resource", "eyJ...");
/// ```
#[derive(Clone)]
pub struct AzureV1Config {
    api_base: String,
    api_key: SecretString,
    auth: AzureAuth,
}

impl AzureV1Config {
    /// Creates a configuration authenticating with an Azure API key.
    ///
    /// The endpoint may be a bare resource name (`my-resource`), a resource
    /// endpoint (`https://my-resource.openai.azure.com`), or a full v1 base
    /// URL; the `/openai/v1` suffix is appended when missing.
    pub fn new(endpoint: impl AsRef<str>, api_key: impl Into<String>) -> Self {
        Self {
            api_base: azure_v1_api_base(endpoint.as_ref()),
            api_key: SecretString::from(api_key.into()),
            auth: AzureAuth::ApiKey,
        }
    }

    /// Creates a configuration authenticating with a Microsoft Entra ID access
    /// token (the recommended production mechanism).
    ///
    /// Get a token for the `https://cognitiveservices.azure.com/.default`
    /// scope, e.g. via `DefaultAzureCredential` or
    /// `az account get-access-token`. Tokens are short-lived: mint a fresh
    /// configuration when yours expires.
    pub fn with_entra_token(endpoint: impl AsRef<str>, access_token: impl Into<String>) -> Self {
        Self {
            api_base: azure_v1_api_base(endpoint.as_ref()),
            api_key: SecretString::from(access_token.into()),
            auth: AzureAuth::EntraToken,
        }
    }
}

// Never derive Debug: it would print the credentials.
impl std::fmt::Debug for AzureV1Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureV1Config")
            .field("api_base", &self.api_base)
            .field(
                match self.auth {
                    AzureAuth::ApiKey => "api_key",
                    AzureAuth::EntraToken => "access_token",
                },
                &"[REDACTED]",
            )
            .finish()
    }
}

impl Config for AzureV1Config {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        let mut value = match self.auth {
            AzureAuth::ApiKey => HeaderValue::from_str(self.api_key.expose_secret()),
            AzureAuth::EntraToken => {
                HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
            }
        }
        .expect("credential contains characters that are invalid in an HTTP header");
        value.set_sensitive(true);
        match self.auth {
            AzureAuth::ApiKey => headers.insert(AZURE_API_KEY_HEADER, value),
            AzureAuth::EntraToken => headers.insert(AUTHORIZATION, value),
        };
        headers
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base, path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn api_key(&self) -> &SecretString {
        &self.api_key
    }
}

/// Normalizes an endpoint into a v1 API base:
/// `my-resource` → `https://my-resource.openai.azure.com/openai/v1`.
fn azure_v1_api_base(endpoint: &str) -> String {
    let trimmed = endpoint.trim().trim_end_matches('/');
    let origin = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}.openai.azure.com")
    };
    if origin.ends_with(AZURE_V1_PATH) {
        origin
    } else {
        format!("{origin}{AZURE_V1_PATH}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_normalize_to_v1_api_bases() {
        for endpoint in [
            "my-resource",
            "https://my-resource.openai.azure.com",
            "https://my-resource.openai.azure.com/",
            "https://my-resource.openai.azure.com/openai/v1",
            "https://my-resource.openai.azure.com/openai/v1/",
        ] {
            assert_eq!(
                azure_v1_api_base(endpoint),
                "https://my-resource.openai.azure.com/openai/v1",
                "endpoint: {endpoint}"
            );
        }
    }

    #[test]
    fn urls_join_the_v1_base_and_path() {
        let config = AzureV1Config::new("my-resource", "key");
        assert_eq!(
            config.url("/chat/completions"),
            "https://my-resource.openai.azure.com/openai/v1/chat/completions"
        );
        assert!(config.query().is_empty());
    }

    #[test]
    fn api_keys_travel_in_the_api_key_header() {
        let config = AzureV1Config::new("my-resource", "azure-key");
        let headers = config.headers();
        assert_eq!(headers.get(AZURE_API_KEY_HEADER).unwrap(), "azure-key");
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn entra_tokens_travel_as_bearer_auth() {
        let config = AzureV1Config::with_entra_token("my-resource", "eyJ-token");
        let headers = config.headers();
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer eyJ-token");
        assert!(headers.get(AZURE_API_KEY_HEADER).is_none());
    }

    #[test]
    fn debug_never_prints_the_credentials() {
        let config = AzureV1Config::new("my-resource", "azure-secret");
        let debug = format!("{config:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("[REDACTED]"));

        let config = AzureV1Config::with_entra_token("my-resource", "token-secret");
        let debug = format!("{config:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("access_token"));
    }
}
