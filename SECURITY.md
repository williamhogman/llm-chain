# Security Policy

We take the security of `llm-chain` very seriously. The behaviour of models is
however unpredictable, and care should be taken when deploying LLMs. Use
extreme caution when allowing LLMs to run code or invoke tools on your
computer.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.14.x  | :white_check_mark: |
| < 0.14  | :x:                |

## Credential handling

All API keys and tokens accepted by the driver crates are stored as
[`secrecy::SecretString`](https://docs.rs/secrecy): they are redacted from
`Debug` output and zeroized on drop. Never log request headers, and prefer
environment variables over hardcoded keys.

## Reporting a Vulnerability

To report a vulnerability privately, write an email to william@sobel.io.
