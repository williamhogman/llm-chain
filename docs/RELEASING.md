# Releasing `llm-chain`

All crates share the workspace version in the root `Cargo.toml`
(`[workspace.package] version`). A release ships every crate at that version.

## 1. Pre-flight

- [ ] `CHANGELOG.md` has a dated entry for the new version, and the compare
      links at the bottom are updated.
- [ ] The workspace version is bumped in the root `Cargo.toml`
      (`[workspace.package] version` **and** the `llm-chain` entry in
      `[workspace.dependencies]` — they must match).
- [ ] Docs mention the new version where relevant
      (`docs/README.md` dependency snippet).
- [ ] CI is green on `main`.

## 2. Verify locally

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Then dry-run the packaging for every crate (catches missing metadata and
files not included in the package):

```bash
for crate in llm-chain llm-chain-openai llm-chain-anthropic llm-chain-gemini \
             llm-chain-bedrock llm-chain-ollama llm-chain-llama llm-chain-tools; do
  cargo publish -p "$crate" --dry-run --allow-dirty || exit 1
done
```

Note: dependent crates' dry-runs fail dependency resolution until the new
`llm-chain` version is actually on crates.io — `error: failed to select a
version for llm-chain` from a *driver* crate's dry-run is expected before
`llm-chain` itself is published. `cargo package -p <crate> --list` still
verifies file inclusion for all of them.

## 3. Publish

Publish in dependency order. `cargo publish` (1.66+) waits for each crate to
land in the index before returning, so this can run back-to-back:

```bash
cargo publish -p llm-chain
for crate in llm-chain-openai llm-chain-anthropic llm-chain-gemini \
             llm-chain-bedrock llm-chain-ollama llm-chain-llama llm-chain-tools; do
  cargo publish -p "$crate"
done
```

Alternatively, push a `v*` tag and let `.github/workflows/release.yaml` do
the publishing (requires the `CARGO_REGISTRY_TOKEN` repository secret).

## 4. Tag and announce

```bash
git tag -a v0.14.0 -m "llm-chain 0.14.0"
git push origin v0.14.0
```

Create a GitHub release from the tag and paste the matching `CHANGELOG.md`
section as the release notes (the release workflow drafts this
automatically).

## If something goes wrong

- A bad publish cannot be replaced — bump the patch version and publish
  again.
- `cargo yank --version X.Y.Z -p <crate>` prevents *new* projects from
  resolving a broken version without breaking existing lockfiles.
