# Same gates as .github/workflows/ci.yml — keep the two in sync.
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
