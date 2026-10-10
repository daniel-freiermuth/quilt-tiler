set shell := ["bash", "-cu"]

# Everything CI's lint/test/dependency jobs run, in CI's order.
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo nextest run --workspace --profile ci
    cargo deny check
    cargo machete

# Mutation-test one file: does the suite notice when it breaks?
mutants FILE:
    cargo mutants --workspace -j 4 --file {{FILE}}

# Diffs against the merge-base, so committing straight to master still
# audits everything not yet pushed. A failed `git diff` must abort: an
# empty diff yields zero mutants, which is indistinguishable from a clean
# audit.

# Mutants introduced since BASE, including uncommitted changes.
mutants-diff BASE="github/master":
    diff=$(mktemp) && trap 'rm -f "$diff"' EXIT && \
    git diff --no-color "$(git merge-base {{BASE}} HEAD)" > "$diff" && \
    cargo mutants --workspace -j 4 --in-diff "$diff"

# Whole workspace. Hours; for periodic audits, not per change.
mutants-all:
    cargo mutants --workspace -j 4
