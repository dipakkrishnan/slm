#!/bin/sh

# Redirect output to stderr.
exec 1>&2

# Check if we have rustfmt and clippy installed
if ! command -v rustfmt >/dev/null 2>&1; then
    echo "Error: rustfmt not installed. Run 'rustup component add rustfmt'."
    exit 1
fi

if ! command -v cargo-clippy >/dev/null 2>&1; then
    echo "Error: clippy not installed. Run 'rustup component add clippy'."
    exit 1
fi

# Format all staged .rs files
staged_rs_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rs$')
if [ -n "$staged_rs_files" ]; then
    echo "Formatting Rust files..."
    echo "$staged_rs_files" | xargs rustfmt --edition 2021
    git add $staged_rs_files
fi

# Run clippy
echo "Running Clippy..."
cargo clippy -- -D warnings

if [ $? -ne 0 ]; then
    echo "Clippy found issues. Please fix them before committing."
    exit 1
fi

exit 0

