# 🦀 SLM

This project implements a **small auto-regressive transformer** in **Rust**. The transformer is designed for language modeling tasks, and it is trained on the **Wikitext-2** dataset. The project leverages the power of Rust for efficient model training and execution, while Python is used mostly for scripting and preprocessing the dataset. 

## Quick start

Install python dependencies and download datasets:
```
make setup
```

Set necessary environment variables:
```
export LIBTORCH_USE_PYTORCH=1
export RUST_LOG=info cargo run
```

Kickoff training:
```
make train
```

## ✨ Features
- Implementation of an auto-regressive transformer model in Rust.
- Efficient training loop utilizing `tch-rs` for PyTorch-like tensor operations.
