# Marine radar cross-site benchmark

This repository trains and evaluates marine radar detectors across the DAAN, DARC, and MANV acquisition sites. The fixed benchmark compares four learned detectors and five classical methods, followed by controlled image-preprocessing ablations and a small supervised adaptation study. J.C. Vaught maintains the implementation.

The repository contains source code, fixed configurations, dependency locks, and tests. Obtain the dataset through its original provider under the provider's terms. Checkpoints, predictions, and evaluation records belong to the separate [Zenodo research-artifact record](https://zenodo.org/records/24266410). [Data and artifact access](docs/data.md) describes the input layout and download interface.

## Installation

Use Python 3.11, `uv`, and Linux x86-64 with a compatible NVIDIA driver for training. The lock pins PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1, Ultralytics 8.4.14, and pycocotools 2.0.11. CPU checkpoint replay uses the same Python lock.

```sh
uv sync --locked --python 3.11
uv run --locked python -m src.revision.public --help
```

Build the Otsu and Wavelet CPU programs for classical benchmark jobs. Set `NATIVE` to a writable directory outside this checkout.

```sh
cargo build --locked --release --manifest-path src/otsu/Cargo.toml --target-dir "$NATIVE/otsu"
cargo build --locked --release --manifest-path src/wavelet/Cargo.toml --target-dir "$NATIVE/wavelet"
```

The [reproduction guide](docs/reproduction.md) gives commands for the five-seed benchmark, adjacency study, 42-fit ablation, six-fit adaptation, checkpoint inference, and stored-prediction evaluation. [Protocol](docs/protocol.md) defines selection and comparison conditions.

## Verification

Run the Python checks in the locked environment. The tests use synthetic fixtures and do not download the research dataset or model weights. The separate [raw conversion checks](docs/preprocessing.md) use their own small environment.

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked ty check
uv run --locked pytest -q
cargo test --locked --manifest-path src/otsu/Cargo.toml
cargo test --locked --manifest-path src/wavelet/Cargo.toml
```

## License

Original project code is licensed under AGPL-3.0-or-later. Third-party packages retain their own licenses. The code license does not cover external radar data, annotations, pretrained assets, or trained checkpoints. [Third-party notices](THIRD_PARTY_NOTICES.md) identifies the model dependencies and their separate terms.
