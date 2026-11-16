# Reproduction

Set paths to authorized processed data and a writable experiment directory outside this checkout. Set `MANIFEST` to the frozen split manifest from the artifact package. The commands below preserve all source and destination identities in each resolved `job.json`.

```sh
export DATA=/path/to/processed
export WORK=/path/to/experiments
export MANIFEST=/path/to/split_manifest.json
export NATIVE=/path/to/native-builds
export WEIGHTS=/path/to/pretrained-assets
export TORCH_HOME=/path/to/torch-cache
export CUDA_VISIBLE_DEVICES=GPU-your-device-uuid
```

Obtain the device UUID with `nvidia-smi --query-gpu=uuid,name --format=csv`. Use one device per process. Keep hardware, thread count, and environment matched within each paired comparison. `--execute` runs a prepared job immediately. Omitting it writes inspectable job specifications without starting computation. Use a fresh output directory for each execution.

## Five-seed benchmark

The fixed matrix is `configs/benchmark-jobs.json`. Select an exact job ID and run its source training, source-only selection, and all three target evaluations. The following example runs one of the 60 learned core fits.

```sh
uv run --locked python -m src.revision.public benchmark \
  --data "$DATA" --manifest "$MANIFEST" --native "$NATIVE" --weights "$WEIGHTS" \
  --output "$WORK/benchmark" --job-id core-faster_rcnn-DAAN-s0 --device cuda:0 --execute
```

Repeat for each core job in the matrix. Classical jobs use `--device cpu` and the two native CPU binaries built in the README. Run the 17 `diagnostic-*` jobs before a full experiment; these use source data and leave test evaluation disabled. The 18 `adjacency-*` jobs use the same command and fixed validation/test partitions. Match each adjacency fit to its same-source, same-seed core fit on hardware and execution policy.

Each completed job writes its selected configuration, training metadata, selected checkpoint when applicable, source-validation predictions, and destination prediction/metric files under `WORK/benchmark/job_id`. The exact retained candidates and source-selection rules are defined in [Protocol](protocol.md).

## Preprocessing ablation

Set `BASE_PROTOCOL` to the byte-exact processed-image protocol supplied with the artifact metadata. Its hash is checked before training. First create one pretrained initialization for each source and seed. This creates nine initial states and their provenance sidecars. It requires the verified torchvision asset in the configured Torch cache.

```sh
uv run --locked python -m src.revision.public initialize \
  --initial-states "$WORK/initial-states" --output "$WORK/initialization-jobs" \
  --device cpu --execute
```

Run all 42 ablation fits sequentially on the assigned GPU, or select one fit with `--job-id` to distribute work over independent devices. All arms for a source/seed load the same initial state. Keep the checkout commit unchanged between initialization and fitting.

```sh
uv run --locked python -m src.revision.public ablation \
  --data "$DATA" --manifest "$MANIFEST" --protocol "$BASE_PROTOCOL" \
  --initial-states "$WORK/initial-states" --output "$WORK/ablation" \
  --device cuda:0 --execute
```

The fixed experiment includes 36 factorial fits and six 316-frame count controls. It performs 126 destination evaluations after source-validation checkpoint selection. Each fit writes `best.pt`, `metrics.json`, source-training transformation audits, and raw destination predictions.

## Supervised adaptation

The frozen adaptation implementation requires an NVIDIA RTX A5000 for both diagnostics and scoring. Set `CUDA_VISIBLE_DEVICES` to that device’s UUID. Set `ADAPT_PROTOCOL` to the frozen adaptation protocol. The source and target-local weights come from the completed S0I0 ablation fits. First run the two support-only diagnostics, then build a gate that verifies their metrics and checksums.

```sh
uv run --locked python -m src.revision.public adaptation-diagnostic \
  --data "$DATA" --manifest "$MANIFEST" --protocol "$BASE_PROTOCOL" \
  --adaptation-protocol "$ADAPT_PROTOCOL" --fits "$WORK/ablation" \
  --output "$WORK/adaptation-diagnostics" --device cuda:0 --execute

uv run --locked python -m src.revision.public adaptation-gate \
  --diagnostics "$WORK/adaptation-diagnostics" --output "$WORK/adaptation-gate.json"
```

The six scoring fits then train only the predictor head and evaluate matched source, adapted, and target-local checkpoints. The gate binds the diagnostics to this implementation commit and protocol.

```sh
uv run --locked python -m src.revision.public adaptation \
  --data "$DATA" --manifest "$MANIFEST" --protocol "$BASE_PROTOCOL" \
  --adaptation-protocol "$ADAPT_PROTOCOL" --fits "$WORK/ablation" \
  --gate "$WORK/adaptation-gate.json" --output "$WORK/adaptation" \
  --device cuda:0 --execute
```

## Checkpoint inference and metric replay

A portable research-artifact `run.json` identifies the model configuration, selected checkpoint, preprocessing, and checkpoint digest. Set `RUN` to this file and `TEST_COCO` to an authorized COCO reference subset for the destination. The strict replay loader verifies the checkpoint and input identities before inference.

```sh
uv run --locked python -m src.revision.release_replay \
  --run "$RUN" --annotations "$TEST_COCO" --images "$DATA/images" \
  --split-manifest "$MANIFEST" --device cpu --output "$WORK/replay.json"
```

Use `--device cuda:0` for GPU inference. Add `--reference-predictions "$REFERENCE"` to check prediction parity with a supplied reference. Parity uses explicit absolute and relative tolerances, both zero by default. Hardware and floating-point execution can change predictions, so record any chosen tolerance.

Score a COCO prediction list or the output of checkpoint replay with the common evaluator. This command writes metrics and input digests without retraining.

```sh
uv run --locked python -m src.revision.evaluate \
  --annotations "$TEST_COCO" --predictions "$WORK/replay.json" \
  --output "$WORK/replay-metrics.json"
```

Exact checkpoint replay requires the corresponding model asset, authorized image/reference subset, and frozen preprocessing. Replaying stored predictions reproduces evaluation metrics; it does not measure training or inference runtime.
