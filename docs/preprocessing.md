# Raw conversion

The conversion harness extracts radar rasters and reference boxes from legally obtained DLR captures. It preserves the source helper bytes and verifies their recorded SHA-256 and Git blob identities before execution. These helpers implement a candidate processing recipe. Original raw captures were unavailable for validation, and neither preprocessing calibration nor upstream label provenance has been resolved. The synthetic tests verify implementation contracts; they do not establish equivalence with the frozen benchmark dataset.

Use a dedicated Python 3.11 environment because this converter pins NumPy 2.2.6, OpenCV headless 4.10.0.84, SciPy 1.16.3, and scikit-learn 1.7.2. The requirements file includes package hashes. Set `RAW` to a directory containing `DAAN/polarMeas.json`, `DAAN/radarImages/000.png`, and corresponding DARC and MANV directories. Set `REBUILT` to a new directory outside the raw input tree.

```sh
uv venv --python 3.11 .venv-preprocessing
uv pip sync --python .venv-preprocessing/bin/python --require-hashes \
  src/revision/raw_rebuild.requirements.txt
.venv-preprocessing/bin/python -m src.revision.raw_rebuild \
  --raw-root "$RAW" --output-root "$REBUILT"
```

The harness writes both `clahe_off` and `clahe_on` image variants with the same reference boxes. Contrast-limited adaptive histogram equalization (CLAHE) occurs after masking. The converter associates detected radar blobs with the supplied `polarMeas.json` records and consumes no independent AIS file. Original helper names and comments containing “AIS” or “ground truth” do not independently verify those descriptions. A `referenceMeas.mat` file, when present, is inventoried but not used to construct labels.

Each variant contains `annotations/all.json`, `images/`, and an image manifest. The conversion also writes an input inventory, frame ledger, settings, source hashes, dependency versions, and completion state. It refuses existing output roots, missing requested sites, ambiguous frame keys, incompatible image geometry, and input-tree symlinks.

If authorized processed benchmark inputs are available, add `--release-annotations "$DATA/annotations/all.json" --release-images "$DATA/images"`. The resulting comparison records pixel equality, annotation multiset equality, ordering, categories, missing frames, and unreadable images. Counts alone do not establish equality. The fixed ablation and adaptation runners require the original frozen hashes and therefore reject a differing reconstruction.

Run the synthetic conversion tests in this dedicated environment. Keep the frozen helpers unchanged, since formatting their bytes invalidates the recorded source digests.

```sh
.venv-preprocessing/bin/python -m unittest discover \
  -s tests-preprocessing -p 'test_raw_rebuild.py' -v
```
