"""Portable recipe and artifact-download boundaries."""

import hashlib
import io
import json
from argparse import Namespace

import pytest

from src.revision import download, public


def test_fixed_benchmark_has_current_five_seed_matrix():
    jobs = public.load(public.ROOT / "configs/benchmark-jobs.json")
    core = [j for j in jobs if j["stage"] == "core"]
    learned = [j for j in core if j["method"] in {"unet", "faster_rcnn", "yolo", "rtdetr"}]
    assert len(learned) == 60
    for method in {j["method"] for j in learned}:
        for source in public.SITES:
            assert {
                j["seed"] for j in learned if (j["method"], j["source"]) == (method, source)
            } == set(range(5))
    assert len(core) == 75
    assert sum(j["stage"] == "adjacency" for j in jobs) == 18
    assert all(j["eval_targets"] == [] for j in jobs if j["stage"] == "diagnostic")


def test_benchmark_resolves_paths_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "commit", lambda: "a" * 40)
    args = Namespace(
        data=tmp_path / "data",
        manifest=tmp_path / "split.json",
        native=tmp_path / "native",
        output=tmp_path / "runs",
        weights=tmp_path / "weights",
        job_id="core-otsu-DAAN-s0",
        device="cpu",
        execute=False,
    )
    public.benchmark(args)
    job = json.loads((args.output / args.job_id / "job.json").read_text())
    assert job["ann"] == str(args.data / "annotations/all.json")
    assert job["manifest"] == str(args.manifest)
    assert job["native_otsu"] == str(args.native / "otsu/release/otsu")
    assert all(token not in json.dumps(job) for token in ("{root}", "{data}", "{native}"))
    with pytest.raises(FileExistsError):
        public.benchmark(args)


def test_download_verifies_before_commit_and_rejects_unsafe_names(tmp_path, monkeypatch):
    payload = b"a synthetic artifact"
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda _: io.BytesIO(payload))
    entry = {
        "key": "artifact.tar.gz",
        "checksum": "md5:" + hashlib.md5(payload).hexdigest(),
        "links": {"self": "https://example.invalid/artifact"},
    }
    assert download.download(entry, tmp_path).read_bytes() == payload
    with pytest.raises(FileExistsError):
        download.download(entry, tmp_path)
    entry["key"] = "../outside"
    with pytest.raises(ValueError, match="Unsafe"):
        download.download(entry, tmp_path)
    entry.update(key="bad.tar.gz", checksum="md5:" + "0" * 32)
    with pytest.raises(ValueError, match="checksum"):
        download.download(entry, tmp_path)
    assert not (tmp_path / "bad.tar.gz").exists()
    assert not (tmp_path / "bad.tar.gz.partial").exists()


def test_ablation_selects_only_its_initial_state_and_rejects_unknown_id(tmp_path, monkeypatch):
    initial = tmp_path / "DARC-s1.pt"
    initial.write_bytes(b"synthetic initialization")
    metadata = {
        "initial_state_sha256": public.sha256_file(initial),
        "initial_tensor_sha256": "f" * 64,
    }
    public.save_json(str(initial) + ".json", metadata)
    monkeypatch.setattr(public, "checked", lambda path, digest: str(path))
    emitted = []
    monkeypatch.setattr(public, "emit", lambda job, *args: emitted.append(job))
    args = Namespace(
        protocol=tmp_path / "protocol.json",
        initial_states=tmp_path,
        data=tmp_path / "data",
        manifest=tmp_path / "split.json",
        job_id="darc-v2-fit-DARC-s1-S0I1",
    )
    public.ablation(args)
    assert len(emitted) == 1 and emitted[0]["intensity"] == "I1"
    args.job_id = "nonexistent-job"
    with pytest.raises(ValueError, match="No ablation job"):
        public.ablation(args)
