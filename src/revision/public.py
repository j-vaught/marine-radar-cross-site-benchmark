"""Prepare portable benchmark, ablation, and adaptation jobs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from .protocol import save_json, sha256_file

SITES = ("DAAN", "DARC", "MANV")
ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_SHA = "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384"
BASE_SHA = "14a06aad755ef18a9fb4b90371ea036057af22e57f28e7a892278ac3e721ea16"
ADAPT_SHA = "d7c4d4f2395bdc5fa8bb00f9badf12fdd57c94f3c48ae5d1428ed2a6352dc549"


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load(path):
    return json.loads(Path(path).read_text())


def checked(path, digest):
    path = Path(path).resolve()
    if sha256_file(path) != digest:
        raise ValueError(f"Frozen input digest mismatch: {path.name}")
    return str(path)


def replace_paths(value, values):
    if isinstance(value, str):
        return value.format(**values)
    if isinstance(value, list):
        return [replace_paths(item, values) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, values) for key, item in value.items()}
    return value


def emit(job, args, module, action=None):
    output = args.output.resolve() / job["job_id"]
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    job.update(out_dir=str(output), code_commit=commit(), device=args.device)
    if args.device.startswith("cuda"):
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not gpu.startswith("GPU-") or "," in gpu:
            raise ValueError("Set CUDA_VISIBLE_DEVICES to one GPU UUID")
        job["gpu_uuid"] = gpu
    save_json(output / "job.json", job)
    if args.execute:
        command = [sys.executable, "-m", module]
        if action:
            command.append(action)
        subprocess.run(command + [str(output / "job.json")], cwd=ROOT, check=True)
    print(output / "job.json")


def inputs(args):
    return {
        "ann": str((args.data / "annotations/all.json").resolve()),
        "images": str((args.data / "images").resolve()),
        "manifest": str(args.manifest.resolve()),
    }


def benchmark(args):
    values = {
        "data": str(args.data.resolve()),
        "root": str(args.output.resolve()),
        "native": str(args.native.resolve()),
    }
    jobs = load(ROOT / "configs/benchmark-jobs.json")
    selected = [job for job in jobs if job["job_id"] == args.job_id]
    if len(selected) != 1:
        raise ValueError("Choose one job_id from configs/benchmark-jobs.json")
    job = replace_paths(selected[0], values)
    job.update(inputs(args), weights_dir=str(args.weights.resolve()))
    emit(job, args, "src.revision.run")


def initialize(args):
    for site, seed in itertools.product(SITES, range(3)):
        job = {
            "job_id": f"initialize-{site}-s{seed}",
            "source": site,
            "seed": seed,
            "initial_state": str((args.initial_states / f"{site}-s{seed}.pt").resolve()),
            "pretrained_weights_sha256": PRETRAINED_SHA,
        }
        emit(job, args, "src.revision.darc_train", "initialize")


def ablation(args):
    frozen = checked(args.protocol, BASE_SHA)
    selected = 0
    for site, seed in itertools.product(SITES, range(3)):
        if args.job_id and not args.job_id.startswith(f"darc-v2-fit-{site}-s{seed}-"):
            continue
        initial = (args.initial_states / f"{site}-s{seed}.pt").resolve()
        meta = load(str(initial) + ".json")
        if sha256_file(initial) != meta["initial_state_sha256"]:
            raise ValueError("Initial-state digest mismatch")
        arms = [(s, i, False) for s, i in itertools.product(("S0", "S1"), ("I0", "I1"))]
        if site != "DARC":
            arms.append(("S0", "I0", True))
        for scale, intensity, reduced in arms:
            name = f"darc-v2-fit-{site}-s{seed}-{scale}{intensity}" + ("-n316" if reduced else "")
            if args.job_id and args.job_id != name:
                continue
            selected += 1
            job = {
                **inputs(args),
                "job_id": name,
                "method": "faster_rcnn",
                "source": site,
                "seed": seed,
                "stage": "core",
                "condition": "buffered",
                "scale": scale,
                "intensity": intensity,
                "size_control": reduced,
                "eval_targets": list(SITES),
                "frozen_protocol": frozen,
                "frozen_protocol_sha256": BASE_SHA,
                "initial_state": str(initial),
                "initial_state_sha256": meta["initial_state_sha256"],
                "initial_tensor_sha256": meta["initial_tensor_sha256"],
                "pretrained_weights_sha256": PRETRAINED_SHA,
                "workers": 2,
            }
            emit(job, args, "src.revision.darc_train", "run")
    if not selected:
        raise ValueError("No ablation job matches the requested job ID")


def checkpoint_fields(root, site, seed, prefix):
    run = root / f"darc-v2-fit-{site}-s{seed}-S0I0"
    weights, metrics = run / "best.pt", run / "metrics.json"
    return {
        f"{prefix}_checkpoint": str(weights.resolve()),
        f"{prefix}_checkpoint_sha256": sha256_file(weights),
        f"{prefix}_metrics": str(metrics.resolve()),
        f"{prefix}_metrics_sha256": sha256_file(metrics),
    }


def adaptation(args):
    frozen = checked(args.protocol, BASE_SHA)
    adaptation_file = checked(args.adaptation_protocol, ADAPT_SHA)
    diagnostic = args.action == "adaptation-diagnostic"
    for row in load(adaptation_file)["jobs"]:
        source, target, seed = row["source"], row["target"], row["seed"]
        if diagnostic and seed != 0:
            continue
        job = {
            **inputs(args),
            "source": source,
            "target": target,
            "seed": seed,
            "job_id": row["job_id"] + ("-diagnostic" if diagnostic else ""),
            "stage": "diagnostic" if diagnostic else "pilot",
            "adaptation_protocol": adaptation_file,
            "adaptation_protocol_sha256": ADAPT_SHA,
            "base_protocol": frozen,
            "base_protocol_sha256": BASE_SHA,
            "pretrained_weights_sha256": PRETRAINED_SHA,
            "workers": 2,
            **checkpoint_fields(args.fits, source, seed, "source"),
        }
        if not diagnostic:
            job.update(checkpoint_fields(args.fits, target, seed, "target_local"))
            job.update(
                score_gate=str(args.gate.resolve()), score_gate_sha256=sha256_file(args.gate)
            )
        emit(job, args, "src.revision.darc_adaptation")


def gate(args):
    from .darc_adaptation import _check_score_gate

    diagnostics = []
    for source, target in (("DARC", "MANV"), ("MANV", "DARC")):
        name = f"darc-adapt-v1-{source}-to-{target}-s0-diagnostic"
        path = (args.diagnostics / name / "metrics.json").resolve()
        diagnostics.append(
            {"job_id": name, "metrics_path": str(path), "metrics_sha256": sha256_file(path)}
        )
    document = {
        "schema": "darc-adaptation-score-gate-v1",
        "protocol_sha256": ADAPT_SHA,
        "code_commit": commit(),
        "passed": True,
        "diagnostics": diagnostics,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    save_json(args.output, document)
    try:
        _check_score_gate(
            {
                "score_gate": str(args.output),
                "score_gate_sha256": sha256_file(args.output),
                "code_commit": commit(),
            },
            ADAPT_SHA,
        )
    except Exception:
        args.output.unlink()
        raise
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("benchmark", "initialize", "ablation", "adaptation-diagnostic", "adaptation"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--execute", action="store_true", help="Execute each prepared job")
        if name != "initialize":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--manifest", type=Path, required=True)
        if name == "benchmark":
            command.add_argument("--job-id", required=True)
            command.add_argument("--native", type=Path, required=True)
            command.add_argument("--weights", type=Path, required=True)
        if name in ("initialize", "ablation"):
            command.add_argument("--initial-states", type=Path, required=True)
        if name == "ablation":
            command.add_argument("--job-id")
        if name in ("ablation", "adaptation-diagnostic", "adaptation"):
            command.add_argument("--protocol", type=Path, required=True)
        if name.startswith("adaptation"):
            command.add_argument("--adaptation-protocol", type=Path, required=True)
            command.add_argument("--fits", type=Path, required=True)
        if name == "adaptation":
            command.add_argument("--gate", type=Path, required=True)
    command = commands.add_parser("adaptation-gate")
    command.add_argument("--diagnostics", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    functions = {
        "benchmark": benchmark,
        "initialize": initialize,
        "ablation": ablation,
        "adaptation-diagnostic": adaptation,
        "adaptation-gate": gate,
        "adaptation": adaptation,
    }
    functions[args.action](args)


if __name__ == "__main__":
    main()
