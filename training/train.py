#!/usr/bin/env python3
"""Fine-tune FaceNet on the fixed DigiFace split: B1 (base) or B2 (occluded).

Place this file at <repo>/training/train.py. Uses only DigiFace training images for
updates, and only DigiFace validation for checkpoint/threshold selection. Test,
ROF, and MFR2 images are never opened. See README_TRAINING.md for commands.

Validated against the 2026-10-28 B0 evaluator's *pipeline A* definition:
PIL BILINEAR 160x160, RGB with white alpha background, (pixel-127.5)/128,
512-D L2 embeddings, mean-then-L2 gallery (3 images/person), rank-1 cosine,
max-DIR threshold under empirical FPIR<=5% (tie: higher threshold).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import importlib.metadata
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_VERSION = "1.0.1"
IMAGE_SIZE = 160
MEAN = 127.5
STD = 128.0
ROLE_REL = "ban_giao_nhom/protocol/notebook_outputs/tables/05_image_roles_draft.csv"
PROTOCOL_REL = "ban_giao_nhom/protocol/notebook_outputs/protocol_draft.json"
DECISIONS_REL = "ban_giao_nhom/quyet_dinh_nhom/input/project_decisions.csv"
CHOICES = {
    "P1": "conservative_withhold",
    "P2": "base_gallery_3",
    "P3": "external_test_only",
    "P4": "fpir_05_empirical",
    "P5": "direct_resize_digiface",
}


@dataclass(frozen=True)
class FaceRow:
    image_id: str
    identity: str
    role: str
    partition: str
    relative_path: str
    sha256: str
    qa_flagged: bool


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def check_contract(project: Path) -> dict[str, Any]:
    p = project / PROTOCOL_REL
    d = project / DECISIONS_REL
    if not p.is_file() or not d.is_file():
        raise FileNotFoundError(f"Missing protocol or confirmed decisions: {p}, {d}")
    contract = read_json(p).get("evaluation_contract", {})
    required = {
        "problem": "open_set_identification_rank1",
        "primary_metric": "DIR_at_empirical_validation_FPIR_target",
        "target_validation_fpir": 0.05,
        "threshold_selection": "max_DIR_subject_to_empirical_FPIR_le_target; ties:higher_threshold",
        "score_accept_rule": "top_cosine >= threshold",
        "gallery_template": "mean_of_L2_embeddings_then_L2_normalize",
        "known_failure_denominator": "all_input_known_queries",
        "unknown_failure_denominator": "all_input_unknown_queries",
        "external_tuning_allowed": False,
    }
    for k, expected in required.items():
        if contract.get(k) != expected:
            raise ValueError(f"Protocol mismatch: {k}: got {contract.get(k)!r}, expected {expected!r}")
    with d.open(newline="", encoding="utf-8-sig") as f:
        decisions = list(csv.DictReader(f))
    chosen = {r["decision_id"]: r["chosen_option"] for r in decisions}
    if chosen != CHOICES or len(decisions) != len(CHOICES) or not all(r.get("reviewer", "").strip() for r in decisions):
        raise ValueError("P1-P5 decisions not confirmed or different from B0 notebook")
    return contract


def load_manifest(project: Path) -> tuple[list[FaceRow], list[FaceRow], dict[str, Any]]:
    """Read all metadata for leakage checks, but NEVER open test images."""
    path = project / ROLE_REL
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed image-role CSV: {path}")
    train: list[FaceRow] = []
    val: list[FaceRow] = []
    counts: Counter[tuple[str, str]] = Counter()
    identities: dict[str, set[str]] = defaultdict(set)
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    role_detail: Counter[tuple[str, str, str]] = Counter()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"image_id", "source", "canonical_identity", "path", "sha256", "partition", "role", "identity_role", "qa_reasons"}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Manifest missing fields: {sorted(needed - set(reader.fieldnames or []))}")
        for r in reader:
            if r["source"] != "DigiFace":
                raise ValueError("Expected DigiFace-only manifest; found another source")
            raw_path = r["path"]
            pure = PurePosixPath(raw_path)
            if pure.is_absolute() or ".." in pure.parts or not raw_path.startswith("data/raw/digiface/images_eda/"):
                raise ValueError(f"Unexpected/unsafe relative path: {raw_path!r}")
            if len(r["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["sha256"].lower()):
                raise ValueError(f"Invalid SHA-256 for {raw_path}")
            if r["image_id"] in image_ids or raw_path in image_paths:
                raise ValueError(f"Repeated image ID or path: {raw_path}")
            image_ids.add(r["image_id"]); image_paths.add(raw_path)
            part, role, ident = r["partition"], r["role"], r["canonical_identity"]
            counts[(part, role)] += 1
            identities[part].add(ident)
            role_detail[(part, role, r["identity_role"])] += 1
            row = FaceRow(r["image_id"], ident, role, part, raw_path, r["sha256"], bool(r["qa_reasons"]))
            if part == "train" and role == "train":
                train.append(row)
            elif part == "val" and role in {"gallery", "known_query", "unknown_query"}:
                val.append(row)
    if len(image_ids) != 80_000:
        raise ValueError(f"Expected 80,000 total images, got {len(image_ids)}")
    expected_counts = {
        ("train", "train"): 57_600, ("train", "train_monitor"): 6_400,
        ("val", "gallery"): 300, ("val", "known_query"): 3_700,
        ("val", "unknown_query"): 4_000,
        ("test", "gallery"): 300, ("test", "known_query"): 3_700,
        ("test", "unknown_query"): 4_000,
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"Manifest roles differ from fixed protocol: {dict(counts)}")
    if {k: len(v) for k, v in identities.items()} != {"train": 1600, "val": 200, "test": 200}:
        raise ValueError("Expected 1,600/200/200 identities")
    parts = ["train", "val", "test"]
    if any(identities[a] & identities[b] for i, a in enumerate(parts) for b in parts[i+1:]):
        raise ValueError("Identity leakage across train/val/test")
    if len({r.identity for r in train}) != 1600:
        raise ValueError("Training identity count inconsistent")
    if not (role_detail[("val", "gallery", "known")] == 300 and
            role_detail[("val", "known_query", "known")] == 3700 and
            role_detail[("val", "unknown_query", "unknown")] == 4000):
        raise ValueError("Validation known/unknown roles do not match B0")
    val_known = {r.identity for r in val if r.role in {"gallery", "known_query"}}
    val_unknown = {r.identity for r in val if r.role == "unknown_query"}
    if len(val_known) != 100 or len(val_unknown) != 100 or val_known & val_unknown:
        raise ValueError("Validation must have 100 known and 100 unknown identities")
    gallery_counts = Counter(r.identity for r in val if r.role == "gallery")
    if any(gallery_counts[identity] != 3 for identity in val_known):
        raise ValueError("Each validation-known identity must have exactly 3 gallery images")
    return sorted(train, key=lambda r: r.image_id), sorted(val, key=lambda r: r.image_id), {
        "manifest_sha256": file_sha256(path),
        "train_images": len(train), "train_monitor_excluded": counts[("train", "train_monitor")],
        "val_images": len(val), "train_identities": 1600,
        "val_known_identities": 100, "val_unknown_identities": 100,
        "train_qa_flagged": sum(r.qa_flagged for r in train),
        "val_qa_flagged": sum(r.qa_flagged for r in val),
        "test_images_opened": 0,
    }


def verify_files(data_root: Path, rows: Sequence[FaceRow], check_hash: bool) -> None:
    """Verify only train+validation. Never read the test images."""
    for i, row in enumerate(rows, start=1):
        path = data_root / row.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Data missing for image {row.image_id}: {path}")
        if check_hash and file_sha256(path) != row.sha256:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if i % 10_000 == 0:
            print(f"[preflight] verified {i}/{len(rows)} files", flush=True)


def read_rgb(path: Path) -> Image.Image:
    """Exact B0 image-loading policy (EXIF, alpha composited on white, RGB)."""
    with Image.open(path) as src:
        src.load()
        im = ImageOps.exif_transpose(src)
        if "A" in im.getbands():
            rgba = im.convert("RGBA")
            if (np.asarray(rgba.getchannel("A")) < 255).any():
                im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
        return im.convert("RGB").copy()


MASK_NAMES = ("eyes", "lower", "left", "right")


def mask_for_image(seed: int, epoch: int, image_id: str, probability: float) -> dict[str, Any]:
    """Pure deterministic augmentation schedule, reproducible without worker RNG state."""
    key = f"{SCRIPT_VERSION}/{seed}/{epoch}/{image_id}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.default_rng(derived_seed)
    enabled = bool(rng.random() < probability)
    if not enabled:
        return {"seed": derived_seed, "kind": "none", "x0": 0, "y0": 0, "x1": 0, "y1": 0, "fill": 0}
    kind = MASK_NAMES[int(rng.integers(0, len(MASK_NAMES)))]
    if kind == "eyes":
        x0, y0 = int(rng.integers(22, 37)), int(rng.integers(33, 47))
        x1, y1 = int(rng.integers(124, 139)), int(rng.integers(70, 86))
    elif kind == "lower":
        x0, y0 = int(rng.integers(10, 23)), int(rng.integers(81, 97))
        x1, y1 = int(rng.integers(139, 153)), IMAGE_SIZE
    elif kind == "left":
        x0, y0 = 0, int(rng.integers(11, 25))
        x1, y1 = int(rng.integers(71, 89)), IMAGE_SIZE
    else:
        x0, y0 = int(rng.integers(71, 89)), int(rng.integers(11, 25))
        x1, y1 = IMAGE_SIZE, IMAGE_SIZE
    return {"seed": derived_seed, "kind": kind, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "fill": 0}


def make_tensor(path: Path, mask: dict[str, Any] | None = None) -> torch.Tensor:
    rgb = read_rgb(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pix = np.asarray(rgb, dtype=np.uint8).copy()
    if mask is not None and mask["kind"] != "none":
        pix[mask["y0"]:mask["y1"], mask["x0"]:mask["x1"], :] = mask["fill"]
    # Same B0 pipeline A normalization, exactly once.
    return (torch.from_numpy(np.ascontiguousarray(pix)).permute(2, 0, 1).float() - MEAN) / STD


class Faces(Dataset):
    def __init__(self, rows: Sequence[FaceRow], data_root: Path, labels: dict[str, int] | None = None,
                 mode: str = "val", seed: int = 0, occlusion_probability: float = 0.5):
        self.rows = list(rows)
        self.data_root = data_root
        self.labels = labels or {}
        self.mode = mode
        self.seed = seed
        self.epoch = 0
        self.probability = occlusion_probability

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        mask = mask_for_image(self.seed, self.epoch, r.image_id, self.probability) if self.mode == "B2" else None
        img = make_tensor(self.data_root / r.relative_path, mask)
        if self.mode == "val":
            return img, idx
        return img, self.labels[r.identity]


def export_mask_schedule(rows: Sequence[FaceRow], path: Path, seed: int, epoch: int, prob: float) -> dict[str, int]:
    """Auditable CSV, sufficient to exactly reconstruct masks without saving generated images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    counts: Counter[str] = Counter()
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            fields = ["image_id", "epoch", "seed", "kind", "x0", "y0", "x1", "y1", "fill"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                mask = mask_for_image(seed, epoch, r.image_id, prob)
                w.writerow({"image_id": r.image_id, "epoch": epoch, **mask})
                counts[mask["kind"]] += 1
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dict(counts)


def select_threshold(known_scores: np.ndarray, correct_known: np.ndarray,
                     unknown_scores: np.ndarray, target: float = 0.05) -> dict[str, Any]:
    """Exact B0 operating-point rule; all queries count in denominators."""
    if len(known_scores) != 3700 or len(unknown_scores) != 4000:
        raise ValueError("Must include 3,700 known + 4,000 unknown queries")
    if not np.isfinite(known_scores).all() or not np.isfinite(unknown_scores).all():
        raise ValueError("Nonfinite validation cosine scores")
    sorted_correct = np.sort(known_scores[correct_known]); sorted_unknown = np.sort(unknown_scores)
    candidates = np.concatenate([np.unique(np.concatenate([known_scores, unknown_scores])), [np.inf]])
    known_count = len(sorted_correct) - np.searchsorted(sorted_correct, candidates, side="left")
    unknown_count = len(sorted_unknown) - np.searchsorted(sorted_unknown, candidates, side="left")
    feasible = unknown_count / len(unknown_scores) <= target
    max_correct = int(known_count[feasible].max())
    best_threshold = float(candidates[feasible & (known_count == max_correct)].max())
    if not math.isfinite(best_threshold):
        # All-rejected finite equivalent on this validation cohort; JSON compliant.
        max_score = np.float32(max(float(known_scores.max()), float(unknown_scores.max())))
        best_threshold = float(np.nextafter(max_score, np.float32(np.inf)))
    accepted_known = known_scores >= best_threshold
    accepted_unknown = unknown_scores >= best_threshold
    hits = int((accepted_known & correct_known).sum())
    false_accepted = int(accepted_unknown.sum())
    wrong_identity = int((accepted_known & ~correct_known).sum())
    assert false_accepted <= int(target * len(unknown_scores) + 1e-9)
    assert hits == max_correct
    return {
        "DIR": hits / len(known_scores), "FPIR": false_accepted / len(unknown_scores),
        "threshold": best_threshold, "known_correct_accepted": hits,
        "known_wrong_identity_accepted": wrong_identity,
        "known_rejected": int((~accepted_known).sum()),
        "unknown_accepted": false_accepted,
        "unknown_rejected": int((~accepted_unknown).sum()),
        "known_queries": len(known_scores), "unknown_queries": len(unknown_scores),
        "closed_set_rank1": float(correct_known.mean()),
    }


@torch.inference_mode()
def evaluate_a(encoder: nn.Module, rows: Sequence[FaceRow], data_root: Path,
               device: torch.device, batch_size: int, workers: int) -> dict[str, Any]:
    """Evaluate on DigiFace validation ONLY; never use external/test for tuning."""
    encoder.eval()
    ds = Faces(rows, data_root, mode="val")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    embeddings = np.empty((len(rows), 512), dtype=np.float32)
    for images, indices in loader:
        e = F.normalize(encoder(images.to(device, non_blocking=True)), p=2, dim=1)
        if tuple(e.shape[1:]) != (512,) or not bool(torch.isfinite(e).all()):
            raise ValueError("Encoder must emit 512 finite embedding coordinates")
        embeddings[indices.numpy()] = e.float().cpu().numpy()
    enrolled_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    for r, e in zip(rows, embeddings):
        if r.role == "gallery":
            enrolled_by_id[r.identity].append(e)
    names = sorted(enrolled_by_id)
    if len(names) != 100 or any(len(enrolled_by_id[x]) != 3 for x in names):
        raise ValueError("Gallery incomplete: expected 100 people x 3 images")
    templates = np.stack([np.mean(enrolled_by_id[x], axis=0) for x in names], axis=0).astype(np.float32)
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    query_indices = [i for i, r in enumerate(rows) if r.role in {"known_query", "unknown_query"}]
    query_rows = [rows[i] for i in query_indices]
    similarities = embeddings[query_indices] @ templates.T
    best = similarities.argmax(axis=1)
    top_scores = similarities[np.arange(len(query_rows)), best]
    is_known = np.asarray([r.role == "known_query" for r in query_rows])
    correct = np.asarray([names[int(i)] == r.identity for i, r in zip(best, query_rows)])
    metric = select_threshold(top_scores[is_known], correct[is_known], top_scores[~is_known], target=0.05)
    return {**metric, "validation_images": len(rows), "gallery_identities": len(names),
            "preprocessing": "A_direct_resize_160_PIL_BILINEAR", "embedding_dim": 512}


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda specified, but CUDA is unavailable")
    return torch.device(name)


def create_encoder(pretrained: bool) -> nn.Module:
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2" if pretrained else None, classify=False)
    # facenet-pytorch creates an unused VGGFace2 logits head (8631 classes) when
    # pretrained=True, but NOT when pretrained=None and classify=False. Remove
    # it in both cases so training checkpoints can be reloaded strictly without
    # downloading pretrained weights, and so optimizer does not track dead params.
    if hasattr(model, "logits"):
        del model.logits
    return model


def model_sha256(model: nn.Module) -> str:
    """Fingerprint the initial state (including BN buffers) for B1/B2 comparability."""
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["B0", "B1", "B2"], required=True,
                   help="B0 evaluates pretrained weights only; B1/B2 fine-tune")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                   help="Repository root containing ban_giao_nhom/")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Root for CSV relative paths; default --project-root. Can use symlink/mount.")
    p.add_argument("--output-root", type=Path, default=None, help="Defaults to <project-root>/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--occlusion-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--skip-hash-check", action="store_true", help="Faster but less safe: check paths only")
    p.add_argument("--preflight-only", action="store_true", help="Verify manifest, decisions, and data, without loading weights")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="Debug only: run N optimizer steps, no validation, checkpoint named smoke.pt")
    p.add_argument("--resume", type=Path, default=None, help="Resume a trusted last.pt checkpoint")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch sizes must be positive; num_workers >= 0")
    if not (0.0 <= args.occlusion_prob <= 1.0) or args.smoke_steps < 0:
        raise ValueError("occlusion-prob must be in [0,1]; smoke-steps >= 0")
    if min(args.backbone_lr, args.head_lr) <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if args.mode == "B0" and (args.resume or args.smoke_steps):
        raise ValueError("B0 only supports validation of pretrained model")
    project = args.project_root.expanduser().resolve()
    data_root = (args.data_root or project).expanduser().resolve()
    contract = check_contract(project)
    train, val, manifest_summary = load_manifest(project)
    selected = val if args.mode == "B0" else train + val
    print("[protocol]", json.dumps(manifest_summary, ensure_ascii=False), flush=True)
    print(f"[paths] project={project} data_root={data_root}", flush=True)
    verify_files(data_root, selected, not args.skip_hash_check)
    if args.preflight_only:
        print("[PASS] Preflight complete. No model loaded and no train/test images used.")
        return 0

    device = resolve_device(args.device)
    set_reproducibility(args.seed)
    if args.mode == "B0":
        encoder = create_encoder(pretrained=True).to(device)
        metrics = evaluate_a(encoder, val, data_root, device, args.eval_batch_size, args.num_workers)
        print("[B0 validation A]", json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    run_name = args.run_name or f"{args.mode}_seed{args.seed}"
    if not run_name or run_name in {".", ".."} or any(x in run_name for x in ("/", "\\")):
        raise ValueError("run-name must be a simple, nonempty folder name")
    output_root = (args.output_root or project / "runs").expanduser().resolve()
    run_dir = output_root / run_name
    if args.resume:
        if args.smoke_steps:
            raise ValueError("Do not resume a smoke test")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory missing: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite previous experiments
    (run_dir / "masks").mkdir(exist_ok=True)

    # Same label map for B1/B2 (sorted canonical identities); never use validation identities.
    label_map = {identity: i for i, identity in enumerate(sorted({r.identity for r in train}))}
    assert len(label_map) == 1600
    ds = Faces(train, data_root, labels=label_map, mode=args.mode,
               seed=args.seed, occlusion_probability=args.occlusion_prob)
    encoder = create_encoder(pretrained=not bool(args.resume)).to(device)
    head = nn.Linear(512, len(label_map)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, global_step = 1, 0
    best: dict[str, Any] | None = None
    settings = {
        "script_version": SCRIPT_VERSION, "mode": args.mode, "epochs": args.epochs,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers, "seed": args.seed,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "occlusion_prob": args.occlusion_prob if args.mode == "B2" else 0.0,
        "training_loss": "cross_entropy_over_1600_train_identities",
        "augmentation": "deterministic_rectangular_occlusion_RGB_black_after_resize_before_normalization",
        "backbone": "InceptionResnetV1_VGGFace2_classify_False",
        "preprocessing": "PIL_BILINEAR_RGB160_(pixel-127.5)/128",
        "monitor_excluded_from_gradient_updates": True,
        "checkpoint_selection": "max_validation_DIR_at_empirical_FPIR_le_0.05; ties:min_FPIR_then_earliest_epoch",
        "validation": "DigiFace val only; no test or ROF/MFR2",
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "protocol_target_fpir": contract["target_validation_fpir"],
        "smoke_test": bool(args.smoke_steps),
    }
    if args.resume:
        settings["initial_encoder_sha256"] = read_json(run_dir / "config.json")["initial_encoder_sha256"]
        settings["initial_head_sha256"] = read_json(run_dir / "config.json")["initial_head_sha256"]
        # Only load checkpoints produced locally or from trusted team storage.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("settings") != settings or ckpt.get("label_map") != label_map:
            raise ValueError("Resume configuration/identity mapping differs; refusing to continue")
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best = ckpt.get("best_metrics")
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if amp and ckpt.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_states"])
        print(f"[resume] epoch {start_epoch}/{args.epochs}, step {global_step}")
    else:
        settings["initial_encoder_sha256"] = model_sha256(encoder)
        settings["initial_head_sha256"] = model_sha256(head)
        atomic_json(run_dir / "config.json", settings)
        atomic_json(run_dir / "manifest_summary.json", manifest_summary)
        atomic_json(run_dir / "environment.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__, "facenet_pytorch": importlib.metadata.version("facenet-pytorch"),
            "numpy": np.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if amp else None,
        })
        atomic_json(run_dir / "label_map.json", label_map)
    if start_epoch > args.epochs:
        print("Already finished the configured number of epochs.")
        return 0

    for epoch in range(start_epoch, args.epochs + 1):
        ds.set_epoch(epoch)
        if args.mode == "B2":
            counts = export_mask_schedule(train, run_dir / "masks" / f"epoch_{epoch:03d}.csv",
                                          args.seed, epoch, args.occlusion_prob)
            print(f"[augmentation] epoch {epoch}: {counts}", flush=True)
        # Explicit per-epoch seed makes B1/B2 sample order identical for paired runs.
        order_generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            generator=order_generator, num_workers=args.num_workers,
                            pin_memory=amp, persistent_workers=False, drop_last=False)
        encoder.train(); head.train()
        running_loss, samples, epoch_steps = 0.0, 0, 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                features = encoder(images)
                logits = head(features)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = len(targets)
            running_loss += float(loss.detach()) * count
            samples += count
            epoch_steps += 1
            global_step += 1
            if args.smoke_steps and epoch_steps >= args.smoke_steps:
                break
            if epoch_steps % 100 == 0:
                print(f"[{args.mode}] epoch {epoch}/{args.epochs}, steps={epoch_steps}, "
                      f"avg_loss={running_loss/samples:.5f}", flush=True)
        if not args.smoke_steps and (samples != len(train) or epoch_steps != math.ceil(len(train) / args.batch_size)):
            raise RuntimeError("Incomplete epoch; refusing to call this a full B1/B2 run")
        training = {"mode": args.mode, "epoch": epoch, "step": global_step,
                    "optimizer_steps_this_epoch": epoch_steps, "samples_seen": samples,
                    "train_loss": running_loss / max(1, samples)}
        if args.smoke_steps:
            print("[SMOKE]", json.dumps(training, ensure_ascii=False))
            save_checkpoint(run_dir / "smoke.pt", {
                "settings": settings, "label_map": label_map, "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(), "head_state_dict": head.state_dict(),
                "training": training, "smoke_only": True})
            print(f"[SMOKE] Saved {run_dir/'smoke.pt'}; NOT an official experiment.")
            return 0

        metrics = evaluate_a(encoder, val, data_root, device, args.eval_batch_size, args.num_workers)
        epoch_record = {**training, **metrics}
        atomic_json(run_dir / f"epoch_{epoch:03d}_validation.json", epoch_record)
        print(f"[{args.mode}] epoch {epoch}: loss={training['train_loss']:.5f}, "
              f"DIR={metrics['DIR']:.4f}, FPIR={metrics['FPIR']:.4f}, "
              f"threshold={metrics['threshold']:.5f}", flush=True)
        is_best = (best is None or metrics["DIR"] > best["DIR"] + 1e-12 or
                   (abs(metrics["DIR"] - best["DIR"]) <= 1e-12 and
                    metrics["FPIR"] < best["FPIR"] - 1e-12))
        if is_best:
            best = {"epoch": epoch, **metrics}
        state = {
            "schema_version": 1, "settings": settings, "label_map": label_map,
            "epoch": epoch, "global_step": global_step, "training": training,
            "validation_metrics": metrics, "best_metrics": best,
            "encoder_state_dict": encoder.state_dict(), "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if amp else None,
        }
        save_checkpoint(run_dir / "last.pt", state)
        if is_best:
            save_checkpoint(run_dir / "best.pt", state)
            atomic_json(run_dir / "best_validation.json", best)
        with (run_dir / "history.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_record.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(epoch_record)
    print(f"[DONE] {args.mode}; best validation = {best}. Checkpoints: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
