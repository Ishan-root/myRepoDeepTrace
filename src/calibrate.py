# src/calibrate.py
#
# Split-conformal calibration for DeepTrace.
#
#   python src/calibrate.py               full run -> outputs/conformal_calibration.json
#                                                    outputs/conformal_report.txt
#   python src/calibrate.py --limit 200   smoke test on 200 images; writes to the
#                                         system temp folder, NOT to outputs/
#
# Steps
#   1. Score data/raw_sala_sorted/test with the CLEAN transform (no augmentation),
#      model in eval mode, float64 softmax of the logits.
#   2. Split it 50/50 into calibration / validation, stratified by class, seed 42.
#   3. Calibration half -> scores -> qhat per alpha -> JSON (with checkpoint fingerprint).
#   4. Validation half -> coverage report (never measured on the calibration half).

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from dataset import test_transform
from model import get_model
from conformal import (
    checkpoint_fingerprint,
    conformal_threshold,
    load_calibration,
    nonconformity_scores,
    prediction_set,
    probs_from_logits,
    save_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw_sala_sorted" / "test"
CHECKPOINT_PATH = ROOT / "checkpoints" / "resnet18_deeptrace.pth"
OUT_JSON = ROOT / "outputs" / "conformal_calibration.json"
OUT_REPORT = ROOT / "outputs" / "conformal_report.txt"

CLASS_NAMES = ["FAKE", "REAL"]
ALPHAS = [0.01, 0.05, 0.10, 0.20]
SEED = 42


def stratified_split(labels, seed, frac=0.5):
    """Per class: shuffle with a seeded RNG, first ceil(frac*n_c) -> calibration, rest -> validation."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    calib, valid = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(idx.size)]
        cut = int(np.ceil(frac * idx.size))
        calib.append(idx[:cut])
        valid.append(idx[cut:])
    return np.sort(np.concatenate(calib)), np.sort(np.concatenate(valid))


def stratified_limit(labels, limit, seed):
    """Seeded, class-balanced-ish subset of `limit` indices (for --limit smoke tests)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    picked = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        take = limit // len(classes) + (1 if i < limit % len(classes) else 0)
        picked.append(rng.choice(idx, size=min(take, idx.size), replace=False))
    return np.sort(np.concatenate(picked))


def score_dataset(model, loader, device):
    all_probs, all_labels = [], []
    done, total, t0 = 0, len(loader.dataset), time.time()
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(loader, start=1):
            logits = model(images.to(device))
            all_probs.append(probs_from_logits(logits))
            all_labels.append(labels.numpy())
            done += labels.size(0)
            if done == total or batch_i % 25 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  scored {done}/{total}  ({rate:.1f} img/s)", flush=True)
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(np.int64)


def evaluate(probs, labels, qhat):
    n = len(labels)
    covered = np.zeros(n, dtype=bool)
    sizes = np.zeros(n, dtype=np.int64)
    empty = np.zeros(n, dtype=bool)
    for i in range(n):
        s, was_empty = prediction_set(probs[i], qhat)
        covered[i] = labels[i] in s
        sizes[i] = len(s)
        empty[i] = was_empty
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        m = labels == c
        per_class[name] = (float(covered[m].mean()) if m.any() else float("nan"), int(m.sum()))
    return {
        "coverage": float(covered.mean()),
        "per_class": per_class,
        "singleton": float(np.mean(sizes == 1)),
        "both": float(np.mean(sizes == 2)),
        "empty_before_fix": float(empty.mean()),
        "avg_size": float(sizes.mean()),
    }


def fmt_q(q):
    return "+inf" if np.isinf(q) else f"{q:.6e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="smoke test on N images (writes to temp folder)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    if not DATA_DIR.is_dir():
        sys.exit(f"ERROR: test data not found at {DATA_DIR}")
    if not CHECKPOINT_PATH.is_file():
        sys.exit(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")

    if args.limit is not None:
        out_dir = Path(tempfile.gettempdir()) / "deeptrace_calibrate_smoke"
        out_json, out_report = out_dir / "conformal_calibration.json", out_dir / "conformal_report.txt"
        print(f"SMOKE TEST (--limit {args.limit}): outputs go to {out_dir}, outputs/ is not touched")
    else:
        out_json, out_report = OUT_JSON, OUT_REPORT

    dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=test_transform)
    expected = {name: i for i, name in enumerate(CLASS_NAMES)}
    if dataset.class_to_idx != expected:
        sys.exit(f"ERROR: class_to_idx is {dataset.class_to_idx}, expected {expected}")
    labels_all = np.asarray(dataset.targets, dtype=np.int64)
    print(f"Test split: {len(dataset)} images  "
          + ", ".join(f"{n}={int((labels_all == i).sum())}" for i, n in enumerate(CLASS_NAMES)))

    if args.limit is not None:
        keep = stratified_limit(labels_all, args.limit, SEED)
        dataset = Subset(dataset, keep.tolist())
        print(f"Using {len(dataset)} images")

    print("Fingerprinting checkpoint...")
    fingerprint = checkpoint_fingerprint(CHECKPOINT_PATH)

    model = get_model(pretrained=False)  # checkpoint overwrites every weight
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("Scoring...")
    probs, labels = score_dataset(model, loader, device)

    calib_idx, valid_idx = stratified_split(labels, SEED)
    assert not set(calib_idx.tolist()) & set(valid_idx.tolist()), "calibration/validation overlap"
    cal_scores = nonconformity_scores(probs[calib_idx], labels[calib_idx])
    v_probs, v_labels = probs[valid_idx], labels[valid_idx]

    acc_all = float((probs.argmax(1) == labels).mean())
    acc_val = float((v_probs.argmax(1) == v_labels).mean())

    extra = {
        "data_dir": "data/raw_sala_sorted/test",
        "limit": args.limit,
        "n_total_scored": int(len(labels)),
        "n_validation": int(len(valid_idx)),
        "accuracy_all_scored": acc_all,
        "accuracy_validation": acc_val,
    }
    save_calibration(out_json, cal_scores, SEED, CLASS_NAMES, ALPHAS, fingerprint, extra=extra)
    load_calibration(out_json)  # round-trip sanity: raises if anything is off

    lines = []
    lines.append("DeepTrace split-conformal calibration report")
    lines.append("=" * 60)
    lines.append("Method: split conformal, LAC score 1 - p[true], float64 softmax of logits")
    lines.append(f"Data: data/raw_sala_sorted/test{'  (LIMIT ' + str(args.limit) + ')' if args.limit else ''}")
    lines.append(f"Checkpoint sha256: {fingerprint['sha256']}  size: {fingerprint['size']}")
    lines.append(f"Scored: {len(labels)}  calibration: {len(calib_idx)}  validation: {len(valid_idx)}  seed: {SEED}")
    for c, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name}: calibration {int((labels[calib_idx] == c).sum())}, "
                     f"validation {int((v_labels == c).sum())}")
    lines.append(f"Argmax accuracy: all scored {acc_all * 100:.2f}%, validation {acc_val * 100:.2f}%")
    lines.append(f"Calibration scores: min {cal_scores.min():.3e}, median {np.median(cal_scores):.3e}, "
                 f"max {cal_scores.max():.3e}, exactly 0: {int((cal_scores == 0).sum())}")
    lines.append("")
    lines.append("Validation half (never used for calibration):")
    for a in ALPHAS:
        q = conformal_threshold(cal_scores, a)
        r = evaluate(v_probs, v_labels, q)
        pc = r["per_class"]
        lines.append(f"alpha={a:.2f}  target coverage {(1 - a) * 100:.0f}%   qhat={fmt_q(q)}")
        lines.append(f"    marginal coverage : {r['coverage'] * 100:.2f}%")
        for name in CLASS_NAMES:
            cov, cnt = pc[name]
            lines.append(f"    coverage {name:<5}    : {cov * 100:.2f}%  (n={cnt})")
        lines.append(f"    singleton sets    : {r['singleton'] * 100:.2f}%")
        lines.append(f"    both-class sets   : {r['both'] * 100:.2f}%")
        lines.append(f"    empty before fix  : {r['empty_before_fix'] * 100:.2f}%  (replaced by {{argmax}})")
        lines.append(f"    average set size  : {r['avg_size']:.4f}")
    lines.append("")
    lines.append("Note: coverage holds only for images exchangeable with this test split.")
    report = "\n".join(lines) + "\n"

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_report}")


if __name__ == "__main__":
    main()
