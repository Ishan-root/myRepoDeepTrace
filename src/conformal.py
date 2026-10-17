# src/conformal.py
#
# Self-contained split conformal prediction (LAC score) for DeepTrace.
# Imports only stdlib + numpy, so it works both as `from conformal import ...`
# (scripts in src/) and `from src.conformal import ...` (the Gradio app).
#
# Method
#   score of a labeled example:   s = 1 - p[true_class]
#   threshold for miscoverage a:  k = ceil((n + 1) * (1 - a))
#                                 qhat = +inf if k > n else sorted_scores[k - 1]
#   prediction set:               { y : 1 - p[y] <= qhat }, empty -> {argmax}
#
# Probabilities are computed in float64 from the raw logits (probs_from_logits)
# both at calibration time and in the app, so very confident predictions do
# not all collapse to exactly 1.0 (which would create massive ties at 0).

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

CALIBRATION_FORMAT_VERSION = 1
_K_TOL = 1e-9  # guards ceil() against float noise, e.g. 20 * 0.9 = 18.000000000000004


def probs_from_logits(logits):
    """Numerically stable float64 softmax over the last axis. Accepts lists, numpy or torch (cpu/cuda)."""
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().double().numpy()
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nonconformity_scores(probs, labels):
    """LAC scores 1 - p[true] for arrays probs (n, K) and labels (n,)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or labels.ndim != 1 or probs.shape[0] != labels.shape[0]:
        raise ValueError(f"bad shapes: probs {probs.shape}, labels {labels.shape}")
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_k(n, alpha):
    """Rank k = ceil((n + 1) * (1 - alpha)), 1-based."""
    return int(math.ceil((n + 1) * (1.0 - alpha) - _K_TOL))


def conformal_threshold(scores, alpha):
    """Split-conformal threshold qhat. Returns float('inf') when n is too small for this alpha."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    if not np.all(np.isfinite(s)):
        raise ValueError("calibration scores must be finite")
    k = conformal_k(n, alpha)
    if k > n:
        return float("inf")
    return float(s[k - 1])


def prediction_set(probs, qhat):
    """
    probs: 1-D array of class probabilities for ONE image.
    Returns (classes, was_empty): sorted list of class indices, and whether the
    raw set was empty (and therefore replaced by {argmax}).
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    if p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("probs must be a non-empty finite vector")
    q = float(qhat)
    classes = [int(y) for y in range(p.size) if (1.0 - p[y]) <= q]
    if not classes:
        return [int(np.argmax(p))], True
    return classes, False


# ---------------------------------------------------------------------------
# checkpoint fingerprint
# ---------------------------------------------------------------------------

def checkpoint_fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"size": os.path.getsize(path), "sha256": h.hexdigest()}


# ---------------------------------------------------------------------------
# save / load (strict JSON: +inf is stored as null)
# ---------------------------------------------------------------------------

def _qhat_to_json(q):
    return None if math.isinf(q) else float(q)


def _qhat_from_json(q):
    return float("inf") if q is None else float(q)


def save_calibration(path, scores, seed, class_names, alphas, fingerprint, extra=None):
    """Writes the calibration file. qhat values are recomputed here from the scores."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    data = {
        "format_version": CALIBRATION_FORMAT_VERSION,
        "method": "split conformal, LAC score s = 1 - p[true], probs = float64 softmax of logits",
        "n": int(s.size),
        "seed": int(seed),
        "class_names": list(class_names),
        "alphas": [
            {"alpha": float(a), "qhat": _qhat_to_json(conformal_threshold(s, a))}
            for a in alphas
        ],
        "checkpoint": {"size": int(fingerprint["size"]), "sha256": str(fingerprint["sha256"])},
        "scores": [float(x) for x in s],
    }
    if extra:
        data["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, allow_nan=False)
    os.replace(tmp, path)  # atomic: never leaves a half-written file
    return data


def load_calibration(path):
    """Loads and validates the calibration file. Raises ValueError on anything suspicious."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format_version") != CALIBRATION_FORMAT_VERSION:
        raise ValueError(f"unsupported calibration format: {data.get('format_version')}")
    scores = np.asarray(data["scores"], dtype=np.float64)
    if scores.size != data["n"] or scores.size == 0:
        raise ValueError("calibration file: n does not match number of scores")
    if not np.all(np.isfinite(scores)) or np.any(np.diff(scores) < 0):
        raise ValueError("calibration file: scores must be finite and sorted")
    table = {}
    for row in data["alphas"]:
        a = float(row["alpha"])
        stored = _qhat_from_json(row["qhat"])
        recomputed = conformal_threshold(scores, a)
        if stored != recomputed:
            raise ValueError(f"calibration file: stored qhat for alpha={a} does not match the scores")
        table[a] = stored
    data["scores"] = scores
    data["qhat"] = table
    return data


def qhat_for(calibration, alpha):
    """qhat for any alpha, computed exactly from the stored scores."""
    return conformal_threshold(calibration["scores"], alpha)


def fingerprint_matches(calibration, checkpoint_path):
    fp = checkpoint_fingerprint(checkpoint_path)
    ck = calibration.get("checkpoint", {})
    return fp["size"] == ck.get("size") and fp["sha256"] == ck.get("sha256")
