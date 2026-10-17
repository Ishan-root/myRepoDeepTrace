# src/selfcheck_explain.py
#
# Plain assert-style checks for DeepTrace's Grad-CAM (Phase 1) and
# conformal prediction (Phase 2) code.
# Run from the repo root:
#     python src/selfcheck_explain.py
#     python src/selfcheck_explain.py --device cpu      (force CPU)
#
# Every check runs even if an earlier one fails; the script exits with code 1
# if anything failed. Preview overlays are written OUTSIDE the repo (system temp
# folder) so they never end up in git.

import argparse
import copy
import hashlib
import re
import sys
import tempfile
import traceback
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import get_model
from gradcam import DISAGREE_LUT, GradCAM, disagreement_map, overlay, resized_rgb, to_png_base64
from conformal import (
    conformal_k,
    conformal_threshold,
    fingerprint_matches,
    load_calibration,
    prediction_set,
    probs_from_logits,
    qhat_for,
    save_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "resnet18_deeptrace.pth"
SAMPLE_DIR = ROOT / "sample_images"
BASELINE_PATH = ROOT / "outputs" / "baseline_predictions.txt"
CALIBRATION_PATH = ROOT / "outputs" / "conformal_calibration.json"
PREVIEW_DIR = Path(tempfile.gettempdir()) / "deeptrace_gradcam_preview"

CLASS_NAMES = ["FAKE", "REAL"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LAYER_NAMES = ["layer2", "layer3", "layer4"]

# Same clean pipeline as predict.py / test_transform in dataset.py.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

RESULTS = []  # (name, passed, detail)


def record(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def run_check(name, fn):
    try:
        detail = fn()
        record(name, True, detail or "")
    except AssertionError as e:
        record(name, False, str(e) or "assertion failed")
    except Exception as e:  # unexpected crash counts as failure, with traceback
        traceback.print_exc()
        record(name, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_model(device):
    model = get_model(pretrained=False)  # checkpoint overwrites every weight
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sample_images():
    if not SAMPLE_DIR.is_dir():
        return []
    seen, files = set(), []
    for p in sorted(SAMPLE_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(p)
    return files


def to_tensor(pil, device):
    return transform(pil.convert("RGB")).unsqueeze(0).to(device)


def plain_probs(model, x):
    with torch.no_grad():
        return F.softmax(model(x), dim=1)[0]


def assert_cam_ok(cam, label):
    assert isinstance(cam, np.ndarray), f"{label}: not a numpy array"
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE), f"{label}: shape {cam.shape}"
    assert cam.dtype == np.float32, f"{label}: dtype {cam.dtype}"
    assert np.all(np.isfinite(cam)), f"{label}: contains NaN/Inf"
    assert cam.min() >= 0.0 and cam.max() <= 1.0, f"{label}: range [{cam.min()}, {cam.max()}]"


def pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def read_baseline():
    """Parse outputs/baseline_predictions.txt (PowerShell may write UTF-16)."""
    raw = BASELINE_PATH.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-16")
    out, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Image:"):
            current = PureWindowsPath(line[len("Image:"):].strip()).name.lower()
        m = re.match(r"Raw probabilities -> FAKE: ([\d.]+)%, REAL: ([\d.]+)%", line)
        if m and current:
            out[current] = (m.group(1), m.group(2))
            current = None
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)
    print("torch", torch.__version__)

    assert CHECKPOINT_PATH.is_file(), f"checkpoint not found: {CHECKPOINT_PATH}"
    model = load_model(device)
    cam_engine = GradCAM(model)
    images = sample_images()
    print(f"Sample images found: {len(images)}\n")

    hash_before = state_hash(model)
    results = {}  # filename -> GradCAMResult

    # 1 + 2: shapes/ranges and probability equality, for every sample image
    def check_shapes_and_probs():
        assert images, f"no images in {SAMPLE_DIR}"
        worst = 0.0
        for p in images:
            pil = Image.open(p)
            x = to_tensor(pil, device)
            res = cam_engine(x)
            results[p.name] = (res, pil.convert("RGB"))
            for n in LAYER_NAMES:
                assert n in res.cams, f"{p.name}: missing cam {n}"
                assert_cam_ok(res.cams[n], f"{p.name}/{n}")
            ref = plain_probs(model, x)
            diff = float((res.probs[0] - ref).abs().max().item())
            worst = max(worst, diff)
            assert diff <= 1e-6, f"{p.name}: Grad-CAM probs differ from no_grad probs by {diff:.3e}"
            assert res.class_idx == int(torch.argmax(ref).item()), f"{p.name}: class_idx != argmax"
        return f"{len(images)} images, max prob diff {worst:.2e}"

    run_check("1+2. shapes/dtype/range OK and probs equal no_grad pass (<=1e-6)", check_shapes_and_probs)

    # 3: model not mutated
    def check_not_mutated():
        assert state_hash(model) == hash_before, "state_dict hash changed"
        assert model.training is False, "model.training became True"
        with_grad = [n for n, prm in model.named_parameters() if prm.grad is not None]
        assert not with_grad, f"param.grad populated for {len(with_grad)} params, e.g. {with_grad[:3]}"
        leaked = [n for n, m in model.named_modules()
                  if len(m._forward_hooks) or len(m._forward_pre_hooks) or len(m._backward_hooks)]
        assert not leaked, f"hooks left on modules: {leaked[:5]}"
        return "weights hash identical, eval mode, no grads, no hooks"

    run_check("3. model not mutated", check_not_mutated)

    # 4: determinism
    def check_determinism():
        p = images[0]
        x = to_tensor(Image.open(p), device)
        a, b = cam_engine(x), cam_engine(x)
        for n in LAYER_NAMES:
            d = float(np.abs(a.cams[n] - b.cams[n]).max())
            assert np.array_equal(a.cams[n], b.cams[n]), f"{p.name}/{n}: max diff {d:.3e} between two calls"
        return p.name

    run_check("4. determinism (two calls identical)", check_determinism)

    # 5: different images -> different maps
    def check_different():
        names = list(results.keys())
        assert len(names) >= 2, "need at least 2 sample images"
        same = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if np.array_equal(results[names[i]][0].cams["layer4"], results[names[j]][0].cams["layer4"]):
                    same.append((names[i], names[j]))
        assert not same, f"identical layer4 maps for {same[:3]}"
        flat = [n for n in names if results[n][0].cams["layer4"].max() == 0.0]
        detail = f"{len(names)} images, all pairwise different"
        if flat:
            detail += f"; NOTE all-zero layer4 map for: {flat}"
        return detail

    run_check("5. different images give different maps", check_different)

    # 6: randomization sanity test (re-initialized fc)
    def check_randomization():
        rand_model = copy.deepcopy(model)
        torch.manual_seed(0)
        rand_model.fc.reset_parameters()
        rand_model.eval()
        rand_engine = GradCAM(rand_model)
        corrs = []
        for p in images:
            res = results[p.name][0]
            x = to_tensor(Image.open(p), device)
            rres = rand_engine(x, class_idx=res.class_idx)  # same target class
            c = pearson(res.cams["layer4"], rres.cams["layer4"])
            corrs.append(c)
        valid = [c for c in corrs if np.isfinite(c)]
        assert valid, "all correlations undefined (flat maps)"
        mean_c = float(np.mean(valid))
        assert mean_c < 0.8, f"mean Pearson {mean_c:.3f} >= 0.8: maps barely depend on the trained head"
        assert max(valid) < 0.99, f"some map is ~identical after randomizing fc (r={max(valid):.4f})"
        assert state_hash(model) == hash_before, "original model changed during randomization test"
        return f"mean r={mean_c:.3f}, min={min(valid):.3f}, max={max(valid):.3f}"

    run_check("6. randomization test (fc re-init changes layer4 map)", check_randomization)

    # 7: odd image modes and sizes
    def check_odd_images():
        rng = np.random.default_rng(0)
        rgb = Image.fromarray(rng.integers(0, 256, (300, 200, 3), dtype=np.uint8))
        cases = {
            "grayscale L": rgb.convert("L"),
            "RGBA": rgb.convert("RGBA"),
            "palette P": rgb.convert("P"),
            "1x1": Image.new("RGB", (1, 1), (120, 30, 200)),
            "1-bit": rgb.convert("1"),
        }
        for label, im in cases.items():
            res = cam_engine(to_tensor(im.convert("RGB"), device))
            for n in LAYER_NAMES:
                assert_cam_ok(res.cams[n], f"{label}/{n}")
            ov = overlay(resized_rgb(im), res.cams["layer4"])
            assert ov.size == (IMAGE_SIZE, IMAGE_SIZE) and ov.mode == "RGB", f"{label}: bad overlay"
        return ", ".join(cases)

    run_check("7. grayscale/RGBA/palette/1x1/1-bit images", check_odd_images)

    # Extra: overlay helper + base64 encode
    def check_overlay():
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for name, (res, pil) in results.items():
            ov = overlay(resized_rgb(pil), res.cams["layer4"])
            assert ov.size == (IMAGE_SIZE, IMAGE_SIZE) and ov.mode == "RGB"
            b64 = to_png_base64(ov)
            assert len(b64) > 100
            label = CLASS_NAMES[res.class_idx]
            ov.save(PREVIEW_DIR / f"{Path(name).stem}__{label}.png")
        return f"overlays written to {PREVIEW_DIR}"

    run_check("extra. overlay renders and encodes", check_overlay)

    # Extra: regression vs Phase 0 baseline (probabilities to 2 decimals)
    def check_baseline():
        assert BASELINE_PATH.is_file(), f"missing {BASELINE_PATH}"
        base = read_baseline()
        assert base, "could not parse any entry from baseline file"
        mismatches, checked = [], 0
        for name, (res, _) in results.items():
            key = name.lower()
            if key not in base:
                mismatches.append(f"{name}: not in baseline")
                continue
            got = (f"{res.probs[0, 0].item() * 100:.2f}", f"{res.probs[0, 1].item() * 100:.2f}")
            if got != base[key]:
                mismatches.append(f"{name}: got FAKE {got[0]} REAL {got[1]}, baseline {base[key]}")
            checked += 1
        assert not mismatches, "; ".join(mismatches)
        return f"{checked} images match baseline_predictions.txt"

    run_check("extra. Grad-CAM pass reproduces Phase 0 baseline", check_baseline)


    # ======================= Phase 2: conformal =======================

    # 8: conformal_threshold unit tests
    def check_threshold():
        inf = float("inf")
        # n=1: k = ceil(2*0.9) = 2 > 1 -> +inf
        assert conformal_threshold([0.3], 0.1) == inf, "n=1 should give +inf"
        # n=8: k = ceil(9*0.9) = ceil(8.1) = 9 > 8 -> +inf
        assert conformal_threshold(np.linspace(0, 1, 8), 0.1) == inf, "n=8 should give +inf"
        # n=9: k = ceil(10*0.9) = 9 <= 9 -> largest score (finite!)
        s9 = np.arange(1, 10) / 10.0
        assert conformal_threshold(s9, 0.1) == 0.9, f"n=9 got {conformal_threshold(s9, 0.1)}"
        # n=19, alpha=0.1: k = ceil(20*0.9) = 18 -> 18th smallest
        s19 = np.arange(1, 20) / 100.0             # 0.01 .. 0.19
        assert conformal_k(19, 0.1) == 18
        assert conformal_threshold(s19, 0.1) == 0.18, f"n=19 got {conformal_threshold(s19, 0.1)}"
        # unsorted input is sorted internally
        assert conformal_threshold(s19[::-1].copy(), 0.1) == 0.18
        # float-noise guard: (n+1)(1-a) that is an exact integer must not be bumped up
        for n, a, k in [(19, 0.1, 18), (99, 0.1, 90), (99, 0.05, 95), (99, 0.01, 99), (999, 0.2, 800), (4, 0.2, 4)]:
            assert conformal_k(n, a) == k, f"k(n={n}, alpha={a}) = {conformal_k(n, a)}, expected {k}"
        # monotone: smaller alpha -> larger-or-equal qhat
        rng = np.random.default_rng(1)
        sc = rng.random(500)
        qs = [conformal_threshold(sc, a) for a in [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]]
        assert all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)), f"not monotone: {qs}"
        # ties
        assert conformal_threshold([0.2] * 50, 0.1) == 0.2
        assert conformal_threshold([0.0] * 30 + [0.5] * 20, 0.1) == 0.5
        # bad input
        for bad in [([], 0.1), ([0.1], 0.0), ([0.1], 1.0), ([float("nan")], 0.1)]:
            try:
                conformal_threshold(*bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"no ValueError for {bad}")
        return "n=1/n=8 -> inf, n=9 -> max score, n=19 hand calc, k float guard, monotone, ties, bad input"

    run_check("8. conformal_threshold unit tests", check_threshold)

    # 9: prediction_set unit tests
    def check_prediction_set():
        assert prediction_set([0.9, 0.1], 0.2) == ([0], False)
        assert prediction_set([0.1, 0.9], 0.2) == ([1], False)
        assert prediction_set([0.6, 0.4], 0.7) == ([0, 1], False)
        assert prediction_set([0.6, 0.4], float("inf")) == ([0, 1], False)
        assert prediction_set([0.6, 0.4], 0.01) == ([0], True)    # empty -> argmax
        assert prediction_set([0.3, 0.7], 0.0) == ([1], True)
        assert prediction_set([1.0, 0.0], 0.0) == ([0], False)    # boundary is inclusive (<=)
        p = probs_from_logits([[2.0, -1.0], [0.0, 0.0]])
        assert p.dtype == np.float64 and np.allclose(p.sum(1), 1.0)
        assert np.allclose(p[1], [0.5, 0.5])
        big = probs_from_logits([1000.0, -1000.0])                 # no overflow
        assert np.all(np.isfinite(big)) and big[0] == 1.0
        return "singletons, both, inf, empty->argmax, inclusive boundary, stable softmax"

    run_check("9. prediction_set / probs_from_logits unit tests", check_prediction_set)

    # 10: synthetic coverage simulation
    def check_simulation():
        rng = np.random.default_rng(123)
        lines = []
        for alpha in [0.1, 0.05, 0.2]:
            for n in [50, 500]:
                covs = []
                for _ in range(400):
                    cal = rng.beta(0.5, 3.0, size=n)
                    test = rng.beta(0.5, 3.0, size=2000)
                    q = conformal_threshold(cal, alpha)
                    covs.append(np.mean(test <= q))
                mean_cov = float(np.mean(covs))
                upper = 1 - alpha + 1.0 / (n + 1)
                assert mean_cov >= 1 - alpha - 0.005, f"alpha={alpha} n={n}: coverage {mean_cov:.4f} too low"
                assert mean_cov <= upper + 0.005, f"alpha={alpha} n={n}: coverage {mean_cov:.4f} above {upper:.4f}"
                lines.append(f"a={alpha} n={n}: {mean_cov:.3f}")
        # end-to-end binary classifier simulation, including the empty-set fix
        covs = []
        for _ in range(200):
            def draw(m):
                p_real = rng.beta(0.6, 0.6, size=m)
                y = (rng.random(m) < p_real).astype(int)          # labels drawn from the probs
                probs = np.stack([1 - p_real, p_real], axis=1)
                return probs, y
            cp, cy = draw(400)
            q = conformal_threshold(1.0 - cp[np.arange(400), cy], 0.1)
            tp, ty = draw(1000)
            covs.append(np.mean([ty[i] in prediction_set(tp[i], q)[0] for i in range(1000)]))
        e2e = float(np.mean(covs))
        assert e2e >= 0.9 - 0.005, f"end-to-end coverage {e2e:.4f} below 0.895"
        return "; ".join(lines) + f"; end-to-end a=0.1: {e2e:.3f}"

    run_check("10. synthetic coverage simulation", check_simulation)

    # 11: save/load round trip
    def check_roundtrip():
        rng = np.random.default_rng(7)
        scores = np.concatenate([rng.random(57) ** 3, [0.0, 0.0, 1e-300, 0.5]])
        fp = {"size": 123, "sha256": "ab" * 32}
        alphas = [0.01, 0.05, 0.10, 0.20]
        tmp = Path(tempfile.gettempdir()) / "deeptrace_selfcheck_calib.json"
        try:
            save_calibration(tmp, scores, 42, CLASS_NAMES, alphas, fp, extra={"note": "test"})
            loaded = load_calibration(tmp)
        finally:
            if tmp.exists():
                tmp.unlink()
        assert np.array_equal(loaded["scores"], np.sort(scores)), "scores changed in round trip"
        assert loaded["n"] == len(scores) and loaded["seed"] == 42
        assert loaded["class_names"] == CLASS_NAMES and loaded["checkpoint"] == fp
        for a in alphas:
            assert loaded["qhat"][a] == conformal_threshold(scores, a), f"qhat mismatch at {a}"
        assert loaded["qhat"][0.01] == float("inf"), "n=61, alpha=0.01 should be +inf and survive JSON"
        return "scores bit-exact, qhat (incl. +inf) restored"

    run_check("11. calibration JSON round trip", check_roundtrip)

    # 12: the real calibration file
    def check_real_calibration():
        assert CALIBRATION_PATH.is_file(), f"missing {CALIBRATION_PATH} (run: python src/calibrate.py)"
        cal = load_calibration(CALIBRATION_PATH)   # validates qhat vs scores
        assert cal["class_names"] == CLASS_NAMES, f"class_names {cal['class_names']}"
        assert fingerprint_matches(cal, CHECKPOINT_PATH), "checkpoint fingerprint does not match"
        assert not (cal.get("extra") or {}).get("limit"), "this is a --limit smoke-test file, not a full run"
        q = qhat_for(cal, 0.10)
        return f"n={cal['n']}, qhat(0.10)={q:.6e}, fingerprint matches"

    run_check("12. outputs/conformal_calibration.json valid and matches checkpoint", check_real_calibration)

    # 13: float64 conformal probs agree with the float32 probs used for the verdict
    def check_prob_consistency():
        worst = 0.0
        for name, (res, _) in results.items():
            p64 = probs_from_logits(res.logits)[0]
            p32 = res.probs[0].detach().cpu().double().numpy()
            worst = max(worst, float(np.abs(p64 - p32).max()))
            assert int(np.argmax(p64)) == res.class_idx, f"{name}: argmax differs"
        assert worst <= 1e-6, f"float64 vs float32 probs differ by {worst:.2e}"
        return f"max diff {worst:.2e}, argmax identical"

    run_check("13. conformal probs consistent with verdict probs", check_prob_consistency)

    # 14: layer-disagreement map (tile 2)
    def check_disagreement():
        for name, (res, pil) in results.items():
            d = disagreement_map(res.cams)
            assert_cam_ok(d, f"{name}/disagreement")
            ov = overlay(resized_rgb(pil), d, lut=DISAGREE_LUT)
            assert ov.size == (IMAGE_SIZE, IMAGE_SIZE) and ov.mode == "RGB"
        same = {n: np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, np.float32) for n in LAYER_NAMES}
        assert disagreement_map(same).max() == 0.0, "identical layer maps should give an all-zero map"
        return f"{len(results)} images, identical layers -> zero map"

    run_check("14. layer-disagreement map", check_disagreement)

    # Final: still not mutated after everything
    run_check("3b. model still not mutated after all checks", check_not_mutated)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n================ SUMMARY ================")
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{passed}/{len(RESULTS)} checks passed")
    print("=========================================")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
