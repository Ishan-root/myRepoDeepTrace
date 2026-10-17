# src/selfcheck_explain.py
#
# Plain assert-style checks for the Grad-CAM (Phase 1) part of DeepTrace.
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
from gradcam import GradCAM, overlay, resized_rgb, to_png_base64

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "resnet18_deeptrace.pth"
SAMPLE_DIR = ROOT / "sample_images"
BASELINE_PATH = ROOT / "outputs" / "baseline_predictions.txt"
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
