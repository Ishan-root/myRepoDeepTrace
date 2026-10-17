# src/selfcheck_app.py
#
# Phase 3 checks for app/DeepTrace_app_explain.py. Run from the repo root:
#     python src/selfcheck_app.py
#
# Imports the app (without launching it), calls its scan/analyze functions
# directly, injects failures, and finally starts the server briefly on a free
# port to confirm the page is served. Exits with code 1 if anything fails.

import importlib.util
import json
import socket
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from selfcheck_explain import read_baseline, sample_images

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "DeepTrace_app_explain.py"
BASELINE_PATH = ROOT / "outputs" / "baseline_predictions.txt"
B64_MARK = "data:image/png;base64,"
CRASH_MARK = "Something went wrong"

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def run_check(name, fn):
    try:
        record(name, True, fn() or "")
    except AssertionError as e:
        record(name, False, str(e) or "assertion failed")
    except Exception as e:
        traceback.print_exc()
        record(name, False, f"{type(e).__name__}: {e}")


def import_app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("deeptrace_app_explain", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    print("Importing", APP_PATH)
    app = import_app()
    print("Device:", app.device)
    print(f"Conformal alpha={app.CONFORMAL_ALPHA}, qhat={app.CONFORMAL_QHAT}\n")
    images = sample_images()

    # A0: calibration actually loaded (otherwise the app silently runs without sets)
    def check_calibration_loaded():
        assert app.CONFORMAL_QHAT is not None, "app started with prediction sets disabled (see WARNING above)"
        return f"qhat={app.CONFORMAL_QHAT:.6e} for {app.SET_LABEL}"

    run_check("A0. calibration loaded at startup", check_calibration_loaded)

    # A1: every sample image + odd inputs
    def check_inputs():
        assert images, "no sample images"
        rng = np.random.default_rng(0)
        rgb = Image.fromarray(rng.integers(0, 256, (300, 200, 3), dtype=np.uint8))
        odd = {
            "grayscale": rgb.convert("L"),
            "RGBA": rgb.convert("RGBA"),
            "palette": rgb.convert("P"),
            "1x1": Image.new("RGB", (1, 1), (10, 200, 30)),
            "large 4000x3000": Image.new("RGB", (4000, 3000), (90, 120, 150)),
        }
        cases = [(p.name, Image.open(p)) for p in images] + list(odd.items())
        for name, im in cases:
            html = app.analyze_image(im)
            assert CRASH_MARK not in html, f"{name}: scan crashed"
            assert html.count(B64_MARK) == 2, f"{name}: expected 2 map images, found {html.count(B64_MARK)}"
            assert app.SET_LABEL in html, f"{name}: no prediction set line in HTML"
        none_html = app.analyze_image(None)
        assert "Upload an image" in none_html, "None input not handled"
        return f"{len(images)} samples + {', '.join(odd)} + None"

    run_check("A1. analyze_image on samples and odd inputs", check_inputs)

    # A2: regression against Phase 0 baseline
    def check_regression():
        base = read_baseline()
        bad = []
        for p in images:
            r = app.scan(Image.open(p))
            key = p.name.lower()
            if key not in base:
                bad.append(f"{p.name}: not in baseline")
                continue
            got = (f"{r['probs'][0] * 100:.2f}", f"{r['probs'][1] * 100:.2f}")
            want_label = "FAKE" if float(base[key][0]) > float(base[key][1]) else "REAL"
            if got != base[key]:
                bad.append(f"{p.name}: probs {got} vs baseline {base[key]}")
            if r["label"] != want_label:
                bad.append(f"{p.name}: label {r['label']} vs baseline {want_label}")
        assert not bad, "; ".join(bad)
        return f"{len(images)} images: labels and probabilities match to 2 decimals"

    run_check("A2. regression vs baseline_predictions.txt", check_regression)

    # A3: a singleton set always equals the plain verdict
    def check_set_consistency():
        sizes = {1: 0, 2: 0}
        for p in images:
            r = app.scan(Image.open(p))
            assert r["pred_set"], f"{p.name}: no prediction set"
            sizes[len(r["pred_set"])] += 1
            if len(r["pred_set"]) == 1:
                assert r["pred_set"][0] == r["label"], f"{p.name}: set {r['pred_set']} != verdict {r['label']}"
        return f"singleton {sizes[1]}, UNCERTAIN {sizes[2]} (of {len(images)} samples)"

    run_check("A3. prediction set consistent with verdict", check_set_consistency)

    probe = Image.open(images[0]) if images else Image.new("RGB", (64, 64))
    reference = app.scan(probe)

    # A4: calibration missing / mismatched -> plain verdict, no crash
    def check_missing_calibration():
        missing = Path(tempfile.gettempdir()) / "deeptrace_does_not_exist.json"
        assert app.load_conformal_qhat(missing) is None, "missing file did not disable sets"
        # mismatched checkpoint fingerprint
        cal = json.loads(Path(app.CALIBRATION_PATH).read_text(encoding="utf-8"))
        cal["checkpoint"]["sha256"] = "0" * 64
        bad = Path(tempfile.gettempdir()) / "deeptrace_bad_fingerprint.json"
        bad.write_text(json.dumps(cal), encoding="utf-8")
        try:
            assert app.load_conformal_qhat(bad) is None, "mismatched fingerprint did not disable sets"
        finally:
            bad.unlink()
        saved = app.CONFORMAL_QHAT
        app.CONFORMAL_QHAT = None
        try:
            r = app.scan(probe)
            html = app.analyze_image(probe)
        finally:
            app.CONFORMAL_QHAT = saved
        assert r["pred_set"] is None and r["label"] == reference["label"], "verdict changed without calibration"
        assert CRASH_MARK not in html and r["label"] in html and B64_MARK in html, "bad fallback HTML"
        assert app.SET_LABEL not in html, "set line shown although disabled"
        return "missing file and wrong fingerprint both disable sets; verdict + heatmap still shown"

    run_check("A4. failure injection: calibration missing/mismatched", check_missing_calibration)

    # A5: Grad-CAM raises -> verdict still correct
    def check_gradcam_failure():
        saved = app.cam_engine

        def broken(*args, **kwargs):
            raise RuntimeError("injected Grad-CAM failure")

        app.cam_engine = broken
        try:
            r = app.scan(probe)
            html = app.analyze_image(probe)
        finally:
            app.cam_engine = saved
        assert r["heatmap_b64"] is None, "heatmap present despite failure"
        assert r["label"] == reference["label"], "verdict changed"
        assert np.allclose(r["probs"], reference["probs"], atol=1e-6), "probs changed"
        assert r["pred_set"] == reference["pred_set"], "prediction set changed"
        assert CRASH_MARK not in html and B64_MARK not in html and "Heatmap unavailable" in html
        return "verdict, probs and set unchanged; note shown"

    run_check("A5. failure injection: Grad-CAM raises", check_gradcam_failure)

    # A6: prediction_set raises -> verdict still shown
    def check_conformal_failure():
        saved = app.prediction_set

        def broken(*args, **kwargs):
            raise RuntimeError("injected conformal failure")

        app.prediction_set = broken
        try:
            html = app.analyze_image(probe)
        finally:
            app.prediction_set = saved
        assert CRASH_MARK not in html and reference["label"] in html, "verdict missing"
        assert "Prediction set unavailable" in html and B64_MARK in html
        return "verdict + heatmap shown, set skipped with note"

    run_check("A6. failure injection: prediction set raises", check_conformal_failure)

    # A7: model untouched after all of the above
    def check_model_clean():
        assert app.model.training is False, "model left in training mode"
        assert all(p.grad is None for p in app.model.parameters()), "param.grad populated"
        leaked = [n for n, m in app.model.named_modules() if len(m._forward_hooks)]
        assert not leaked, f"hooks left: {leaked[:3]}"
        return "eval mode, no grads, no hooks"

    run_check("A7. model clean after scans", check_model_clean)

    # A8: server starts and serves the page
    def check_server():
        port = free_port()
        app.demo.launch(server_name="127.0.0.1", server_port=port, share=False, quiet=True,
                        prevent_thread_lock=True, allowed_paths=[str(app.APP_DIR / "assets")])
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30).read().decode("utf-8", "replace")
        finally:
            app.demo.close()
        assert "verdict.uncertain" in page and "glass-card" in page, "page served without the app CSS"
        return f"served on temporary port {port}, CSS present"

    run_check("A8. server starts and serves the page", check_server)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n============== APP SUMMARY ==============")
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{passed}/{len(RESULTS)} checks passed")
    print("=========================================")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
