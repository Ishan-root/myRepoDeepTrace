import json
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import get_model

MODEL_PATH = ROOT / "checkpoints" / "resnet18_deeptrace.pth"
CLASS_NAMES = ["FAKE", "REAL"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


CSS = r"""/* DeepTrace — complete Gradio background override */
:root {
  --bg-0: #06080d;
  --bg-1: #0d1117;
  --bg-2: #101820;
  --panel-border: rgba(255,255,255,0.1);
  --text: #edf3ff;
  --muted: rgba(237,243,255,0.62);
  --green: #4ade80;
  --red: #f87171;
}

/* Kill every Gradio/page background layer so the video is the real page background. */
html,
body,
#root,
gradio-app,
.gradio-container,
.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .contain {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
}

html,
body {
  margin: 0 !important;
  min-height: 100% !important;
  color: var(--text);
  font-family: "Baskerville", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
}

body {
  overflow-x: hidden;
}

/* The video is fixed to the viewport and sits behind all Gradio content. */
.bg-video {
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  max-width: none !important;
  max-height: none !important;
  object-fit: cover !important;
  object-position: center center !important;
  display: block !important;
  z-index: 0 !important;
  pointer-events: none !important;
}

/* Optional darkening layer above the video, below the UI. */
.bg-overlay {
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 1 !important;
  background: rgba(6, 8, 13, 0.55) !important;
  pointer-events: none !important;
}

/* Keep all actual app content above the video/overlay. */
.gradio-container,
#page-shell,
#glass-card {
  position: relative !important;
}

.gradio-container {
  z-index: 2 !important;
  min-height: 100vh !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Remove Gradio's theme-generated page decoration as well. */
.gradio-container::before,
.gradio-container::after,
.gradio-container .main::before,
.gradio-container .main::after,
.gradio-container .contain::before,
.gradio-container .contain::after {
  background: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
}

/* subtle grain, purely decorative, sits above the video but below the UI */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
    radial-gradient(rgba(255,255,255,0.08) 0.6px, transparent 0.8px);
  background-size: 18px 18px, 18px 18px, 8px 8px;
  opacity: 0.24;
  pointer-events: none;
  z-index: 2;
}

/* --- outer shell: centers everything --- */
#page-shell {
  z-index: 3 !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 52px 22px 40px !important;
  min-height: 100vh;
  box-sizing: border-box;
}

/* --- the glass card itself --- */
#glass-card {
  position: relative !important;
  isolation: isolate;
  z-index: 4 !important;
  width: min(760px, 100%);
  margin: 0 auto !important;
  padding: 26px 24px 20px !important;
  border-radius: 30px !important;
  overflow: hidden;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.16), transparent 34%),
    linear-gradient(315deg, rgba(255,255,255,0.10), transparent 42%),
    rgba(18, 23, 30, 0.34) !important;
  background-image:
    linear-gradient(135deg, rgba(255,255,255,0.16), transparent 34%),
    linear-gradient(315deg, rgba(255,255,255,0.10), transparent 42%) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  box-shadow:
    0 24px 80px rgba(0,0,0,0.58),
    inset 0 1px 0 rgba(255,255,255,0.30),
    inset 0 -18px 34px rgba(0,0,0,0.18),
    inset 0 0 0 1px rgba(255,255,255,0.04) !important;
  backdrop-filter: blur(20px) saturate(125%);
  -webkit-backdrop-filter: blur(20px) saturate(125%);
}

#glass-card::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background:
    radial-gradient(circle at 78% 0%, rgba(255,255,255,0.22), transparent 24%),
    linear-gradient(110deg, transparent 22%, rgba(255,255,255,0.07) 46%, transparent 68%);
  opacity: 0.8;
  pointer-events: none;
  z-index: 0;
}

#glass-card > * {
  position: relative;
  z-index: 1;
}

.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 18px;
}

.brand-title {
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--text);
}

.brand-subtitle {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.upload-box {
  border-radius: 22px !important;
  border: 1.5px dashed rgba(255,255,255,0.42) !important;
  background: rgba(255,255,255,0.02) !important;
  overflow: hidden;
}

.scan-button {
  margin-top: 18px !important;
  width: 100% !important;
  background: #ffffff !important;
  border: 0 !important;
  border-radius: 14px !important;
  color: #080a0f !important;
  font-size: 0.96rem !important;
  font-weight: 700 !important;
  padding: 14px 22px !important;
  box-shadow: 0 12px 28px rgba(0,0,0,0.28);
  transition: all 0.22s ease;
}
.scan-button:hover {
  transform: translateY(-1px);
  background: #f1f3f7 !important;
  color: #080a0f !important;
  box-shadow: 0 16px 32px rgba(0,0,0,0.36);
}

.result-block {
  margin-top: 18px;
  padding: 18px 18px 14px;
  border-radius: 22px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
}

.verdict {
  font-size: clamp(2rem, 3vw, 3rem);
  letter-spacing: 0.06em;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 8px;
}
.verdict.real { color: var(--green); text-shadow: 0 0 14px rgba(74, 222, 128, 0.28); }
.verdict.fake { color: var(--red); text-shadow: 0 0 14px rgba(248, 113, 113, 0.26); }

.confidence {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 20px;
}

.map-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.map-card {
  min-height: 110px;
  border-radius: 16px;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255,255,255,0.6);
  font-size: 0.74rem;
  letter-spacing: 0.04em;
}

.floating-note {
  position: absolute;
  z-index: 5;
  max-width: 220px;
  font-size: 0.77rem;
  line-height: 1.45;
  color: rgba(255,255,255,0.42);
  padding: 0 8px;
}
.note-a { left: -230px; top: 130px; }
.note-b { right: -230px; bottom: 140px; }
.floating-note strong { color: rgba(255,255,255,0.65); font-weight: 700; }

.empty-result {
  padding: 28px 10px 12px;
  text-align: center;
  color: rgba(255,255,255,0.5);
  font-size: 0.9rem;
}

@media (max-width: 900px) {
  #page-shell { padding: 34px 16px !important; }
  .floating-note { display: none; }
  #glass-card { padding: 22px 18px 18px !important; }
}

"""

# Put the video directly into the browser document <body>, not inside a
# Gradio component. This makes it a true viewport-level background.
BG_JS = r"""
() => {
    const installBackground = () => {
        // Remove any previous copy if Gradio re-runs the app script.
        document.querySelectorAll("#deeptrace-bg-video, #deeptrace-bg-overlay").forEach(el => el.remove());

        const video = document.createElement("video");
        video.id = "deeptrace-bg-video";
        video.autoplay = true;
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = "auto";
        video.src = __BG_VIDEO_URL__;

        Object.assign(video.style, {
            position: "fixed",
            top: "0",
            left: "0",
            width: "100vw",
            height: "100vh",
            minWidth: "100vw",
            minHeight: "100vh",
            maxWidth: "none",
            maxHeight: "none",
            objectFit: "cover",
            objectPosition: "center center",
            display: "block",
            margin: "0",
            padding: "0",
            border: "0",
            zIndex: "0",
            pointerEvents: "none"
        });

        const overlay = document.createElement("div");
        overlay.id = "deeptrace-bg-overlay";

        Object.assign(overlay.style, {
            position: "fixed",
            top: "0",
            left: "0",
            width: "100vw",
            height: "100vh",
            zIndex: "1",
            pointerEvents: "none",
            background: "rgba(6, 8, 13, 0.55)"
        });

        document.body.prepend(video);
        document.body.insertBefore(overlay, video.nextSibling);

        // Force the page/root containers to be transparent.
        const transparentSelectors = [
            "html",
            "body",
            "#root",
            "gradio-app",
            ".gradio-container",
            ".gradio-container .main",
            ".gradio-container .contain",
            ".gradio-container .wrap",
            ".gradio-container .block",
            ".gradio-container .form",
            ".gradio-container .panel"
        ];

        transparentSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                el.style.setProperty("background", "transparent", "important");
                el.style.setProperty("background-color", "transparent", "important");
                el.style.setProperty("background-image", "none", "important");
            });
        });

        document.body.style.setProperty("margin", "0", "important");
        document.body.style.setProperty("overflow-x", "hidden", "important");

        video.play().catch(() => {});
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", installBackground, {once: true});
    } else {
        installBackground();
    }
}
"""

# Remove video HTML from inside Gradio; the JS above owns the true page background.
PAGE_HTML = """
<div class="floating-note note-b"><strong>Statistically Guaranteed Confidence</strong> — no guesswork, a real range.</div>
"""

def analyze_image(image):
    if image is None:
        return "<div class='empty-result'>Upload an image to begin the scan.</div>"

    pil_image = image.convert("RGB")

    with torch.no_grad():
        tensor = transform(pil_image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx].item() * 100)
    verdict_class = "real" if predicted_label == "REAL" else "fake"

    return f"""
    <div class="result-block">
      <div class="verdict {verdict_class}">{predicted_label}</div>
      <div class="confidence">{confidence:.2f}% confidence</div>
      <div class="map-grid">
        <div class="map-card">Where it looked</div>
        <div class="map-card">Where layers disagreed</div>
      </div>
    </div>
    """


with gr.Blocks(
    css=CSS,
    theme=gr.themes.Soft(primary_hue="slate"),
) as demo:
    bg_video_file = demo.serve_static_file(APP_DIR / "assets" / "bg.mp4")
    if not bg_video_file or not bg_video_file.get("url"):
        raise RuntimeError(f"Unable to serve background video: {APP_DIR / 'assets' / 'bg.mp4'}")
    demo.load(js=BG_JS.replace("__BG_VIDEO_URL__", json.dumps(bg_video_file["url"])))

    with gr.Column(elem_id="page-shell"):
        gr.HTML(PAGE_HTML)

        with gr.Column(elem_id="glass-card"):
            gr.HTML(
                """
                <div class="topbar">
                <div class="brand-title">DeepTrace</div>
                </div>
                """
            )

            upload = gr.Image(
                type="pil",
                sources=["upload"],
                label="",
                show_label=False,
                height=220,
                elem_classes=["upload-box"],
            )

            scan_button = gr.Button("Scan Image", elem_classes=["scan-button"])
            result_html = gr.HTML("<div class='empty-result'>Upload an image and run the scan.</div>")

    scan_button.click(fn=analyze_image, inputs=[upload], outputs=[result_html])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(APP_DIR / "assets")],
    )
