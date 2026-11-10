from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import csv
import tempfile
from collections import Counter

import gradio as gr
from PIL import Image, ImageOps, ImageDraw, ImageFont
from ultralytics import YOLO

APP_TITLE = "Blood Cancer Cell Analysis Studio"
DETECT_MODEL_PATH = Path("detectorbc.pt")
CLASSIFY_MODEL_PATH = Path("classifymodel.pt")


def load_model(path: Path, label: str) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(f"{label} model not found: {path.resolve()}")
    return YOLO(str(path))


detector = load_model(DETECT_MODEL_PATH, "Detector")
classifier = load_model(CLASSIFY_MODEL_PATH, "Classifier")


def palette() -> List[Tuple[int, int, int]]:
    return [
        (15, 118, 110),
        (30, 64, 175),
        (190, 24, 93),
        (124, 58, 237),
        (217, 119, 6),
        (22, 163, 74),
        (220, 38, 38),
        (8, 145, 178),
    ]


def label_color_map(names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    colors = palette()
    return {name: colors[i % len(colors)] for i, name in enumerate(sorted(names))}


def safe_crop(img: Image.Image, box: Tuple[int, int, int, int], pad: int = 6) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_professional_overlay(base_img: Image.Image, detections: List[dict]) -> Image.Image:
    img = base_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(18)
    small_font = get_font(15)
    class_names = [d["classified_as"] for d in detections] or ["none"]
    cmap = label_color_map(class_names)

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cls_name = d["classified_as"]
        color = cmap.get(cls_name, (15, 118, 110))
        outline = (*color, 255)
        fill = (*color, 60)
        badge_fill = (*color, 230)

        draw.rounded_rectangle([x1, y1, x2, y2], radius=10, outline=outline, width=4, fill=fill)
        title = f"#{d['id']}  {cls_name}"
        subtitle = f"det {d['det_conf']:.2%} | cls {d['cls_conf']:.2%}"

        t1 = draw.textbbox((0, 0), title, font=font)
        t2 = draw.textbbox((0, 0), subtitle, font=small_font)
        bw = max(t1[2] - t1[0], t2[2] - t2[0]) + 20
        bh = (t1[3] - t1[1]) + (t2[3] - t2[1]) + 18
        by = y1 - bh - 8
        if by < 0:
            by = y1 + 8
        bx = x1

        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=12, fill=badge_fill)
        draw.text((bx + 10, by + 6), title, font=font, fill=(255, 255, 255))
        draw.text((bx + 10, by + 28), subtitle, font=small_font, fill=(240, 248, 250))

    return img


def build_gallery(crops: List[Image.Image], labels: List[str]) -> List[Tuple[Image.Image, str]]:
    items = []
    for i, (crop, label) in enumerate(zip(crops, labels), start=1):
        thumb = ImageOps.contain(crop, (240, 240))
        framed = Image.new("RGB", (260, 290), (248, 250, 252))
        framed.paste(thumb, ((260 - thumb.width) // 2, 12))
        d = ImageDraw.Draw(framed)
        font = get_font(16)
        d.rounded_rectangle([10, 250, 250, 280], radius=12, fill=(15, 118, 110))
        d.text((20, 257), f"Cell {i} · {label}", fill=(255, 255, 255), font=font)
        items.append((framed, f"Cell {i}"))
    return items


def export_csv(rows: List[List]):
    if not rows:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(tmp.name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "detected_class", "det_conf", "classified_as", "cls_conf", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    return tmp.name


def analyze_image(img: Image.Image, conf_threshold: float, iou_threshold: float, imgsz: int):
    if img is None:
        return None, [], [], "Upload an image to start analysis.", "0", "-", "-", "-", None

    det_results = detector.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=int(imgsz),
        verbose=False,
    )
    det = det_results[0]

    rows = []
    detections = []
    crops: List[Image.Image] = []
    crop_labels: List[str] = []

    if det.boxes is not None and len(det.boxes) > 0:
        xyxy = det.boxes.xyxy.cpu().numpy().astype(int)
        cls = det.boxes.cls.cpu().numpy().astype(int)
        confs = det.boxes.conf.cpu().numpy()

        for idx in range(len(cls)):
            x1, y1, x2, y2 = xyxy[idx].tolist()
            det_name = det.names.get(int(cls[idx]), str(int(cls[idx])))
            crop = safe_crop(img, (x1, y1, x2, y2))

            cls_result = classifier.predict(source=crop, verbose=False)[0]
            top1 = int(cls_result.probs.top1)
            cls_name = cls_result.names.get(top1, str(top1))
            cls_conf = float(cls_result.probs.top1conf.cpu())
            det_conf = float(confs[idx])

            row = [idx + 1, det_name, round(det_conf, 4), cls_name, round(cls_conf, 4), x1, y1, x2, y2]
            rows.append(row)
            detections.append(
                {
                    "id": idx + 1,
                    "detected_class": det_name,
                    "det_conf": det_conf,
                    "classified_as": cls_name,
                    "cls_conf": cls_conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            crops.append(crop)
            crop_labels.append(f"{cls_name} ({cls_conf:.2%})")

    annotated = draw_professional_overlay(img, detections)
    total = len(rows)

    if total == 0:
        summary = "No cells were detected in the uploaded image."
        total_cells = "0"
        avg_det = "-"
        avg_cls = "-"
        dominant = "-"
    else:
        avg_det_val = sum(d["det_conf"] for d in detections) / total
        avg_cls_val = sum(d["cls_conf"] for d in detections) / total
        counts = Counter(d["classified_as"] for d in detections)
        dominant_name, dominant_count = counts.most_common(1)[0]
        dominant = f"{dominant_name} ({dominant_count})"
        summary = (
            f"Detected {total} cell(s). Dominant predicted class: {dominant_name}. "
            f"Class distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "."
        )
        total_cells = str(total)
        avg_det = f"{avg_det_val:.2%}"
        avg_cls = f"{avg_cls_val:.2%}"

    csv_path = export_csv(rows)
    return annotated, rows, build_gallery(crops, crop_labels), summary, total_cells, avg_det, avg_cls, dominant, csv_path


CSS = """
:root {
  --bg1: #f3f7fb;
  --bg2: #e7eff5;
  --panel: rgba(255,255,255,0.92);
  --panel-strong: #ffffff;
  --border: #dbe3ea;
  --text: #0f172a;
  --muted: #64748b;
  --accent: #0f766e;
  --accent2: #164e63;
  --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
.gradio-container {
  max-width: 1460px !important;
  margin: 0 auto !important;
}
.main-shell {
  background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
  border: 1px solid var(--border);
  border-radius: 26px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.hero {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 4px 2px 16px 2px;
}
.hero h1 {
  margin: 0;
  font-size: 2.2rem;
  color: var(--text);
}
.hero p {
  margin: 6px 0 0 0;
  color: var(--muted);
  font-size: 1rem;
}
.badges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.badge {
  border-radius: 999px;
  padding: 9px 14px;
  background: #ecfeff;
  color: #155e75;
  border: 1px solid #bae6fd;
  font-size: 0.92rem;
  font-weight: 600;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  box-shadow: var(--shadow);
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
  margin-bottom: 12px;
}
.metric-card {
  background: var(--panel-strong);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px;
}
.metric-label {
  color: var(--muted);
  font-size: 0.84rem;
  margin-bottom: 6px;
}
.metric-value {
  color: var(--text);
  font-size: 1.2rem;
  font-weight: 700;
}
.action-note {
  color: var(--muted);
  font-size: 0.92rem;
  margin-top: 4px;
}
#summary_box textarea {
  font-size: 0.97rem !important;
}
footer {display:none !important;}
@media (max-width: 980px) {
  .hero {flex-direction: column; align-items: flex-start;}
  .badges {justify-content: flex-start;}
  .metric-row {grid-template-columns: 1fr 1fr;}
}
@media (max-width: 640px) {
  .metric-row {grid-template-columns: 1fr;}
}
"""


def clear_all():
    return None, None, [], [], "", "0", "-", "-", "-", None


with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Soft(), fill_width=True) as demo:
    with gr.Column(elem_classes=["main-shell"]):
        gr.HTML(
            """
            <div class='hero'>
              <div>
                <h1>Blood Cancer Cell Analysis Studio</h1>
                <p>Professional two-stage AI workflow for microscopy images: object detection, crop-level classification, structured review, and CSV export.</p>
              </div>
              <div class='badges'>
                <div class='badge'>Detector · detectorbc.pt</div>
                <div class='badge'>Classifier · classifymodel.pt</div>
                <div class='badge'>Side-by-side clinical dashboard</div>
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=360):
                with gr.Group(elem_classes=["panel"]):
                    input_image = gr.Image(type="pil", label="Input Microscopy Image", height=520)
                    with gr.Row():
                        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Detection Confidence")
                        iou = gr.Slider(0.05, 0.95, value=0.45, step=0.01, label="NMS IoU")
                    imgsz = gr.Dropdown(choices=[320, 480, 640, 960, 1280], value=640, label="Inference Resolution")
                    with gr.Row():
                        run_btn = gr.Button("Run Full Analysis", variant="primary")
                        clear_btn = gr.Button("Clear Session")
                    gr.Markdown("Use higher resolution for denser blood smear images and lower thresholds if detections are being missed.", elem_classes=["action-note"])

            with gr.Column(scale=7, min_width=420):
                with gr.Group(elem_classes=["panel"]):
                    output_image = gr.Image(type="pil", label="Annotated Diagnostic View", height=520)

                with gr.Row(elem_classes=["metric-row"]):
                    with gr.Column(elem_classes=["metric-card"]):
                        gr.HTML("<div class='metric-label'>Detected Cells</div>")
                        total_cells = gr.Textbox(value="0", show_label=False, interactive=False, container=False, elem_classes=["metric-value"])
                    with gr.Column(elem_classes=["metric-card"]):
                        gr.HTML("<div class='metric-label'>Avg Detector Confidence</div>")
                        avg_det = gr.Textbox(value="-", show_label=False, interactive=False, container=False, elem_classes=["metric-value"])
                    with gr.Column(elem_classes=["metric-card"]):
                        gr.HTML("<div class='metric-label'>Avg Classifier Confidence</div>")
                        avg_cls = gr.Textbox(value="-", show_label=False, interactive=False, container=False, elem_classes=["metric-value"])
                    with gr.Column(elem_classes=["metric-card"]):
                        gr.HTML("<div class='metric-label'>Dominant Class</div>")
                        dominant = gr.Textbox(value="-", show_label=False, interactive=False, container=False, elem_classes=["metric-value"])

                summary = gr.Textbox(label="Case Summary", interactive=False, lines=4, elem_id="summary_box")

        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=420):
                output_table = gr.Dataframe(
                    headers=["id", "detected_class", "det_conf", "classified_as", "cls_conf", "x1", "y1", "x2", "y2"],
                    datatype=["number", "str", "number", "str", "number", "number", "number", "number", "number"],
                    label="Structured Detection and Classification Results",
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )
                download_csv = gr.File(label="Download CSV Report")
            with gr.Column(scale=5, min_width=320):
                gallery = gr.Gallery(label="Detected Cell Crop Review", columns=2, height=500, object_fit="contain")

    run_btn.click(
        fn=analyze_image,
        inputs=[input_image, conf, iou, imgsz],
        outputs=[output_image, output_table, gallery, summary, total_cells, avg_det, avg_cls, dominant, download_csv],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_image, output_image, output_table, gallery, summary, total_cells, avg_det, avg_cls, dominant, download_csv],
    )


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=False, server_name="127.0.0.1", server_port=7860)
