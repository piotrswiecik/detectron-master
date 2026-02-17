import glob
import json
import os

import cv2
import numpy as np
import streamlit as st
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from coro_dt.config import ParamsConfig, apply_pointrend_overrides

NUM_CLASSES = 1
VESSEL_COLOR = (0, 255, 0)  # BGR green for overlay
OVERLAY_ALPHA = 0.45


@st.cache_resource
def load_predictor(
    model_path: str,
    params_json: str,
    threshold: float,
    use_cpu: bool,
) -> DefaultPredictor:
    """Build and cache a DefaultPredictor from model weights + params."""
    with open(params_json, "r") as f:
        config = ParamsConfig(**json.load(f))

    cfg = get_cfg()

    if config.use_pointrend:
        from detectron2.projects.point_rend import add_pointrend_config
        add_pointrend_config(cfg)

    cfg.merge_from_file(model_zoo.get_config_file(config.backbone.value))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = config.anchor_sizes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = config.anchor_ratios
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.roi_batch_size
    cfg.MODEL.BACKBONE.FREEZE_AT = config.freeze_at

    if config.use_pointrend:
        apply_pointrend_overrides(cfg, num_classes=NUM_CLASSES)

    if use_cpu:
        cfg.MODEL.DEVICE = "cpu"

    return DefaultPredictor(cfg)


def render_overlay(image_bgr: np.ndarray, instances, show_boxes: bool = True) -> np.ndarray:
    """Draw masks and optionally bounding boxes on a copy of the image."""
    canvas = image_bgr.copy()

    if len(instances) == 0:
        return canvas

    instances = instances.to("cpu")
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()

    # Composite masks
    for mask in masks:
        colored = np.zeros_like(canvas)
        colored[:] = VESSEL_COLOR
        mask_bool = mask.astype(bool)
        canvas[mask_bool] = cv2.addWeighted(
            canvas[mask_bool], 1 - OVERLAY_ALPHA, colored[mask_bool], OVERLAY_ALPHA, 0
        )

    # Draw boxes + scores
    if show_boxes:
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), VESSEL_COLOR, 2)
            cv2.putText(
                canvas,
                f"{score:.2f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                VESSEL_COLOR,
                1,
                cv2.LINE_AA,
            )

    return canvas


# ---- Streamlit layout ----

st.set_page_config(page_title="Coronary Vessel Viewer", layout="wide")
st.title("Coronary Vessel Viewer")

# --- Sidebar: file selection & parameters ---
with st.sidebar:
    st.header("Configuration")

    image_path = st.text_input(
        "Image path or directory",
        placeholder="/path/to/images",
    )

    image_files: list[str] = []
    if image_path:
        if os.path.isfile(image_path):
            image_files = [image_path]
        elif os.path.isdir(image_path):
            image_files = sorted(
                glob.glob(os.path.join(image_path, "**", "*.png"), recursive=True)
                + glob.glob(os.path.join(image_path, "**", "*.jpg"), recursive=True)
                + glob.glob(os.path.join(image_path, "**", "*.jpeg"), recursive=True)
                + glob.glob(os.path.join(image_path, "**", "*.bmp"), recursive=True)
            )

    selected_image: str | None = None
    if image_files:
        base = image_path if os.path.isdir(image_path) else os.path.dirname(image_path)
        labels = [os.path.relpath(f, base) for f in image_files]
        choice = st.selectbox("Select image", labels)
        if choice is not None:
            selected_image = image_files[labels.index(choice)]
    elif image_path:
        st.warning("No images found at the given path.")

    st.divider()

    model_dir = st.text_input(
        "Model artifacts directory",
        placeholder="/path/to/trained_models",
    )

    model_files: list[str] = []
    if model_dir and os.path.isdir(model_dir):
        model_files = sorted(
            glob.glob(os.path.join(model_dir, "**", "*.pth"), recursive=True)
        )

    selected_model: str | None = None
    if model_files:
        model_labels = [os.path.relpath(f, model_dir) for f in model_files]
        model_choice = st.selectbox("Select model", model_labels)
        if model_choice is not None:
            selected_model = model_files[model_labels.index(model_choice)]
    elif model_dir:
        st.warning("No .pth files found in the given directory.")

    st.divider()

    params_file = st.text_input(
        "Params JSON file",
        placeholder="/path/to/params.json",
    )

    st.divider()

    threshold = st.slider(
        "Score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    use_cpu = st.checkbox("Force CPU inference", value=False)
    show_boxes = st.checkbox("Show bounding boxes", value=True)

# --- Validate readiness ---
ready = (
    selected_image is not None
    and selected_model is not None
    and params_file
    and os.path.isfile(params_file)
)

# --- Main area: two-column visualization ---
col_input, col_prediction = st.columns(2)

with col_input:
    st.subheader("Input Image")
    if selected_image is not None:
        bgr = cv2.imread(selected_image)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=os.path.basename(selected_image), use_container_width=True)
        else:
            st.error(f"Failed to read image: {selected_image}")
    else:
        st.info("Select an image from the sidebar to preview it here.")

with col_prediction:
    st.subheader("Prediction Overlay")
    if not ready:
        st.info("Select an image, model, and params file to run inference.")
    else:
        try:
            predictor = load_predictor(selected_model, params_file, threshold, use_cpu)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        bgr = cv2.imread(selected_image)
        if bgr is None:
            st.error(f"Failed to read image: {selected_image}")
        else:
            with st.spinner("Running inference..."):
                outputs = predictor(bgr)
            instances = outputs["instances"]
            st.caption(f"{len(instances)} instance(s) detected")
            overlay = render_overlay(bgr, instances, show_boxes=show_boxes)
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption="Prediction",
                use_container_width=True,
            )
