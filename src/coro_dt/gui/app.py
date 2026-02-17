import glob
import os

import cv2
import numpy as np
import streamlit as st


st.set_page_config(page_title="Coronary Viewer", layout="wide")
st.title("Coronary Vessel Viewer")

# --- Sidebar: file selection & parameters ---
with st.sidebar:
    st.header("Configuration")

    image_path = st.text_input(
        "Image path or directory",
        placeholder="/path/to/images",
    )

    # Resolve images from the provided path
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
        labels = [os.path.relpath(f, image_path if os.path.isdir(image_path) else os.path.dirname(image_path)) for f in image_files]
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
    st.info("Prediction visualization will appear here once inference is implemented.")
