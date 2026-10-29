"""Streamlit entry point for the OptiSense AI application."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    MAX_UPLOAD_SIZE_MB,
    MODEL_PATH,
    SUPPORTED_IMAGE_TYPES,
)
from optisense.detector import (
    DetectionError,
    DetectionSettings,
    Detector,
    ModelLoadError,
)
from optisense.image_processor import process_image
from optisense.validators import MediaValidationError


st.set_page_config(
    page_title="OptiSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --optisense-accent: #5b8cff;
            --optisense-muted: #8e9bb0;
            --optisense-border: rgba(148, 163, 184, 0.18);
        }
        .block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101a2b 0%, #121e32 100%);
        }
        .brand-mark {
            color: #f7f9ff;
            font-size: 1.5rem;
            font-weight: 750;
            letter-spacing: -0.04em;
            margin-bottom: 0;
        }
        .brand-tagline { color: #aab7cb; font-size: 0.82rem; }
        .eyebrow {
            color: var(--optisense-accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .metric-panel {
            background: rgba(24, 38, 59, 0.72);
            border: 1px solid var(--optisense-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 112px;
        }
        .metric-label { color: var(--optisense-muted); font-size: 0.85rem; }
        .metric-value {
            color: #f4f7ff;
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        .section-panel {
            background: rgba(24, 38, 59, 0.46);
            border: 1px solid var(--optisense-border);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }
        .status-dot { color: #f0b94b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the YOLO model...")
def get_detector(model_path: str | None) -> Detector:
    detector = Detector(model_path=model_path)
    detector.load()
    return detector


def render_overview() -> None:
    st.markdown('<p class="eyebrow">Computer vision workspace</p>', unsafe_allow_html=True)
    st.title("OptiSense AI")
    st.markdown("### See Smarter. Detect Faster.")
    st.write(
        "A local-first workspace for object detection and tracking across images, "
        "videos, and live camera streams."
    )

    st.write(
        "Initialize the YOLO model when you are ready. Image, video, and camera "
        "workflows will be added in later phases."
    )
    st.caption(
        "The first load may download the selected default weights. Configure "
        "`OPTISENSE_MODEL_PATH` to use an existing local weights file."
    )

    if st.button("Load YOLO model", type="primary"):
        try:
            detector = get_detector(str(MODEL_PATH) if MODEL_PATH is not None else None)
            st.session_state["detector_loaded"] = True
            st.session_state["detector_device"] = detector.device
            st.session_state["detector_name"] = detector.model_name
        except ModelLoadError as exc:
            st.session_state["detector_loaded"] = False
            st.error(str(exc))

    metric_columns = st.columns(4)
    loaded = st.session_state.get("detector_loaded", False)
    metrics = (
        ("Model", st.session_state.get("detector_name", "Not loaded") if loaded else "Not loaded"),
        ("Execution device", st.session_state.get("detector_device", "—") if loaded else "—"),
        ("Detections", "—"),
        ("Processing FPS", "—"),
    )
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown(
                f'<div class="metric-panel"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns((1.15, 0.85))
    with left:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.subheader("Project status")
        status = "YOLO model loaded" if loaded else "Model not loaded"
        st.markdown(f'<span class="status-dot">●</span> {status}', unsafe_allow_html=True)
        st.caption(
            "No media has been processed yet. Detection counts and performance "
            "metrics appear after inference is connected to the interface."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.subheader("Detection defaults")
        st.write(f"Confidence threshold: **{DEFAULT_CONFIDENCE_THRESHOLD:.2f}**")
        st.write(f"IoU threshold: **{DEFAULT_IOU_THRESHOLD:.2f}**")
        st.write(f"Model input size: **{DEFAULT_IMAGE_SIZE} px**")
        st.caption("Confidence filters low-scoring boxes; IoU controls non-maximum suppression.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_image_detection() -> None:
    st.markdown('<p class="eyebrow">Image analysis</p>', unsafe_allow_html=True)
    st.title("Image Detection")
    st.write("Upload a supported image to run local YOLO inference and export its annotation.")
    uploaded = st.file_uploader(
        "Choose an image",
        type=list(SUPPORTED_IMAGE_TYPES),
        max_upload_size=MAX_UPLOAD_SIZE_MB,
        key="image_detection_upload",
    )

    settings_column, action_column = st.columns((1.1, 0.9))
    with settings_column:
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.01,
            help="Higher values filter more low-confidence predictions, which may reduce false positives but miss objects.",
        )
        iou = st.slider(
            "IoU threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_IOU_THRESHOLD,
            step=0.01,
            help="Controls non-maximum suppression of overlapping boxes.",
        )
        image_size = st.select_slider(
            "Model input resolution",
            options=(320, 480, 640, 800, 960, 1280),
            value=DEFAULT_IMAGE_SIZE,
            help="Larger inputs can improve small-object detection at additional processing cost.",
        )
        show_boxes = st.checkbox("Draw bounding boxes", value=True)
        show_confidence = st.checkbox("Show confidence labels", value=True, disabled=not show_boxes)

    with action_column:
        st.markdown("#### Model")
        is_loaded = bool(st.session_state.get("detector_loaded"))
        st.caption(
            f"Loaded: {st.session_state.get('detector_name', 'No')} · "
            f"Device: {st.session_state.get('detector_device', '—')}"
        )
        if st.button("Initialize model", key="initialize_image_model"):
            try:
                detector = get_detector(str(MODEL_PATH) if MODEL_PATH is not None else None)
                st.session_state["detector_loaded"] = True
                st.session_state["detector_device"] = detector.device
                st.session_state["detector_name"] = detector.model_name
                is_loaded = True
            except ModelLoadError as exc:
                st.session_state["detector_loaded"] = False
                st.error(str(exc))

        class_names: dict[int, str] = {}
        if is_loaded:
            detector = get_detector(str(MODEL_PATH) if MODEL_PATH is not None else None)
            class_names = detector.class_names
        selected_names = st.multiselect(
            "Categories",
            options=list(class_names.values()),
            help="Leave empty to detect all supported categories.",
            disabled=not class_names,
            key="image_category_filter",
        )
        st.caption(f"Uploads are limited to {MAX_UPLOAD_SIZE_MB} MB.")

    image_bytes = uploaded.getvalue() if uploaded is not None else None
    selected_class_ids = (
        tuple(class_id for class_id, name in class_names.items() if name in selected_names)
        if selected_names
        else None
    )
    parameters = DetectionSettings(
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        class_ids=selected_class_ids,
    )
    signature: str | None = None
    if uploaded is not None and image_bytes is not None:
        signature_input = (
            image_bytes
            + repr((confidence, iou, image_size, selected_class_ids, show_boxes, show_confidence)).encode()
        )
        signature = hashlib.sha256(signature_input).hexdigest()

    run_detection = st.button(
        "Run detection",
        type="primary",
        disabled=uploaded is None,
        key="run_image_detection",
    )
    if run_detection and uploaded is not None and image_bytes is not None and signature is not None:
        try:
            detector = get_detector(str(MODEL_PATH) if MODEL_PATH is not None else None)
            st.session_state["detector_loaded"] = True
            st.session_state["detector_device"] = detector.device
            st.session_state["detector_name"] = detector.model_name
            with st.spinner("Running object detection..."):
                result = process_image(
                    uploaded.name,
                    image_bytes,
                    detector,
                    parameters,
                    draw_boxes=show_boxes,
                    show_confidence=show_confidence,
                )
            st.session_state["image_detection_result"] = {
                "signature": signature,
                "annotated_png": result.annotated_png(),
                "detections": [detection.to_dict() for detection in result.detections],
                "inference_ms": result.inference_ms,
            }
        except (MediaValidationError, DetectionError, ModelLoadError) as exc:
            st.error(str(exc))

    if uploaded is None:
        st.info("Upload a JPG, JPEG, PNG, or WEBP image to begin.")
        return
    try:
        from optisense.validators import validate_image_upload

        width, height = validate_image_upload(uploaded.name, image_bytes or b"")
    except MediaValidationError as exc:
        st.error(str(exc))
        return

    st.caption(f"Image dimensions: {width:,} × {height:,} pixels")
    result_state = st.session_state.get("image_detection_result")
    if not result_state or result_state.get("signature") != signature:
        st.info("Review the settings, then select **Run detection** to process this image.")
        return

    original_column, annotated_column = st.columns(2)
    with original_column:
        st.subheader("Original")
        st.image(image_bytes, use_container_width=True)
    with annotated_column:
        st.subheader("Detection result")
        st.image(result_state["annotated_png"], use_container_width=True)

    detections = result_state["detections"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Objects detected", len(detections))
    metric_columns[1].metric("Categories", len({item["class_name"] for item in detections}))
    metric_columns[2].metric("Inference time", f"{result_state['inference_ms']:.1f} ms")
    if detections:
        counts = (
            pd.DataFrame(detections)
            .groupby("class_name", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )
        st.subheader("Object counts")
        st.dataframe(counts, hide_index=True, use_container_width=True)
        st.subheader("Detection details")
        st.dataframe(
            pd.DataFrame(detections)[
                ["class_name", "confidence", "x1", "y1", "x2", "y2"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No objects were detected with the current settings.")

    st.download_button(
        "Download annotated image",
        data=result_state["annotated_png"],
        file_name="optisense_annotated.png",
        mime="image/png",
        key="download_annotated_image",
    )


def render_about() -> None:
    st.markdown('<p class="eyebrow">About the project</p>', unsafe_allow_html=True)
    st.title("About OptiSense AI")
    st.write(
        "OptiSense AI is an internship project for locally running object "
        "detection and tracking. Its planned architecture separates the "
        "Streamlit interface from model inference, media processing, analytics, "
        "and exports."
    )
    st.caption(
        "Media processing is intended to run on this computer. Model weights may "
        "need to be obtained separately; see the project README."
    )


with st.sidebar:
    st.markdown('<p class="brand-mark">◉ OptiSense AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-tagline">See Smarter. Detect Faster.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Workspace",
        ("Overview Dashboard", "Image Detection", "About OptiSense AI"),
    )
    st.divider()
    st.caption("Detection engine · Phase 2")

if page == "Overview Dashboard":
    render_overview()
elif page == "Image Detection":
    render_image_detection()
else:
    render_about()
