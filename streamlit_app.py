"""ExamBookGenerator — Streamlit web demo.

Launch with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import streamlit as st

from pipeline.scanner import detect_file_type, detect_syllabus_candidate
from core.models import FileType

# ── Page config (must be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="ExamBookGenerator",
    page_icon="📚",
    layout="wide",
)

# ── Constants ───────────────────────────────────────────────────────────────

_OLLAMA_MODELS = [
    "llama3",
    "llama3.1",
    "mistral",
    "mixtral",
    "codellama",
    "phi3",
    "gemma2",
    "qwen2",
]

_SYLLABUS_KEYWORDS = [
    "syllabus",
    "programma",
    "program",
    "course outline",
    "piano di studi",
]

TOTAL_STEPS = 8


# ── Helpers ─────────────────────────────────────────────────────────────────


def _detect_syllabus(folder: Path) -> str | None:
    """Scan *folder* for a file whose name matches syllabus keywords."""
    try:
        for entry in sorted(folder.iterdir()):
            if not entry.is_file():
                continue
            ftype = detect_file_type(entry)
            if ftype in (FileType.UNKNOWN, FileType.IMAGE):
                continue
            stem = entry.stem.lower()
            for kw in _SYLLABUS_KEYWORDS:
                if kw in stem:
                    return entry.name
    except OSError:
        pass
    return None


def _build_args(
    *,
    input_dir: str,
    template: str,
    model: str | None,
    depth: int,
    no_images: bool,
    syllabus: str | None,
    scope: str,
    topic: str | None,
    focus_depth: int,
) -> Namespace:
    """Build an ``argparse.Namespace`` compatible with ``run_pipeline``."""
    return Namespace(
        input=input_dir,
        template=template,
        model=model,
        output=None,
        depth=depth,
        no_images=no_images,
        syllabus=syllabus,
        scope=scope,
        topic=topic,
        focus_depth=focus_depth,
        no_interactive=True,
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # Source folder
    input_folder = st.text_input(
        "Source material folder",
        placeholder="/path/to/StudyMaterial",
    )

    # Template
    template_path = st.text_input("Template file", value="template.md")

    # Model
    model_options = _OLLAMA_MODELS + ["(custom)"]
    model_choice = st.selectbox("LLM model", model_options, index=0)
    if model_choice == "(custom)":
        model_choice = st.text_input("Custom model name", placeholder="e.g. qwen3")
        model_value = model_choice.strip() or None
    else:
        model_value = model_choice

    # Depth slider
    depth = st.slider("Depth level", 1, 10, 5, help="1 = minimal summary, 10 = exhaustive")

    # Checkboxes
    include_images = st.checkbox("Include images from source material", value=True)
    include_toc = st.checkbox("Include table of contents", value=True)

    st.divider()

    # Syllabus
    st.subheader("📋 Syllabus (optional)")
    syllabus_file = st.file_uploader(
        "Upload syllabus file",
        type=["txt", "md", "pdf", "docx"],
        label_visibility="collapsed",
    )

# ── Detect syllabus on folder change ────────────────────────────────────────

syllabus_hint: str | None = None
if input_folder:
    folder_path = Path(input_folder)
    if folder_path.is_dir():
        detected = _detect_syllabus(folder_path)
        if detected:
            syllabus_hint = detected

# ── Main area ───────────────────────────────────────────────────────────────

st.title("📚 ExamBookGenerator")
st.caption("Generate a structured exam manual from your study material — fully local, powered by Ollama.")

# Scope selection
col_scope, col_topic = st.columns([1, 2])
with col_scope:
    scope = st.radio(
        "Generation scope",
        ["Full manual", "Single topic focus"],
        horizontal=True,
    )

scope_value = "full" if scope == "Full manual" else "topic"

topic_name: str | None = None
focus_depth = depth

if scope_value == "topic":
    with col_topic:
        topic_name = st.text_input("Topic name", placeholder="e.g. Linear Algebra")
    focus_depth = st.slider(
        "Focus depth level",
        1,
        10,
        5,
        help="Independent detail level for the focused topic",
    )
else:
    focus_depth = depth

# ── Show syllabus detection hint ────────────────────────────────────────────

if syllabus_hint:
    st.info(f"🔎 Detected syllabus in folder: **{syllabus_hint}**")

if syllabus_file:
    st.info(f"📎 Syllabus uploaded: **{syllabus_file.name}**")

# ── Generate button ─────────────────────────────────────────────────────────

generate = st.button("▶ Generate Manual", type="primary", use_container_width=True)

# ── Pipeline execution ──────────────────────────────────────────────────────

if generate:
    # Validate inputs
    if not input_folder:
        st.error("Please enter a source material folder path.")
        st.stop()

    folder_path = Path(input_folder)
    if not folder_path.is_dir():
        st.error(f"Not a directory: {input_folder}")
        st.stop()

    if not Path(template_path).is_file():
        st.error(f"Template not found: {template_path}")
        st.stop()

    if scope_value == "topic" and not topic_name:
        st.error("Please enter a topic name for focus mode.")
        st.stop()

    # Handle uploaded syllabus file (save to temp)
    syllabus_path: str | None = None
    if syllabus_file:
        tmp = Path(f"/tmp/ebg_syllabus_{syllabus_file.name}")
        tmp.write_bytes(syllabus_file.getvalue())
        syllabus_path = str(tmp)
    elif syllabus_hint and syllabus_path is None:
        syllabus_path = str(folder_path / syllabus_hint)

    # Build args
    args = _build_args(
        input_dir=input_folder,
        template=template_path,
        model=model_value,
        depth=depth,
        no_images=not include_images,
        syllabus=syllabus_path,
        scope=scope_value,
        topic=topic_name,
        focus_depth=focus_depth,
    )

    # Progress UI
    progress_bar = st.progress(0, text="Starting pipeline...")
    log_container = st.expander("📋 Pipeline log", expanded=True)
    log_lines: list[str] = []

    def _streamlit_progress(step: int, msg: str) -> None:
        """Callback invoked by run_pipeline on each step."""
        progress_bar.progress(step / TOTAL_STEPS, text=f"Step {step}/{TOTAL_STEPS}: {msg}")
        log_lines.append(f"[{step}/{TOTAL_STEPS}] {msg}")
        log_container.code("\n".join(log_lines), language=None)

    # Run
    try:
        from main import run_pipeline

        with st.spinner("Running pipeline... this may take several minutes."):
            output_path, validation = run_pipeline(
                args,
                progress_callback=_streamlit_progress,
            )

        progress_bar.progress(1.0, text="Done!")
        log_container.code("\n".join(log_lines), language=None)

        # Results
        st.divider()
        status = validation["overall"].upper()
        summary = validation["summary"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", status)
        col2.metric("Checks passed", summary["passed"])
        col3.metric("Warnings", summary["warnings"])

        st.success(f"Manual generated: `{output_path}`")

        # Download button
        manual_text = output_path.read_text(encoding="utf-8")
        st.download_button(
            label="⬇️ Download Exam_Manual.md",
            data=manual_text,
            file_name=output_path.name,
            mime="text/markdown",
            use_container_width=True,
        )

    except Exception as exc:
        progress_bar.progress(0, text="Error")
        st.error(f"Pipeline error: {exc}")
        log_container.code("\n".join(log_lines), language=None)
