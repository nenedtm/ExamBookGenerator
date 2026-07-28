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
    "qwen3",
    "llama3.1",
    "qwen2.5",
    "llama3",
    "mistral",
    "gemma2",
    "phi3",
    "codellama",
]

_SYLLABUS_KEYWORDS = [
    "syllabus",
    "programma",
    "program",
    "course outline",
    "piano di studi",
]


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
        chapters=None,
        force=False,
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
        "Focus depth level", 1, 10, 5,
        help="Independent detail level for the focused topic",
    )
else:
    focus_depth = depth

# ── Show syllabus detection hint ────────────────────────────────────────────

if syllabus_hint:
    st.info(f"🔎 Detected syllabus in folder: **{syllabus_hint}**")

if syllabus_file:
    st.info(f"📎 Syllabus uploaded: **{syllabus_file.name}**")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Genera Indice
# ═══════════════════════════════════════════════════════════════════════════

# Initialize session state
if "phase" not in st.session_state:
    st.session_state.phase = "idle"  # idle → outline_done → done
if "outline_state" not in st.session_state:
    st.session_state.outline_state = None
if "chapter_selection" not in st.session_state:
    st.session_state.chapter_selection = {}

generate_index = st.button(
    "📋 Genera Indice",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.phase == "outline_done",
)

if generate_index:
    # Validate inputs
    if not input_folder:
        st.error("Please enter a source material folder path.")
        st.stop()
    if not Path(input_folder).is_dir():
        st.error(f"Not a directory: {input_folder}")
        st.stop()
    if not Path(template_path).is_file():
        st.error(f"Template not found: {template_path}")
        st.stop()
    if scope_value == "topic" and not topic_name:
        st.error("Please enter a topic name for focus mode.")
        st.stop()

    # Handle uploaded syllabus
    syllabus_path: str | None = None
    if syllabus_file:
        tmp = Path(f"/tmp/ebg_syllabus_{syllabus_file.name}")
        tmp.write_bytes(syllabus_file.getvalue())
        syllabus_path = str(tmp)
    elif syllabus_hint:
        syllabus_path = str(Path(input_folder) / syllabus_hint)

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

    progress_bar = st.progress(0, text="Starting outline generation...")
    log_container = st.expander("📋 Pipeline log", expanded=True)
    log_lines: list[str] = []

    def _outline_progress(step: int, msg: str) -> None:
        progress_bar.progress(step / 6, text=f"Step {step}/6: {msg}")
        log_lines.append(f"[{step}/6] {msg}")
        log_container.code("\n".join(log_lines), language=None)

    try:
        from main import run_outline_phase

        with st.spinner("Analisi materiali e generazione outline..."):
            outline_state = run_outline_phase(args, progress_callback=_outline_progress)

        progress_bar.progress(1.0, text="Indice generato!")
        log_container.code("\n".join(log_lines), language=None)

        # Store in session
        st.session_state.outline_state = outline_state
        st.session_state.phase = "outline_done"

        # Initialize selection: all selected by default
        outline_chapters = outline_state["outline_chapters"]
        st.session_state.chapter_selection = {
            i: True for i in range(len(outline_chapters))
        }

        st.rerun()

    except Exception as exc:
        progress_bar.progress(0, text="Error")
        st.error(f"Outline generation error: {exc}")
        log_container.code("\n".join(log_lines), language=None)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Indice generato: selezione + generazione manuale
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.phase == "outline_done" and st.session_state.outline_state is not None:
    state = st.session_state.outline_state
    outline_chapters = state["outline_chapters"]
    existing_chapters = state.get("existing_chapters", set())

    st.divider()
    st.subheader(f"📖 Indice del Manuale — {len(outline_chapters)} capitoli")
    st.caption("Seleziona i capitoli da generare. Puoi modificare i titoli e riordinare le sezioni.")

    # Show each chapter with checkbox + editable title + sections
    for i, ch in enumerate(outline_chapters):
        is_existing = i in existing_chapters
        existing_label = " ✅ già generato" if is_existing else ""

        col_check, col_title = st.columns([1, 5])
        with col_check:
            default_val = st.session_state.chapter_selection.get(i, True)
            st.session_state.chapter_selection[i] = st.checkbox(
                f"Cap. {i + 1}",
                value=default_val,
                key=f"ch_check_{i}",
            )
        with col_title:
            st.markdown(f"**{ch.title}**{existing_label}")
            if ch.sections:
                for sec in ch.sections:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {sec}")

        if i < len(outline_chapters) - 1:
            st.markdown("---")

    # Select all / deselect all
    col_sel, col_desel = st.columns(2)
    with col_sel:
        if st.button("✅ Seleziona tutti", use_container_width=True):
            for i in range(len(outline_chapters)):
                st.session_state.chapter_selection[i] = True
            st.rerun()
    with col_desel:
        if st.button("❌ Deseleziona tutti", use_container_width=True):
            for i in range(len(outline_chapters)):
                st.session_state.chapter_selection[i] = False
            st.rerun()

    # Generate manual button
    st.divider()
    selected_set = {i for i, v in st.session_state.chapter_selection.items() if v}
    n_selected = len(selected_set)

    generate_manual = st.button(
        f"▶ Genera Manuale ({n_selected} capitoli selezionati)",
        type="primary",
        use_container_width=True,
        disabled=n_selected == 0,
    )

    if generate_manual:
        progress_bar = st.progress(0, text="Generating chapters...")
        log_container = st.expander("📋 Pipeline log", expanded=True)
        log_lines: list[str] = []

        def _chapter_progress(step: int, msg: str) -> None:
            progress_bar.progress(min(step / 8, 1.0), text=f"Step {step}/8: {msg}")
            log_lines.append(f"[{step}/8] {msg}")
            log_container.code("\n".join(log_lines), language=None)

        try:
            from main import run_chapters_phase

            with st.spinner("Generazione capitoli... questo potrebbe volerci qualche minuto."):
                output_path, validation = run_chapters_phase(
                    st.session_state.outline_state,
                    selected_indices=selected_set if selected_set != set(range(len(outline_chapters))) else None,
                    progress_callback=_chapter_progress,
                )

            progress_bar.progress(1.0, text="Done!")
            log_container.code("\n".join(log_lines), language=None)
            st.session_state.phase = "done"

            # Results
            st.divider()
            status = validation["overall"].upper()
            summary = validation["summary"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Status", status)
            col2.metric("Checks passed", summary["passed"])
            col3.metric("Warnings", summary["warnings"])

            st.success(f"Manual generated: `{output_path}`")

            # Show indice
            indice_path = Path("output/indice.md")
            if indice_path.exists():
                with st.expander("📖 Indice capitoli (indice.md)", expanded=False):
                    indice_text = indice_path.read_text(encoding="utf-8")
                    st.markdown(indice_text)

            # Show individual chapters
            chapters_dir = Path("output/chapters")
            if chapters_dir.exists():
                chapter_files = sorted(chapters_dir.glob("cap_*.md"))
                if chapter_files:
                    with st.expander(f"📄 Capitoli individuali ({len(chapter_files)} file)", expanded=False):
                        for cf in chapter_files:
                            st.text(cf.name)

            # Download button
            manual_text = output_path.read_text(encoding="utf-8", errors="replace")
            st.download_button(
                label="⬇️ Download Exam_Manual.md",
                data=manual_text,
                file_name=output_path.name,
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as exc:
            progress_bar.progress(0, text="Error")
            st.error(f"Chapter generation error: {exc}")
            log_container.code("\n".join(log_lines), language=None)

# ── Reset button ────────────────────────────────────────────────────────────

if st.session_state.phase != "idle":
    if st.button("🔄 Ricomincia da zero", use_container_width=True):
        st.session_state.phase = "idle"
        st.session_state.outline_state = None
        st.session_state.chapter_selection = {}
        st.rerun()
