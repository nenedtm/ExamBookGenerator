"""PySide6 graphical user interface for ExamBookGenerator.

Launch with ``python main.py`` (no ``--input`` flag) or directly::

    python -m gui.app
"""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from core.models import FileType
from pipeline.scanner import detect_file_type
from utils.config import ConfigManager

logger = logging.getLogger(__name__)

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


# ── Worker thread ─────────────────────────────────────────────────────────


class _PipelineWorker(QThread):
    """Runs ``run_pipeline`` in a background thread."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, args: Namespace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._args = args

    def run(self) -> None:  # noqa: D401
        try:
            from main import run_pipeline

            output_path, validation = run_pipeline(
                self._args,
                progress_callback=self._on_progress,
            )
            self.finished.emit((output_path, validation))
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_progress(self, step: int, msg: str) -> None:
        """Forward progress from the pipeline to the GUI thread."""
        self.progress.emit(step, 8, msg)

    progress = Signal(int, int, str)


# ── Main window ───────────────────────────────────────────────────────────


class ExamBookWindow(QMainWindow):
    """Main application window."""

    _WINDOW_TITLE = "ExamBookGenerator"

    _SYLLABUS_KEYWORDS: list[str] = [
        "syllabus",
        "programma",
        "program",
        "course outline",
        "piano di studi",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self._WINDOW_TITLE)
        self.setMinimumSize(640, 620)
        self._worker: _PipelineWorker | None = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Source material ────────────────────────────────────────────
        form = QFormLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select the folder with your study material…")
        self._folder_edit.setReadOnly(True)
        folder_btn = QPushButton("Browse…")
        folder_btn.clicked.connect(self._browse_folder)
        row = QHBoxLayout()
        row.addWidget(self._folder_edit)
        row.addWidget(folder_btn)
        form.addRow("Source material:", row)

        # Template
        self._template_edit = QLineEdit("template.md")
        template_btn = QPushButton("Browse…")
        template_btn.clicked.connect(self._browse_template)
        trow = QHBoxLayout()
        trow.addWidget(self._template_edit)
        trow.addWidget(template_btn)
        form.addRow("Template:", trow)

        # Model
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(_OLLAMA_MODELS)
        self._model_combo.setCurrentText(ConfigManager().get("llm.model", "llama3"))
        form.addRow("LLM model:", self._model_combo)

        layout.addLayout(form)

        # ── Syllabus ──────────────────────────────────────────────────
        syll_group = QGroupBox("Syllabus (optional)")
        syll_lay = QVBoxLayout(syll_group)
        syll_row = QHBoxLayout()
        self._syllabus_edit = QLineEdit()
        self._syllabus_edit.setPlaceholderText("Path to course syllabus file…")
        syll_btn = QPushButton("Browse…")
        syll_btn.clicked.connect(self._browse_syllabus)
        syll_row.addWidget(self._syllabus_edit)
        syll_row.addWidget(syll_btn)
        syll_lay.addLayout(syll_row)
        self._syllabus_label = QLabel("")
        self._syllabus_label.setStyleSheet("color: #2e7d32; font-style: italic;")
        self._syllabus_label.setWordWrap(True)
        syll_lay.addWidget(self._syllabus_label)
        layout.addWidget(syll_group)

        # ── Scope ─────────────────────────────────────────────────────
        scope_group = QGroupBox("Generation scope")
        scope_lay = QVBoxLayout(scope_group)

        self._radio_full = QRadioButton("Generate full manual")
        self._radio_topic = QRadioButton("Focus on a specific topic")
        self._radio_full.setChecked(True)
        scope_bg = QButtonGroup(self)
        scope_bg.addButton(self._radio_full, 0)
        scope_bg.addButton(self._radio_topic, 1)
        scope_lay.addWidget(self._radio_full)
        scope_lay.addWidget(self._radio_topic)

        # Topic controls
        self._topic_widget = QWidget()
        tw_lay = QFormLayout(self._topic_widget)
        tw_lay.setContentsMargins(0, 0, 0, 0)
        self._topic_edit = QLineEdit()
        self._topic_edit.setPlaceholderText("e.g. Linear Algebra")
        tw_lay.addRow("Topic:", self._topic_edit)
        self._topic_widget.hide()
        scope_lay.addWidget(self._topic_widget)

        layout.addWidget(scope_group)

        # ── Options ───────────────────────────────────────────────────
        opt_group = QGroupBox("Options")
        opt_lay = QVBoxLayout(opt_group)

        # Depth slider
        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Depth level:"))
        self._depth_slider = QSlider(Qt.Horizontal)
        self._depth_slider.setRange(1, 10)
        self._depth_slider.setValue(5)
        self._depth_slider.setTickPosition(QSlider.TicksBelow)
        self._depth_slider.setTickInterval(1)
        depth_row.addWidget(self._depth_slider)
        self._depth_value = QLabel("5")
        self._depth_value.setMinimumWidth(20)
        depth_row.addWidget(self._depth_value)
        self._depth_slider.valueChanged.connect(
            lambda v: self._depth_value.setText(str(v))
        )
        opt_lay.addLayout(depth_row)

        # Focus depth slider (topic mode only)
        focus_row = QHBoxLayout()
        focus_row.addWidget(QLabel("Focus depth level:"))
        self._focus_slider = QSlider(Qt.Horizontal)
        self._focus_slider.setRange(1, 10)
        self._focus_slider.setValue(5)
        self._focus_slider.setTickPosition(QSlider.TicksBelow)
        self._focus_slider.setTickInterval(1)
        focus_row.addWidget(self._focus_slider)
        self._focus_value = QLabel("5")
        self._focus_value.setMinimumWidth(20)
        focus_row.addWidget(self._focus_value)
        self._focus_slider.valueChanged.connect(
            lambda v: self._focus_value.setText(str(v))
        )
        self._focus_row_widget = QWidget()
        self._focus_row_widget.setLayout(focus_row)
        self._focus_row_widget.hide()
        opt_lay.addWidget(self._focus_row_widget)

        # Checkboxes
        self._images_cb = QCheckBox("Include images from source material")
        self._images_cb.setChecked(True)
        opt_lay.addWidget(self._images_cb)
        self._toc_cb = QCheckBox("Include table of contents")
        self._toc_cb.setChecked(True)
        opt_lay.addWidget(self._toc_cb)

        layout.addWidget(opt_group)

        # ── Generate button ───────────────────────────────────────────
        self._generate_btn = QPushButton("Generate Manual")
        self._generate_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; font-weight: bold; }"
        )
        self._generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._generate_btn)

        # ── Progress ──────────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 8)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # ── Log output ────────────────────────────────────────────────
        log_group = QGroupBox("Log output")
        log_lay = QVBoxLayout(log_group)
        self._log_area = QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumBlockCount(2000)
        self._log_area.setStyleSheet(
            "QPlainTextEdit { font-family: monospace; font-size: 11px; }"
        )
        log_lay.addWidget(self._log_area)
        layout.addWidget(log_group)

        # ── Connections ───────────────────────────────────────────────
        self._radio_full.toggled.connect(self._on_scope_changed)

    # ── Slots ──────────────────────────────────────────────────────────

    @Slot()
    def _browse_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Select source material folder")
        if path:
            self._folder_edit.setText(path)
            self._detect_syllabus(Path(path))

    @Slot()
    def _browse_template(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Select template file", "", "Markdown (*.md);;All files (*)"
        )
        if path:
            self._template_edit.setText(path)

    @Slot()
    def _browse_syllabus(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select syllabus file",
            "",
            "All supported (*.txt *.md *.pdf *.docx);;All files (*)",
        )
        if path:
            self._syllabus_edit.setText(path)
            self._syllabus_label.setText("")

    def _detect_syllabus(self, folder: Path) -> None:
        """Scan *folder* for a syllabus candidate and show a hint."""
        try:
            for entry in sorted(folder.iterdir()):
                if not entry.is_file():
                    continue
                ftype = detect_file_type(entry)
                if ftype in (FileType.UNKNOWN, FileType.IMAGE):
                    continue
                stem = entry.stem.lower()
                for kw in self._SYLLABUS_KEYWORDS:
                    if kw in stem:
                        self._syllabus_label.setText(
                            f"Detected: {entry.name} — use this?"
                        )
                        return
            self._syllabus_label.setText("No syllabus detected in folder.")
        except OSError:
            self._syllabus_label.setText("")

    @Slot(bool)
    def _on_scope_changed(self, full_checked: bool) -> None:
        topic_mode = not full_checked
        self._topic_widget.setVisible(topic_mode)
        self._focus_row_widget.setVisible(topic_mode)

    # ── Args assembly ──────────────────────────────────────────────────

    def build_args(self) -> Namespace:
        """Read all GUI controls and return an ``argparse.Namespace``.

        The returned namespace matches what ``run_pipeline`` expects.
        """
        scope = "full" if self._radio_full.isChecked() else "topic"
        return Namespace(
            input=self._folder_edit.text().strip() or None,
            template=self._template_edit.text().strip() or "template.md",
            model=self._model_combo.currentText().strip() or None,
            output=None,
            depth=self._depth_slider.value(),
            no_images=not self._images_cb.isChecked(),
            syllabus=self._syllabus_edit.text().strip() or None,
            scope=scope,
            topic=self._topic_edit.text().strip() or None,
            focus_depth=self._focus_slider.value(),
            no_interactive=True,
        )

    # ── Generate ───────────────────────────────────────────────────────

    @Slot()
    def _on_generate(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "Input required", "Please select a source material folder.")
            return
        folder_path = Path(folder)
        if not folder_path.is_dir():
            QMessageBox.warning(self, "Invalid folder", f"Not a directory: {folder}")
            return

        template = self._template_edit.text().strip() or "template.md"
        if not Path(template).is_file():
            QMessageBox.warning(self, "Template not found", f"Template not found: {template}")
            return

        scope = "full" if self._radio_full.isChecked() else "topic"
        focus_topic = self._topic_edit.text().strip() if scope == "topic" else None
        if scope == "topic" and not focus_topic:
            QMessageBox.warning(
                self,
                "Topic required",
                "Please enter a topic name for focus mode.",
            )
            return

        self._set_running(True)
        self._progress_bar.setValue(0)
        self._log_area.clear()
        self._append_log("Pipeline starting…")

        args = self.build_args()

        self._worker = _PipelineWorker(args, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self._generate_btn.setEnabled(not running)
        self._generate_btn.setText("Generating…" if running else "Generate Manual")

    @Slot(int, int, str)
    def _on_progress(self, step: int, total: int, msg: str) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(step)
        self._progress_bar.setFormat(f"Step {step}/{total} — %p%")
        self._append_log(f"[{step}/{total}] {msg}")

    @Slot(object)
    def _on_finished(self, result: object) -> None:
        output_path, validation = result
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._set_running(False)

        status = validation["overall"].upper()
        summary = validation["summary"]
        self._append_log(
            f"\nComplete! Status: {status} "
            f"({summary['passed']} passed, "
            f"{summary['warnings']} warnings, "
            f"{summary['failed']} failed)"
        )
        self._append_log(f"Output: {output_path}")

        QMessageBox.information(
            self,
            "Generation complete",
            f"Status: {status}\n\n{output_path}",
        )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._set_running(False)
        self._progress_bar.setValue(0)
        self._append_log(f"\nERROR: {msg}")
        QMessageBox.critical(self, "Pipeline error", msg)

    def _append_log(self, text: str) -> None:
        self._log_area.appendPlainText(text)
        self._log_area.moveCursor(QTextCursor.End)


# ── Public API ────────────────────────────────────────────────────────────


def launch() -> None:
    """Create the ``QApplication`` and show the main window."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = ExamBookWindow()
    window.show()
    app.exec()


# ── Standalone entry point ────────────────────────────────────────────────


if __name__ == "__main__":
    launch()
