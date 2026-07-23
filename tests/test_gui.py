"""Tests for gui/app.py — PySide6 GUI components."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure a display is available for Qt widget tests (CI / headless).
# PySide6 will use the "offscreen" platform plugin when QT_QPA_PLATFORM
# is set to "offscreen", which allows creating widgets without a real display.
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def app() -> QApplication:
    """Create a single QApplication for the entire test session."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication(sys.argv)
    return instance


@pytest.fixture()
def window(app: QApplication):  # noqa: ANN001
    """Create a fresh ExamBookWindow for each test."""
    from gui.app import ExamBookWindow

    w = ExamBookWindow()
    yield w
    w.close()


# ── Window creation ───────────────────────────────────────────────────────


class TestWindowCreation:
    def test_window_title(self, window) -> None:
        assert window.windowTitle() == "ExamBookGenerator"

    def test_min_size(self, window) -> None:
        size = window.minimumSize()
        assert size.width() >= 600
        assert size.height() >= 600

    def test_generate_btn_exists(self, window) -> None:
        assert window._generate_btn is not None
        assert window._generate_btn.isEnabled()

    def test_progress_bar_initial(self, window) -> None:
        pb = window._progress_bar
        assert pb.maximum() == 8
        assert pb.value() == 0

    def test_depth_slider_default(self, window) -> None:
        assert window._depth_slider.value() == 5
        assert window._depth_value.text() == "5"

    def test_focus_slider_default(self, window) -> None:
        assert window._focus_slider.value() == 5
        assert window._focus_value.text() == "5"

    def test_images_checked_by_default(self, window) -> None:
        assert window._images_cb.isChecked()

    def test_toc_checked_by_default(self, window) -> None:
        assert window._toc_cb.isChecked()

    def test_full_radio_checked_by_default(self, window) -> None:
        assert window._radio_full.isChecked()
        assert not window._radio_topic.isChecked()


# ── Scope toggle ──────────────────────────────────────────────────────────


class TestScopeToggle:
    def test_full_mode_hides_topic_controls(self, window) -> None:
        window.show()
        QApplication.processEvents()
        window._radio_full.setChecked(True)
        QApplication.processEvents()
        assert window._topic_widget.isHidden()
        assert window._focus_row_widget.isHidden()

    def test_topic_mode_shows_topic_controls(self, window) -> None:
        window.show()
        QApplication.processEvents()
        window._radio_topic.setChecked(True)
        QApplication.processEvents()
        assert not window._topic_widget.isHidden()
        assert not window._focus_row_widget.isHidden()

    def test_toggle_back_hides(self, window) -> None:
        window.show()
        QApplication.processEvents()
        window._radio_topic.setChecked(True)
        QApplication.processEvents()
        assert not window._topic_widget.isHidden()
        window._radio_full.setChecked(True)
        QApplication.processEvents()
        assert window._topic_widget.isHidden()


# ── Depth slider label update ─────────────────────────────────────────────


class TestDepthSlider:
    def test_slider_updates_label(self, window) -> None:
        window._depth_slider.setValue(8)
        assert window._depth_value.text() == "8"

    def test_slider_min(self, window) -> None:
        window._depth_slider.setValue(1)
        assert window._depth_value.text() == "1"

    def test_slider_max(self, window) -> None:
        window._depth_slider.setValue(10)
        assert window._depth_value.text() == "10"

    def test_focus_slider_updates_label(self, window) -> None:
        window._focus_slider.setValue(7)
        assert window._focus_value.text() == "7"


# ── Config assembly (build_args) ──────────────────────────────────────────


class TestBuildArgs:
    def test_full_mode_args(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        window._folder_edit.setText(str(folder))
        window._template_edit.setText("template.md")
        window._model_combo.setCurrentText("mistral")
        window._depth_slider.setValue(7)
        window._images_cb.setChecked(False)
        window._toc_cb.setChecked(True)
        window._radio_full.setChecked(True)

        args = window.build_args()
        assert args.input == str(folder)
        assert args.template == "template.md"
        assert args.model == "mistral"
        assert args.depth == 7
        assert args.no_images is True
        assert args.scope == "full"
        assert args.topic is None
        assert args.no_interactive is True

    def test_topic_mode_args(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        window._folder_edit.setText(str(folder))
        window._radio_topic.setChecked(True)
        window._topic_edit.setText("Linear Algebra")
        window._focus_slider.setValue(9)

        args = window.build_args()
        assert args.scope == "topic"
        assert args.topic == "Linear Algebra"
        assert args.focus_depth == 9

    def test_syllabus_passed(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        syllabus = tmp_path / "programma.txt"
        syllabus.write_text("course outline", encoding="utf-8")
        window._folder_edit.setText(str(folder))
        window._syllabus_edit.setText(str(syllabus))

        args = window.build_args()
        assert args.syllabus == str(syllabus)

    def test_empty_syllabus_is_none(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        window._folder_edit.setText(str(folder))
        window._syllabus_edit.setText("")

        args = window.build_args()
        assert args.syllabus is None

    def test_empty_model_is_none(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        window._folder_edit.setText(str(folder))
        self._clear_combo(window._model_combo)

        args = window.build_args()
        assert args.model is None

    @staticmethod
    def _clear_combo(combo) -> None:
        combo.setCurrentText("")


# ── Syllabus auto-detection ──────────────────────────────────────────────


class TestSyllabusDetection:
    def test_detects_syllabus_file(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        (folder / "programma_analisi.txt").write_text("content", encoding="utf-8")
        window._detect_syllabus(folder)
        assert "Detected" in window._syllabus_label.text()
        assert "programma_analisi.txt" in window._syllabus_label.text()

    def test_no_syllabus_found(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        (folder / "notes.txt").write_text("content", encoding="utf-8")
        window._detect_syllabus(folder)
        assert "No syllabus" in window._syllabus_label.text()

    def test_skips_images_in_detection(self, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        (folder / "syllabus_photo.jpg").write_bytes(b"\x00")
        window._detect_syllabus(folder)
        assert "No syllabus" in window._syllabus_label.text()


# ── Log area ──────────────────────────────────────────────────────────────


class TestLogArea:
    def test_append_log(self, window) -> None:
        window._append_log("Hello world")
        text = window._log_area.toPlainText()
        assert "Hello world" in text

    def test_append_multiple_lines(self, window) -> None:
        window._append_log("line 1")
        window._append_log("line 2")
        text = window._log_area.toPlainText()
        assert "line 1" in text
        assert "line 2" in text


# ── Generate button validation ────────────────────────────────────────────


class TestGenerateValidation:
    @patch("gui.app.QMessageBox")
    def test_empty_folder_shows_warning(self, mock_msgbox, window) -> None:
        window._folder_edit.setText("")
        window._on_generate()
        mock_msgbox.warning.assert_called_once()

    @patch("gui.app.QMessageBox")
    def test_invalid_folder_shows_warning(self, mock_msgbox, window) -> None:
        window._folder_edit.setText("/nonexistent/path/that/does/not/exist")
        window._on_generate()
        mock_msgbox.warning.assert_called_once()

    @patch("gui.app.QMessageBox")
    def test_topic_mode_empty_topic_shows_warning(self, mock_msgbox, window, tmp_path: Path) -> None:
        folder = tmp_path / "material"
        folder.mkdir()
        window._folder_edit.setText(str(folder))
        window._radio_topic.setChecked(True)
        window._topic_edit.setText("")
        window._on_generate()
        mock_msgbox.warning.assert_called_once()


# ── Pipeline integration (mocked) ────────────────────────────────────────


class TestPipelineIntegration:
    def test_worker_emits_finished(self, window, tmp_path: Path) -> None:
        from gui.app import _PipelineWorker

        folder = tmp_path / "material"
        folder.mkdir()

        args = MagicMock()
        mock_result = (tmp_path / "out.md", {"overall": "pass", "summary": {}})
        with patch("main.run_pipeline", return_value=mock_result) as mock_rp:
            worker = _PipelineWorker(args)
            received = []
            worker.finished.connect(lambda r: received.append(r))
            worker.run()  # synchronous for testing

            assert len(received) == 1
            assert received[0] == mock_result

    def test_worker_error(self, window) -> None:
        from gui.app import _PipelineWorker

        args = MagicMock()
        with patch("main.run_pipeline", side_effect=RuntimeError("boom")):
            worker = _PipelineWorker(args)
            errors = []
            worker.error.connect(lambda m: errors.append(m))
            worker.run()
            assert errors == ["boom"]

    def test_progress_signal(self, window) -> None:
        from gui.app import _PipelineWorker

        args = MagicMock()
        worker = _PipelineWorker(args)
        received = []
        worker.progress.connect(lambda s, t, m: received.append((s, t, m)))
        worker._on_progress(3, "Processing")
        assert received == [(3, 8, "Processing")]


# ── _set_running ─────────────────────────────────────────────────────────


class TestSetRunning:
    def test_running_disables_button(self, window) -> None:
        window._set_running(True)
        assert not window._generate_btn.isEnabled()
        assert window._generate_btn.text() == "Generating…"

    def test_not_running_enables_button(self, window) -> None:
        window._set_running(False)
        assert window._generate_btn.isEnabled()
        assert window._generate_btn.text() == "Generate Manual"
