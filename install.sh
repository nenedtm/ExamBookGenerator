#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ExamBookGenerator — One-command installation script
#
# Usage:
#   bash install.sh          # install everything
#   bash install.sh --no-gui # skip PySide6 (CLI-only)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${SCRIPT_DIR}/.venv"
NO_GUI=false

for arg in "$@"; do
    case "$arg" in
        --no-gui) NO_GUI=true ;;
        --help|-h)
            echo "Usage: bash install.sh [--no-gui]"
            echo ""
            echo "Options:"
            echo "  --no-gui   Skip PySide6 installation (CLI-only mode)"
            echo "  -h         Show this help"
            exit 0
            ;;
    esac
done

echo "============================================"
echo "  ExamBookGenerator — Installation"
echo "============================================"
echo ""

# ── 1. Check Python ──────────────────────────────────────────────────────────
echo "[1/5] Checking Python..."
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.12+ first."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
    echo "ERROR: Python 3.12+ required, found $PY_VERSION"
    exit 1
fi
echo "  Python $PY_VERSION found."

# ── 2. Create virtual environment ────────────────────────────────────────────
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
else
    echo "  Already exists: $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 3. Install Python dependencies ───────────────────────────────────────────
echo ""
echo "[3/5] Installing Python packages..."
if [ "$NO_GUI" = true ]; then
    grep -v PySide6 "$SCRIPT_DIR/requirements.txt" | pip install -r /dev/stdin --quiet
    echo "  Installed (CLI-only, no PySide6)."
else
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
    echo "  Installed."
fi

# ── 4. Check Ollama ──────────────────────────────────────────────────────────
echo ""
echo "[4/5] Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"

    # Check if Ollama is running
    if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
        echo "  Ollama server is running."

        # Pull text model
        if ! ollama list 2>/dev/null | grep -q "llama3"; then
            echo "  Pulling llama3 (text model)..."
            ollama pull llama3
        else
            echo "  llama3 already available."
        fi

        # Pull vision model (optional)
        if ! ollama list 2>/dev/null | grep -q "llava"; then
            echo "  Pulling llava (vision model)..."
            ollama pull llava
            echo "  llava pulled. (Optional — used for image matching)"
        else
            echo "  llava already available."
        fi
    else
        echo "  WARNING: Ollama is installed but not running."
        echo "  Start it with: ollama serve"
    fi
else
    echo "  WARNING: Ollama not found."
    echo "  Install from: https://ollama.com"
    echo "  Then run: ollama pull llama3"
fi

# ── 5. Check Tesseract ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Checking Tesseract OCR..."
if command -v tesseract &>/dev/null; then
    echo "  Tesseract found: $(tesseract --version 2>&1 | head -1)"
else
    echo "  WARNING: Tesseract not found (optional — needed for scanned PDFs)."
    case "$(uname -s)" in
        Linux)  echo "  Install: sudo apt install tesseract-ocr" ;;
        Darwin) echo "  Install: brew install tesseract" ;;
        *)      echo "  Please install Tesseract manually." ;;
    esac
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "  Activate the virtual environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Launch the GUI:"
echo "    python main.py"
echo ""
echo "  Or use the CLI:"
echo "    python main.py --input ./StudyMaterial"
echo ""
echo "  For help:"
echo "    python main.py --help"
echo "============================================"
