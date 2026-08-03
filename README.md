# ExamBookGenerator

**Local AI-powered academic material transformation system**

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![AI](https://img.shields.io/badge/AI-Ollama-green)
![Offline](https://img.shields.io/badge/Execution-Local-success)
![Markdown](https://img.shields.io/badge/Output-Markdown-orange)
![Multimodal](https://img.shields.io/badge/AI-Vision-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-644_passing-brightgreen)

---

## Overview

**ExamBookGenerator** is a fully local, AI-powered system that turns a disorganized folder of academic material into a single, structured, English-language exam manual. Everything runs on your machine via [Ollama](https://ollama.com) — no data ever leaves your computer.

Point it at a folder (with any level of nesting) containing PDF books, lecture notes, PowerPoint slides, DOCX documents, or Markdown/TXT files — and it will:

1. **Scan** every file recursively, including nested subfolders;
2. **Extract** text and embedded images from each document;
3. **Deduplicate** content using exact hash matching and near-duplicate detection (92% similarity threshold);
4. **Chunk** the text into AI-compatible segments with configurable overlap;
5. **Identify** academic topics automatically — optionally guided by a course syllabus (*programma*);
6. **Build** a logical manual structure with hierarchical chapters and sections;
7. **Generate** detailed, exam-focused chapters — reusing real images from your material where the AI decides they help explain a concept;
8. **Validate** the result with 9 automated quality checks (structure, language, duplicates, image references, TOC integrity, syllabus order);
9. **Merge** everything into one clean Markdown file with table of contents and chapter numbering.

### Three ways to run it

| Interface | Command | Best for |
|---|---|---|
| **CLI** | `python main.py --input ./material` | Scripting, CI, power users |
| **Desktop GUI** | `python main.py` (no `--input`) | Visual workflow on your desktop |
| **Web Demo** | `streamlit run streamlit_app.py` | Quick access from any browser |

### Key features

- **Full manual** or **single-topic focus** mode — generate everything or zoom into one subject
- **Configurable detail level** (1–10) — from a minimal summary to an exhaustive reference
- **Automatic syllabus detection** — finds your `programma` file and orders chapters to match your course
- **Vision-powered image matching** — uses a local vision model to pick the most relevant images and place them in context
- **Per-chapter source citations** — every chapter ends with a `References & Resources` section listing the source documents it was built from, and the LLM is prompted to cite them inline as `[1]`, `[2]`, …
- **Always English output** — regardless of the source material language

**Output structure:**

```
output/
├── Exam_Manual.md          # The final manual
├── topics.json             # Detected topics
├── outline.md              # Chapter structure
├── validation.json         # Quality check results
└── assets/
    └── images/             # Extracted and deduplicated images
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve

# 3. Pull a text model (required)
ollama pull llama3

# 4. (Optional) Pull a vision model for image matching
ollama pull llava

# 5a. Launch the web demo (simplest)
streamlit run streamlit_app.py

# 5b. Or use the CLI
python main.py --input ./StudyMaterial

# 5c. Or launch the desktop GUI
python main.py
```

Or use the installation script:

```bash
bash install.sh
```

---

## Installation

### Requirements

- **Python 3.12+**
- **[Ollama](https://ollama.com)** installed and running locally
- **Tesseract OCR** (only needed for scanned PDFs / image text extraction)

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama models

```bash
# Text model (required)
ollama pull llama3

# Vision model (optional — enables image matching)
ollama pull llava
```

### System dependencies (if needed)

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

### What gets installed

| Package | Purpose |
|---|---|
| PyYAML | Configuration file parsing |
| Pillow | Image processing |
| PyMuPDF | PDF parsing and image extraction |
| python-docx | DOCX document parsing |
| python-pptx | PowerPoint slide parsing |
| pytesseract | OCR for scanned documents |
| tiktoken | Token counting for chunk splitting |
| PySide6 | Desktop graphical interface |
| Streamlit | Web-based demo interface |

---

## Usage

### CLI

```bash
# Basic — scan a folder, generate full manual
python main.py --input ./StudyMaterial

# Set detail level (1=minimal, 10=exhaustive)
python main.py --input ./StudyMaterial --depth 8

# Disable image extraction/matching
python main.py --input ./StudyMaterial --no-images

# Override the LLM model
python main.py --input ./StudyMaterial --model mistral

# Use a custom template
python main.py --input ./StudyMaterial --template my_template.md

# Specify output directory
python main.py --input ./StudyMaterial --output ./my_output/
```

> **Templates:** the default `template-lecture.md` wraps every chapter in a
> readable lecture structure — a 🟡 `[!recap]` recap callout with a local
> index, a `📝 Notes` section, and `🔗 References & Resources`. Chapter
> content is generated with Obsidian callouts (`[!formula]`, `[!theorem]`,
> `[!proof]`, `[!exercise]`, `[!question]`, `[!review]`) for maximum
> readability. Use the classic `template.md` for a plain book layout, or any
> custom template with the `{{title}}`, `{{toc}}`, `{{content}}`,
> `{{sources}}` and `{{images}}` placeholders.

### GUI

```bash
# Launch the graphical interface (omit --input)
python main.py
```

### Streamlit Web Demo (simplest)

```bash
# Launch the web interface in your browser
streamlit run streamlit_app.py
```

Opens `http://localhost:8501` with a visual interface — no desktop app needed.

### All CLI flags

| Flag | Description | Default |
|---|---|---|
| `--input DIR` | Source material directory. If omitted, the GUI launches. | — |
| `--template FILE` | Markdown template file (lecture-style by default) | `template-lecture.md` |
| `--model NAME` | Override the LLM model from config | config value |
| `--output DIR` | Output directory | `output/` |
| `--depth N` | Detail level 1–10 | config value (5) |
| `--no-images` | Disable image extraction and matching | off |
| `--syllabus FILE` | Explicit syllabus path (bypasses auto-detection) | auto |
| `--scope {full,topic}` | Full manual or single-topic focus | `full` |
| `--topic NAME` | Focus topic name (required with `--scope topic`) | — |
| `--focus-depth N` | Detail level for topic-focus mode | `--depth` value |
| `--no-interactive` | Disable interactive prompts (for scripts/CI) | off |

---

## Course Syllabus

The system supports an optional **course syllabus** (also called *programma*) that tells it which topics exist in your course and in what order they should appear.

### How it works

When a syllabus is provided (or auto-detected), the system:

1. Uses it to determine the **topic list** for the manual.
2. Orders chapters in the **same sequence** as the syllabus.
3. If a topic in the syllabus has no matching material, it is still included but marked as under-sourced.

### Providing a syllabus

**Option 1 — Manual file path (CLI):**

```bash
python main.py --input ./StudyMaterial --syllabus ./programma.txt
```

**Option 2 — Manual file path (GUI):**

Use the "Course syllabus" file picker in the GUI.

**Option 3 — Automatic detection:**

If you don't provide a syllabus, the system scans your folder for files whose name contains one of:

- `syllabus`
- `programma`
- `program`
- `course outline`
- `piano di studi`

When using the CLI with interactive prompts, it will ask:

```
Do you have a course syllabus / programma?
If yes, enter the file path.  Press Enter to skip.
```

When using the GUI, a green label appears: *"Detected: programma.pdf — use this?"*

### What happens without a syllabus

The system still works. It discovers topics from the material itself using the LLM and orders them by pedagogical logic (foundational concepts first).

### Configuration

```yaml
syllabus:
  enabled: "auto"    # "auto" | true | false
  path: null         # explicit file path, or null for auto-detection
```

| Value | Behavior |
|---|---|
| `"auto"` | Scans folder for syllabus keywords; asks if not found |
| `true` | Requires `path` to be set; skips auto-detection |
| `false` | Syllabus feature entirely disabled |

---

## Single-Topic Focus Mode

By default the system generates a **full manual** covering every topic found in your material. With focus mode you can generate a dedicated file for a **single topic** at a potentially different detail level.

### CLI

```bash
# Generate a focused chapter on "Linear Algebra" at maximum detail
python main.py --input ./StudyMaterial \
    --scope topic \
    --topic "Linear Algebra" \
    --focus-depth 10

# Focus mode without images
python main.py --input ./StudyMaterial \
    --scope topic \
    --topic "Organic Chemistry" \
    --focus-depth 6 \
    --no-images
```

### GUI

1. Select **"Focus on a specific topic"** radio button.
2. Type the topic name in the **Topic** field.
3. Adjust the **Focus depth level** slider (independent from the global depth slider).
4. Click **Generate Manual**.

### How it differs from full mode

| Aspect | Full mode | Topic mode |
|---|---|---|
| Topics covered | All detected topics | Only the named topic |
| Chapters | One per topic | Single chapter |
| Depth slider | Global depth level | Separate focus-depth slider |
| Output file | `Exam_Manual.md` | `Exam_Manual_<topic-slug>.md` |
| Material scope | All chunks | Only chunks relevant to the topic |
| Syllabus order | Respected | Ignored (manual order) |

### Configuration

```yaml
generation:
  scope: "full"           # "full" | "topic"
  focus_topic: null       # topic name string, or null
  focus_depth_level: null # 1-10, or null (falls back to depth_level)
```

---

## Table of Contents

The manual can optionally include a **table of contents** at the top, generated automatically from the chapter structure.

### Toggle via configuration

```yaml
structure:
  include_toc: true    # true = include, false = omit
```

### CLI

There is no direct CLI flag for the TOC toggle. Edit `config.yaml` or use the **"Include table of contents"** checkbox in the GUI.

### What it looks like

When enabled (`true`), the manual starts with:

```markdown
# My Exam Manual

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Calculus](#calculus)
3. [Probability Theory](#probability-theory)
...

---

## Linear Algebra
...
```

When disabled (`false`), the TOC section is omitted entirely and the manual starts directly with the first chapter.

---

## Configuration

The main configuration file is `config.yaml` at the project root. All settings have sensible defaults.

```yaml
output:
  language: "en"                        # manual language (always English)
  filename: "Exam_Manual.md"            # output filename

llm:
  host: "http://127.0.0.1:11434"       # Ollama server address
  model: "llama3"                       # text model name
  timeout: 120                          # request timeout in seconds
  max_retries: 3                        # retry count on failure

structure:
  include_toc: true                     # include table of contents

syllabus:
  enabled: "auto"                       # "auto" | true | false
  path: null                            # explicit syllabus path

generation:
  depth_level: 5                        # 1-10 detail level
  length_mode: "topic_driven"           # length scales with topics
  scope: "full"                         # "full" | "topic"
  focus_topic: null                     # topic name for focus mode
  focus_depth_level: null               # 1-10 for focus mode

images:
  extract: true                         # extract images from documents
  match_to_chapters: true               # match images to chapters via vision
  vision_model: "llava"                 # Ollama vision model
  assets_dir: "output/assets/images"    # image storage directory
  min_width: 100                        # minimum image width (px)
  min_height: 100                       # minimum image height (px)

logging:
  level: "INFO"                         # DEBUG | INFO | WARNING | ERROR
  file: "output/logs/exam_book_generator.log"
```

### Recommended models

| Purpose | Model | Pull command |
|---|---|---|
| Text generation | llama3 | `ollama pull llama3` |
| Text generation | mistral | `ollama pull mistral` |
| Text generation | qwen3 | `ollama pull qwen3` |
| Vision / image matching | llava | `ollama pull llava` |

---

## Depth Level Guide

The `depth_level` parameter (1–10) controls how much detail every chapter gets **without ever skipping topics**:

| Level | Style | Use case |
|---|---|---|
| 1–2 | Minimal but complete summary | Quick overview, last-minute review |
| 3–5 | Balanced academic explanation | General exam preparation |
| 6–8 | Detailed textbook style | Thorough exam preparation |
| 9–10 | Exhaustive with edge cases | Deep study, thesis preparation |

The length is **never fixed** — it scales with the number of topics, subtopics, and available source material.

---

## Architecture

```
                         USER
                          │
                  CLI  /  GUI  /  Web
                          │
                   Main Controller
                          │
      ┌────────────┬──────────────┬─────────────┐
      │             │              │             │
   Parsers       Pipeline       LLM Layer      Storage
```

### Pipeline (8 steps)

```
Input folder
  → 1. Scan (detect files + syllabus)
  → 2. Parse (PDF / DOCX / PPTX / TXT + image extraction)
  → 3. Deduplicate
  → 4. Chunk (AI-compatible segments)
  → 5. Topic analysis (LLM-powered)
  → 6. Outline generation (structure + index)
  → 7. Chapter generation (LLM + image matching)
  → 8. Merge → Final manual
  → 9. Validate (9 automated checks)
```

### Repository structure

```
ExamBookGenerator/
├── main.py                 # CLI entry point & pipeline orchestrator
├── streamlit_app.py        # Streamlit web demo
├── config.yaml             # default configuration
├── config.example.yaml     # documented example configuration
├── template.md             # classic chapter template
├── template-lecture.md     # lecture-style chapter template (default)
├── requirements.txt        # Python dependencies
├── install.sh              # one-command installation script
│
├── core/
│   └── models.py           # Document, Chunk, Topic, ExtractedImage
│
├── parsers/
│   ├── pdf_parser.py       # PyMuPDF + OCR fallback
│   ├── docx_parser.py      # python-docx
│   ├── pptx_parser.py      # python-pptx
│   └── ocr_parser.py       # Tesseract OCR wrapper
│
├── pipeline/
│   ├── scanner.py          # filesystem scanner + syllabus detection
│   ├── image_extractor.py  # image extraction & deduplication
│   ├── normalizer.py       # document normalization
│   ├── deduplicator.py     # content deduplication
│   ├── chunker.py          # AI-compatible chunking
│   ├── topic_analyzer.py   # LLM-powered topic discovery
│   ├── outline_generator.py # manual structure + index entries
│   ├── template_engine.py  # template loading & variable substitution
│   ├── image_matcher.py    # vision-based image placement
│   ├── chapter_generator.py # per-topic chapter generation
│   ├── validator.py        # 9 automated quality checks
│   └── merge.py            # final assembler (full / topic modes)
│
├── llm/
│   ├── ollama_client.py    # Ollama HTTP client (text + vision)
│   └── prompt_manager.py   # centralized prompt builders
│
├── storage/
│   ├── database.py         # SQLite document tracking
│   └── cache.py            # content-hash caching
│
├── utils/
│   ├── config.py           # ConfigManager (typed YAML access)
│   └── logger.py           # structured logging
│
├── gui/
│   └── app.py              # PySide6 graphical interface
│
└── tests/                  # 644 tests
```

---

## Supported Input Formats

| Format | Text extraction | Image extraction |
|---|---|---|
| PDF (text-based) | PyMuPDF | PyMuPDF |
| PDF (scanned) | Tesseract OCR | Embedded images |
| DOCX | python-docx | Embedded images |
| PPTX | python-pptx | Embedded images |
| TXT / Markdown | Direct read | — |
| PNG / JPG / TIFF | — (standalone images) | — |

---

## Output Validation

After generation the system runs **9 automated checks** and writes a `validation.json` summary:

| Check | What it verifies |
|---|---|
| Markdown structure | Valid headings, no broken syntax |
| Language | Output is in English |
| Empty chapters | No chapter is empty or too short |
| Duplicate content | No repeated paragraphs |
| Image references | All `![...]()` links point to existing files |
| TOC structure | Table of contents anchors match headings |
| Topic focus | Topic-mode output covers the named topic |
| Syllabus order | Chapters follow syllabus sequence (if provided) |
| Index entries | All `IndexEntry` anchors are unique and valid |

---

## Troubleshooting

### "Ollama model 'X' not found locally"

The configured model hasn't been pulled yet:

```bash
ollama pull llama3
```

### "No supported files found"

Your folder contains only unsupported file types. The system recognizes: PDF, DOCX, PPTX, TXT, MD, PNG, JPG, TIFF, GIF, BMP, WEBP, SVG.

### "No text content could be extracted"

All documents are image-only or corrupted. Try running Tesseract OCR separately to verify, or add text-based documents.

### Vision features disabled

The `llava` model isn't pulled. This only affects image matching — the manual is still generated:

```bash
ollama pull llava
```

### Slow generation

Generation time depends on hardware, model size, and material volume. On a modern machine:

| Source | Approximate time |
|---|---|
| 100 pages | a few minutes |
| 500 pages | tens of minutes |
| 1000+ pages | up to an hour |

---

## Testing

```bash
# Run all 644 tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run a specific test file
python -m pytest tests/test_gui.py -v
```

---

## License

MIT License — see [LICENSE](./LICENSE).

---

## Philosophy

> Transform unorganized academic material into a personalized textbook, using local artificial intelligence — nothing leaves your machine, and nothing is invented that wasn't already in your own material.
