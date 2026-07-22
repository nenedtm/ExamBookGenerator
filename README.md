# ExamBookGenerator

### Local AI-powered academic material transformation system

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![AI](https://img.shields.io/badge/AI-Ollama-green)
![Offline](https://img.shields.io/badge/Execution-Local-success)
![Markdown](https://img.shields.io/badge/Output-Markdown-orange)
![Multimodal](https://img.shields.io/badge/AI-Vision-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## Overview

**ExamBookGenerator** is a fully local, AI-powered document processing system that turns a messy folder of academic material into a single, structured, customizable exam manual.

Point it at a folder containing:

- PDF books (including scanned PDFs)
- lecture notes
- PowerPoint slides
- DOCX documents
- Markdown / TXT files
- photos of whiteboards
- duplicated files and randomly organized subfolders

and it will:

1. scan every file;
2. extract text **and images**;
3. normalize and deduplicate the content;
4. split it into AI-compatible chunks;
5. identify the academic topics it covers;
6. build a logical manual structure;
7. generate detailed, exam-focused chapters — reusing real images from your material where they help explain a concept;
8. validate the result;
9. merge everything into one file.

**Output:**

```
output/
├── Exam_Manual.md
└── assets/
    └── images/
```

The manual is **always generated in English**, regardless of the language of the source material.

---

## Why not just "summarize it"?

The goal isn't a summary — it's a **generated textbook** built from your own material. There is no artificial page limit and no fixed word target: the manual's length and depth are a direct function of what you feed it and how you configure it.

| Input | Result |
|---|---|
| A single topic, 5 chunks of material | A short chapter |
| A full course, 120 chunks across dozens of topics | A large, detailed textbook |

---

## Key Features

### 🔒 Fully local — no cloud APIs

Everything runs through [Ollama](https://ollama.com). No data leaves your machine.

Suggested models:

- **Text:** Qwen, Llama, Mistral (or any Ollama-compatible chat model)
- **Vision:** Llava (or any Ollama-compatible multimodal model)

### 📏 Dynamic, topic-driven length

The final length is **not** a fixed target ("write 200 pages"). It scales automatically with:

```
number of topics + number of subtopics + amount of source material + depth_level
```

More topics → a longer manual. A short exam and a full university course never look the same size.

### 🎚️ Configurable depth level (1–10)

A single `depth_level` parameter controls how much detail every chapter gets — **without ever skipping topics**, even at the lowest setting:

| Level | Style |
|---|---|
| 1–2 | Minimal but complete summary — every topic covered, briefly |
| 3–5 | Balanced academic explanation, with examples |
| 6–8 | Detailed textbook style: mechanisms, derivations, exam-focused notes |
| 9–10 | Exhaustive: edge cases, formulas, exceptions, every relevant detail |

### 🖼️ Real images, never generated ones

The system extracts images actually embedded in your PDFs, DOCX files, and PPTX slides (diagrams, graphs, whiteboard photos, slide screenshots) and, using a local vision model, decides — per chapter — whether one of them is worth inserting and where.

**It never fabricates new images.** If nothing relevant exists in your material, no image is added.

```
Source documents
       │
       ▼
Image extraction & deduplication
       │
       ▼
Vision-model description
       │
       ▼
Relevance check per chapter
       │
       ▼
Inserted into the right chapter (or skipped)
```

---

## Architecture

```
                         USER
                          │
                     CLI  /  GUI
                          │
                   Main Controller
                          │
      ┌────────────┬──────────────┬─────────────┐
      │             │              │             │
   Parsers       Pipeline       LLM Layer      Storage
```

### Pipeline

```
Input folder → Scanner → Parsers (+ image extraction) → Normalization
   → Deduplication → Chunking → Topic analysis → Outline generation
   → Template → Image matching → Chapter generation → Validation
   → Final merge → Exam_Manual.md
```

### Repository structure

```
ExamBookGenerator/
├── main.py
├── config.yaml
├── template.md
├── requirements.txt
│
├── core/
│   └── models.py              # Document, Chunk, Topic, Chapter, ExtractedImage
│
├── parsers/
│   ├── pdf_parser.py
│   ├── docx_parser.py
│   ├── pptx_parser.py
│   ├── text_parser.py
│   └── ocr_parser.py
│
├── pipeline/
│   ├── scanner.py
│   ├── image_extractor.py     # [v2] shared image extraction/dedup
│   ├── normalizer.py
│   ├── deduplicator.py
│   ├── chunker.py
│   ├── topic_analyzer.py
│   ├── outline_generator.py
│   ├── template_engine.py
│   ├── image_matcher.py       # [v2] vision-based image placement
│   ├── chapter_generator.py
│   ├── validator.py
│   └── merge.py
│
├── llm/
│   ├── ollama_client.py       # text + vision (generate_with_image)
│   └── prompt_manager.py
│
├── storage/
│   ├── database.py
│   └── cache.py
│
├── utils/
│   ├── logger.py
│   └── config.py
│
├── gui/
│   └── app.py
│
├── tests/
└── output/
```

---

## Core data models

| Model | Purpose |
|---|---|
| `Document` | A parsed source file: metadata, extracted text, linked images |
| `Chunk` | An AI-compatible text section (books don't fit in one prompt) |
| `Topic` | An academic concept: title, description, sources, subtopic count |
| `Chapter` | A generated manual chapter: title, Markdown content, order, images |
| `ExtractedImage` | A real image pulled from the source material, with its AI-generated description |

---

## Supported input formats

| Format | Support |
|---|---|
| PDF (incl. scanned) | ✅ |
| DOCX | ✅ |
| PPTX | ✅ |
| TXT / Markdown | ✅ |
| PNG / JPG | ✅ |

---

## Configuration

`config.yaml`:

```yaml
output:
  language: "en"                    # the manual is always generated in English
  filename: "Exam_Manual.md"

generation:
  depth_level: 5                    # 1-10: 1 = minimal summary, 10 = maximum detail
  length_mode: "topic_driven"       # length always scales with topics, never fixed

images:
  extract: true
  match_to_chapters: true
  vision_model: "llava"
  assets_dir: "output/assets/images"
```

---

## Installation

Requirements: **Python 3.12+** and [Ollama](https://ollama.com) installed locally.

```bash
pip install -r requirements.txt

# text model
ollama pull qwen3

# vision model (optional — required only for image matching)
ollama pull llava
```

---

## Usage

```bash
# basic run
python main.py --input ./StudyMaterial

# custom depth level
python main.py --input ./StudyMaterial --depth 9

# disable image extraction/matching for this run
python main.py --input ./StudyMaterial --no-images
```

---

## Performance

Optimized for large, messy datasets via SQLite caching, document hashing, and incremental, chunk-based generation.

| Source size | Rough time |
|---|---|
| 100 pages | minutes |
| 500 pages | tens of minutes |
| 1000+ pages | hours, depending on hardware and model |

---

## Development approach

This project is built incrementally through **28 independent, self-contained development prompts** — one per module — so that each one can be handed to a fresh LLM chat with zero prior context and still produce correct, integration-ready code. The step order follows real module dependencies (no step ever needs a file produced by a later step), so following them 1 → 28 in order yields the complete, working project.

| Phase | Steps |
|---|---|
| Foundation | 1–4 |
| Document processing & image extraction | 5–11 |
| Data pipeline | 12–15 |
| AI core (Ollama client, prompts, topics, outline) | 16–19 |
| Content & image assembly | 20–24 |
| User interface | 25–26 |
| Testing & packaging | 27–28 |

The full prompt series lives in [`ExamBookGenerator___Serie_Completa_di_Prompt_Indipendenti.md`](./ExamBookGenerator___Serie_Completa_di_Prompt_Indipendenti.md).

---

## Roadmap / possible extensions

- Automatic flashcards and quizzes
- Spaced-repetition integration
- PDF export of the final manual
- Web interface
- Semantic search over your own material
- Voice lecture transcription

---

## Philosophy

> Transform unorganized academic material into a personalized textbook, using local artificial intelligence — nothing leaves your machine, and nothing is invented that wasn't already in your own material.
