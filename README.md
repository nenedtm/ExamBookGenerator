# ExamBookGenerator

Turn a messy folder of course material into one long, well-organized Markdown study manual, using a local AI model through Ollama.

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue)
![Ollama](https://img.shields.io/badge/AI-Ollama-green)
![Offline](https://img.shields.io/badge/Offline-Yes-success)
![MIT License](https://img.shields.io/badge/License-MIT-yellow)

## Why this exists

Anyone who's prepared for a university exam knows what the folder looks like:

```
Exam/
├── Lecture01.pdf
├── old notes.docx
├── professor slides.pptx
├── scanned textbook.pdf
├── whiteboard photos/
├── summaries/
└── a bunch of random files
```

Some of it is duplicated, some of it is scanned and unreadable as text, and none of it is in any particular order. Turning that into something you can actually study from — reading everything, cutting duplicates, figuring out what matters, giving it structure — is the kind of task that eats days.

ExamBookGenerator does that step automatically. Point it at the folder, and it outputs a single Markdown file:

```
Manuale_Esame.md
```

which, depending on how much material you throw at it, can end up hundreds of pages long.

## What it actually handles

**Disorganized input.** No expected folder structure, no naming convention. It walks the directory tree, figures out what each file is, and builds an inventory before doing anything else.

**Mixed formats.**

| Format            | Handled |
| ----------------- | ------- |
| PDF               | yes     |
| Scanned PDF       | yes (OCR) |
| DOCX              | yes     |
| PPTX              | yes     |
| TXT / Markdown    | yes     |
| Images            | yes (OCR) |

**Large documents.** An 800-page textbook can't just be dumped into a model's context window. Everything gets split into chunks that respect paragraph and section boundaries before it goes anywhere near the LLM.

**Duplicates.** Course folders accumulate `lezione.pdf`, `lezione copia.pdf`, `lezione definitiva.pdf` — near-identical files that would otherwise get processed three times. A hash pass catches exact duplicates; a similarity pass catches near-duplicates and keeps whichever version is most complete.

**Structure, not just a summary.** The AI step isn't "summarize this PDF." It looks across the whole corpus, works out what the actual topics are, how they relate, and in what order they should be taught — then writes each chapter from that outline plus the relevant source chunks.

**Everything stays local.** No document is uploaded anywhere. Generation runs against a local Ollama instance, so course material — some of which may be a professor's copyrighted slides or scans of a textbook — never leaves the machine.

## How it works, end to end

```
folder of material
        │
        ▼
filesystem scan  →  inventory.json
        │
        ▼
text extraction (per format)
        │
        ▼
normalization into a common Document type
        │
        ▼
deduplication
        │
        ▼
chunking
        │
        ▼
AI topic analysis  →  topics.json
        │
        ▼
outline generation  →  outline.md
        │
        ▼
chapter generation (one per topic)
        │
        ▼
validation
        │
        ▼
merge
        │
        ▼
Manuale_Esame.md
```

## Project layout

```
ExamBookGenerator/
├── main.py
├── config.yaml
├── template.md
├── requirements.txt
│
├── core/
│   └── models.py              # Document, Chunk, Topic, Chapter
│
├── parsers/
│   ├── pdf_parser.py          # PyMuPDF
│   ├── docx_parser.py         # python-docx
│   ├── pptx_parser.py         # python-pptx
│   ├── text_parser.py
│   └── ocr_parser.py          # Tesseract + Pillow
│
├── pipeline/
│   ├── scanner.py
│   ├── normalizer.py
│   ├── deduplicator.py
│   ├── chunker.py
│   ├── topic_analyzer.py
│   ├── outline_generator.py
│   ├── chapter_generator.py
│   ├── validator.py
│   └── merge.py
│
├── llm/
│   ├── ollama_client.py
│   └── prompt_manager.py
│
├── storage/
│   ├── database.py            # SQLite
│   └── cache.py
│
├── utils/
│   ├── logger.py
│   └── config.py
│
├── tests/
└── output/
```

## The core pieces

**`core/models.py`** defines the shared data types every other module passes around: `Document` (title, source, content, metadata), `Chunk` (a piece of a document sized for the model), `Topic`, and `Chapter`. Nothing else in the codebase should invent its own ad-hoc representation of a document.

**Parsers** all do one job: take a file, return a `Document`. Whatever format it came from is irrelevant past this point.

**The pipeline** is the sequence described above — scan, normalize, deduplicate, chunk, analyze, outline, generate, validate, merge. Each stage is its own module and can be run or tested independently.

**`llm/`** wraps Ollama. `ollama_client.py` handles the actual HTTP calls, connection checks, retries and timeouts; `prompt_manager.py` keeps every prompt template out of the Python code so tone, depth and structure can be tuned without touching logic.

## Templates

Chapter structure is controlled by `template.md`, not hardcoded:

```markdown
# {{titolo}}

## Introduzione

{{introduzione}}

## Spiegazione approfondita

{{contenuto}}

## Domande esame

{{questions}}
```

Change the template, change the shape of every generated chapter.

## Installing it

Requirements: Python 3.12+, [Ollama](https://ollama.ai) installed and running, and ideally 16 GB of RAM — generation quality and speed both depend heavily on which local model you run.

```bash
git clone https://github.com/user/ExamBookGenerator
cd ExamBookGenerator
pip install -r requirements.txt
```

## Configuring it

Everything lives in `config.yaml`:

```yaml
model:
  provider: ollama
  name: qwen3:32b

generation:
  language: italian
  depth: high
```

## Running it

```bash
python main.py \
  --input ./MaterialeEsame \
  --template template.md
```

Output lands at `output/Manuale_Esame.md`.

## How long it takes

Depends entirely on the model and the machine, but roughly:

| Material     | Time            |
| ------------ | --------------- |
| 100 pages    | a few minutes   |
| 500 pages    | tens of minutes |
| 1000+ pages  | hours           |

Runs are cached and resumable, so a crash or interruption partway through a 900-page textbook doesn't mean starting over.

## Where this could go

Video lecture support, audio transcription, semantic search over the generated manual, a web UI instead of the CLI, PDF export, auto-generated flashcards and quizzes, maybe an interactive tutor mode on top of the finished material. None of this is built yet — the current focus is getting the core pipeline (scan → parse → dedupe → generate) solid first.

## The point of it

This isn't meant to be a PDF summarizer. The goal is a manual — something with an actual structure, written for someone trying to learn the material, not just a compressed version of the source files. And it's meant to stay something you run on your own machine, on your own notes, without your course material passing through anyone else's server.
