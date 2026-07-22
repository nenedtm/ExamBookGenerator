# ExamBookGenerator

### Generatore automatico di manuali universitari Markdown tramite AI locale

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![AI](https://img.shields.io/badge/AI-Ollama-green)
![Offline](https://img.shields.io/badge/Offline-Yes-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Indice

- [Descrizione](#descrizione)
- [Obiettivo del progetto](#obiettivo-del-progetto)
- [Problema che risolve](#problema-che-risolve)
  - [1. Materiale disorganizzato](#1-materiale-disorganizzato)
  - [2. Formati diversi](#2-formati-diversi)
  - [3. Documenti enormi](#3-documenti-enormi)
  - [4. Organizzazione intelligente](#4-organizzazione-intelligente)
  - [5. Privacy](#5-privacy)
- [Funzionamento generale](#funzionamento-generale)
- [Caratteristiche principali](#caratteristiche-principali)
- [Architettura software](#architettura-software)
- [Struttura repository](#struttura-repository)
- [Moduli principali](#moduli-principali)
  - [Core](#core)
  - [Parser Layer](#parser-layer)
  - [Pipeline di elaborazione](#pipeline-di-elaborazione)
- [Sistema AI locale](#sistema-ai-locale)
- [Prompt Engine](#prompt-engine)
- [Analisi intelligente](#analisi-intelligente)
- [Creazione indice](#creazione-indice)
- [Generazione capitoli](#generazione-capitoli)
- [Template Markdown](#template-markdown)
- [Validazione](#validazione)
- [Output finale](#output-finale)
- [Installazione](#installazione)
- [Configurazione](#configurazione)
- [Utilizzo](#utilizzo)
- [Prestazioni](#prestazioni)
- [Sviluppi futuri](#sviluppi-futuri)
- [Filosofia del progetto](#filosofia-del-progetto)

---

## Descrizione

### Cos'è ExamBookGenerator

ExamBookGenerator è un sistema software progettato per trasformare una grande quantità di materiale didattico disorganizzato in un unico manuale digitale strutturato.

L'obiettivo è permettere allo studente di fornire una semplice cartella contenente tutto il materiale di un esame:

- dispense
- libri
- slide
- appunti personali
- registrazioni trascritte
- immagini
- PDF scannerizzati

e ottenere automaticamente un documento Markdown completo, approfondito e organizzato.

Il risultato finale è un file:

```
Manuale_Esame.md
```

che può contenere anche centinaia di pagine.

---

## Obiettivo del progetto

Durante la preparazione di un esame universitario il materiale è spesso frammentato:

```
Esame/
├── Lezione01.pdf
├── Appunti vecchi.docx
├── Slide professore.pptx
├── Libro scansione.pdf
├── Foto lavagna/
├── Riassunti/
└── file casuali
```

Lo studente deve manualmente:

- leggere tutto
- eliminare duplicati
- capire cosa è importante
- creare una struttura
- scrivere un riassunto
- collegare gli argomenti

Questo processo richiede giorni o settimane.

ExamBookGenerator automatizza questa fase.

---

## Problema che risolve

Il progetto affronta cinque problemi principali.

### 1. Materiale disorganizzato

Il sistema non richiede una struttura precisa.

Può ricevere:

```
Materiale/
├── lezione_finale.pdf
├── copia2.pdf
├── IMG_20250101.jpg
├── appunti.docx
└── vecchi/
    └── roba.pdf
```

e analizzarlo automaticamente.

### 2. Formati diversi

Supporta:

| Formato           | Supporto |
| ----------------- | -------- |
| PDF               | ✅        |
| PDF scannerizzati | ✅        |
| DOCX              | ✅        |
| PPTX              | ✅        |
| TXT               | ✅        |
| Markdown          | ✅        |
| Immagini          | ✅ OCR    |

### 3. Documenti enormi

Un singolo libro può avere:

- 500 pagine
- 1000 pagine
- migliaia di paragrafi

Il sistema utilizza una pipeline a blocchi per permettere l'elaborazione tramite modelli AI.

### 4. Organizzazione intelligente

L'AI analizza:

- argomenti
- collegamenti
- priorità
- struttura logica

Non produce un semplice riassunto, ma un manuale.

### 5. Privacy

Tutto funziona localmente.

I documenti personali:

- non vengono caricati online
- rimangono sul computer
- vengono elaborati tramite modelli locali

---

## Funzionamento generale

Il flusso completo è:

```
Cartella materiale
        │
        ▼
Scanner filesystem
        │
        ▼
Estrazione testo
        │
        ▼
Normalizzazione
        │
        ▼
Eliminazione duplicati
        │
        ▼
Divisione intelligente
        │
        ▼
Analisi AI
        │
        ▼
Creazione indice
        │
        ▼
Generazione capitoli
        │
        ▼
Validazione
        │
        ▼
Merge finale
        │
        ▼
Manuale_Esame.md
```

---

## Caratteristiche principali

### Gestione automatica file

Il sistema:

- ricerca ricorsivamente
- riconosce formati
- ignora file inutili
- crea inventario

### Parser modulari

Ogni formato ha il proprio modulo.

Esempio:

```
parsers/
├── pdf_parser.py
├── docx_parser.py
├── pptx_parser.py
└── ocr_parser.py
```

Ogni parser produce lo stesso oggetto, `Document`, per mantenere uniformità.

---

## Architettura software

Il progetto segue un'architettura modulare.

```
                         USER
                          │
                          ▼
                       main.py
                          │
                          ▼
                Pipeline Controller
                          │
        ┌─────────────┬───────────────┬──────────┐
        ▼             ▼               ▼          ▼
      Parser        Storage         AI Layer    Utils
```

---

## Struttura repository

```
ExamBookGenerator/
├── main.py
├── config.yaml
├── template.md
├── requirements.txt
│
├── core/
│   └── models.py
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
│   ├── database.py
│   └── cache.py
│
├── utils/
│   ├── logger.py
│   └── config.py
│
├── tests/
│
└── output/
```

---

## Moduli principali

### Core

Contiene gli oggetti condivisi.

**Document** — rappresenta un file elaborato. Contiene:

```python
Document(
    title,
    source,
    content,
    metadata
)
```

**Chunk** — rappresenta una porzione compatibile con il modello AI.

**Topic** — rappresenta un argomento.

**Chapter** — rappresenta un capitolo generato.

### Parser Layer

Responsabilità: convertire qualsiasi formato in testo.

**PDF Parser** — utilizza PyMuPDF. Funzioni: estrazione testo, metadata, numero pagine.

**DOCX Parser** — utilizza python-docx. Estrae: paragrafi, tabelle, struttura.

**PPTX Parser** — utilizza python-pptx. Estrae: slide, titoli, contenuti.

**OCR Parser** — utilizza Tesseract e Pillow. Permette di leggere: foto, scansioni, immagini.

### Pipeline di elaborazione

**1. Scanner**

Analizza la cartella:

```
input/
├── lezione.pdf
├── foto.jpg
└── libro.docx
```

creando:

```
inventory.json
```

**2. Normalizzazione**

Tutti i documenti diventano `Document`, indipendentemente dal formato.

**3. Deduplicazione**

Il sistema identifica copie identiche e versioni simili.

Esempio:

```
lezione.pdf
lezione copia.pdf
lezione definitiva.pdf
```

vengono confrontati.

**4. Chunking**

Documenti grandi (es. un libro da 900 pagine) diventano:

```
Chunk 1
Chunk 2
Chunk 3
```

---

## Sistema AI locale

### Ollama

L'AI viene gestita tramite Ollama.

Modelli compatibili:

- Qwen
- Llama
- Mistral

Esempio:

```bash
ollama run qwen3:32b
```

---

## Prompt Engine

I prompt non sono scritti nel codice. Sono gestiti da:

```
llm/prompt_manager.py
```

Permette:

- modificare stile
- cambiare profondità
- cambiare struttura

---

## Analisi intelligente

Prima di scrivere il manuale, l'AI crea:

```
topics.json
```

Esempio:

```json
[
  {
    "name": "Metabolismo",
    "sources": [1, 5, 7]
  }
]
```

---

## Creazione indice

Produce:

```
outline.md
```

Esempio:

```markdown
# Capitolo 1

## Introduzione

## Concetti fondamentali

## Applicazioni
```

---

## Generazione capitoli

Ogni capitolo viene generato separatamente.

Input: argomento + fonti + template

Output:

```
capitolo01.md
```

---

## Template Markdown

L'utente può definire la struttura.

Esempio:

```markdown
# {{titolo}}

## Introduzione

{{introduzione}}

## Spiegazione approfondita

{{contenuto}}

## Domande esame

{{questions}}
```

---

## Validazione

Prima del risultato finale il sistema controlla:

- sezioni mancanti
- errori Markdown
- capitoli vuoti
- ripetizioni

---

## Output finale

Esempio:

```
output/
└── Manuale_Esame.md
```

Struttura:

```markdown
# Biologia cellulare

## Capitolo 1

### Membrana cellulare

...

### Domande d'esame

1. ...
```

---

## Installazione

### Requisiti

- Python 3.12+
- Ollama
- almeno 16GB RAM consigliati

### Installazione

```bash
git clone https://github.com/user/ExamBookGenerator

cd ExamBookGenerator

pip install -r requirements.txt
```

---

## Configurazione

File `config.yaml`:

```yaml
model:
  provider: ollama
  name: qwen3:32b

generation:
  language: italian
  depth: high
```

---

## Utilizzo

```bash
python main.py \
  --input ./MaterialeEsame \
  --template template.md
```

Output:

```
output/Manuale_Esame.md
```

---

## Prestazioni

Dipendono dal modello.

Indicativamente:

| Materiale    | Tempo         |
| ------------ | ------------- |
| 100 pagine   | alcuni minuti |
| 500 pagine   | decine minuti |
| 1000+ pagine | ore           |

Il sistema usa:

- cache
- elaborazione incrementale
- ripresa automatica

---

## Sviluppi futuri

Possibili estensioni:

- supporto video lezioni
- trascrizione audio
- ricerca semantica
- interfaccia web
- esportazione PDF
- flashcard automatiche
- generazione quiz
- modalità tutor AI

---

## Filosofia del progetto

ExamBookGenerator non nasce come semplice riassuntore.

L'obiettivo è creare un **sistema personale di creazione manuali**, capace di trasformare materiale grezzo e disordinato in una risorsa di studio completa, mantenendo:

- controllo locale
- privacy
- personalizzazione
- qualità del contenuto
- scalabilità
