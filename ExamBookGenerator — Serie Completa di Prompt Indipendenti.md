Ogni prompt ha questa struttura fissa:

- **Titolo STEP**
- **Contesto progetto completo**
- **Obiettivo dello step**
- **File da creare**
- **Interfacce**
- **Vincoli tecnici**
- **Output richiesto all'IA**
- **File da allegare dagli step precedenti**

Copia il blocco di testo di uno STEP in una chat nuova per generare il relativo codice.

---

## Modifiche v2 rispetto alla versione originale

Questa versione introduce 3 cambiamenti trasversali, validi per tutti gli step che generano o gestiscono contenuto:

1. **Lingua fissa: inglese.** I documenti sorgente sono in inglese e il manuale generato (`Manuale_Esame.md` → ora `Exam_Manual.md`) deve essere sempre in inglese, indipendentemente dalla lingua dei prompt di sviluppo (che restano in italiano solo come istruzioni per l'IA che scrive il codice).
2. **Lunghezza guidata dagli argomenti, non fissa.** Non esiste più un target di lunghezza arbitrario ("manuale molto lungo"). La lunghezza del manuale è una funzione diretta del numero di topic/subtopic individuati da `topic_analyzer` e della quantità di chunk associati a ciascuno: più argomenti → manuale più lungo, in modo proporzionale e automatico.
3. **Livello di approfondimento configurabile (1–10).** Un nuovo parametro `depth_level` (da `config.yaml` o da CLI/GUI) controlla il livello di dettaglio di ogni capitolo, da 1 (riassunto minimo ma che copre comunque tutti gli argomenti, nessuno escluso) a 10 (massimo dettaglio, nessun dettaglio tralasciato).
4. **Inserimento immagini quando utili a spiegare un concetto.** I parser PDF/DOCX/PPTX estraggono anche le immagini incorporate (non solo il testo), passandole a un modulo dedicato (`pipeline/image_extractor.py`, STEP 6) che le salva in `output/assets/images/`. Un altro modulo (`pipeline/image_matcher.py`, STEP 21) usa un modello Ollama multimodale (es. `llava`) per descrivere ogni immagine estratta e decidere, per ciascun capitolo, se e quale immagine è pertinente da inserire. Il manuale non genera immagini nuove: **riutilizza solo immagini realmente presenti nel materiale originale** (diagrammi, screenshot di slide, foto di lavagne, grafici nei PDF).

Questi 4 punti sono richiamati nei singoli step tramite il tag **`[v2]`**.

Nota di manutenzione (indipendenza dei prompt): ogni STEP, da 1 a 28, include ora il proprio richiamo esplicito ai 4 punti v2, così da poter essere incollato in una chat/LLM completamente nuovo e restare comprensibile e coerente in autonomia, senza dover allegare README o altri step per capire il contesto generale.

Nota di manutenzione (ordine cronologico): la numerazione degli STEP è stata rivista perché l'ordine originale non seguiva le reali dipendenze tra moduli. In particolare:

- `pipeline/image_extractor.py` (immagini) era STEP 27, cioè dopo test (25) e packaging (26), ma i parser PDF/DOCX/PPTX (allora STEP 6-8) lo richiedevano già per salvare le immagini estratte: ora è STEP 6, creato prima dei parser (STEP 7-9), che lo richiamano direttamente invece di duplicarne la logica.
- `pipeline/image_matcher.py` era STEP 28 (ultimo in assoluto), ma il generatore di capitoli (allora STEP 19) e il merge finale (allora STEP 22) lo richiedevano già come dipendenza: ora è STEP 21, creato prima del generatore di capitoli (ora STEP 22).
- Il supporto per l'invio di immagini a Ollama (`generate_with_image`) è stato spostato dentro il client Ollama fin dal suo step di creazione (ora STEP 16), invece di essere aggiunto in un secondo momento dallo step dell'image matcher.
- Test automatici e packaging, che per natura devono coprire e distribuire l'intero progetto, sono ora davvero gli ultimi due step (27 e 28) invece di essere seguiti da altri due moduli funzionali.

Seguendo gli STEP nell'ordine numerico 1→28 si ottiene quindi il progetto completo, senza dover tornare indietro per dipendenze mancanti.

---

## Indice

- [PARTE 1/4](#parte-14)
    - [ ] [STEP 1 — Creazione struttura base progetto](#step-1-creazione-struttura-base-progetto)
    - [ ] [STEP 2 — Gestione configurazione](#step-2-gestione-configurazione)
    - [ ] [STEP 3 — Sistema logging professionale](#step-3-sistema-logging-professionale)
    - [ ] [STEP 4 — Modelli dati centrali](#step-4-modelli-dati-centrali)
    - [ ] [STEP 5 — Scanner cartella documenti](#step-5-scanner-cartella-documenti)
    - [ ] [STEP 6 — Estrazione immagini dai documenti [v2]](#step-6-estrazione-immagini-dai-documenti-v2)
    - [ ] [STEP 7 — Parser PDF](#step-7-parser-pdf)
- [PARTE 2/4](#parte-24)
    - [ ] [STEP 8 — Parser DOCX](#step-8-parser-docx)
    - [ ] [STEP 9 — Parser PPTX](#step-9-parser-pptx)
    - [ ] [STEP 10 — Parser TXT e Markdown](#step-10-parser-txt-e-markdown)
    - [ ] [STEP 11 — OCR immagini e PDF scannerizzati](#step-11-ocr-immagini-e-pdf-scannerizzati)
    - [ ] [STEP 12 — Normalizzazione documenti](#step-12-normalizzazione-documenti)
    - [ ] [STEP 13 — Storage e cache locale](#step-13-storage-e-cache-locale)
    - [ ] [STEP 14 — Sistema di deduplicazione documenti](#step-14-sistema-di-deduplicazione-documenti)
- [PARTE 3/4](#parte-34)
    - [ ] [STEP 15 — Chunking intelligente per LLM](#step-15-chunking-intelligente-per-llm)
    - [ ] [STEP 16 — Client Ollama locale](#step-16-client-ollama-locale)
    - [ ] [STEP 17 — Gestore prompt AI](#step-17-gestore-prompt-ai)
    - [ ] [STEP 18 — Analisi argomenti automatica](#step-18-analisi-argomenti-automatica)
    - [ ] [STEP 19 — Generazione indice manuale](#step-19-generazione-indice-manuale)
    - [ ] [STEP 20 — Sistema template Markdown](#step-20-sistema-template-markdown)
    - [ ] [STEP 21 — Image matcher e inserimento nel manuale [v2]](#step-21-image-matcher-e-inserimento-nel-manuale-v2)
- [PARTE 4/4](#parte-44)
    - [ ] [STEP 22 — Generatore capitoli](#step-22-generatore-capitoli)
    - [ ] [STEP 23 — Validatore output](#step-23-validatore-output)
    - [ ] [STEP 24 — Merge manuale finale](#step-24-merge-manuale-finale)
    - [ ] [STEP 25 — CLI completa](#step-25-cli-completa)
    - [ ] [STEP 26 — GUI](#step-26-gui)
    - [ ] [STEP 27 — Test automatici](#step-27-test-automatici)
    - [ ] [STEP 28 — Packaging finale](#step-28-packaging-finale)

---

# PARTE 1/4

## STEP 1 — Creazione struttura base progetto

````text
Sei un software architect senior specializzato in Python.

Devi sviluppare un componente del progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO
==================================================

ExamBookGenerator è un'applicazione locale che deve trasformare una cartella disordinata di materiale universitario (in lingua inglese) in un unico manuale Markdown in **lingua inglese**, la cui lunghezza si adatta automaticamente al numero di argomenti individuati (nessuna lunghezza fissa) e il cui livello di dettaglio è controllato da un parametro `depth_level` (1-10, 1 = riassunto minimo ma completo di tutti gli argomenti, 10 = massimo dettaglio). [v2]

La cartella di input può contenere:

- PDF
- PDF scannerizzati
- DOCX
- PPTX
- TXT
- Markdown
- immagini
- sottocartelle casuali
- file duplicati

Il programma deve:

1. analizzare tutti i file;
2. estrarre il contenuto;
3. organizzare gli argomenti;
4. eliminare duplicati;
5. generare capitoli;
6. seguire un template Markdown;
7. creare un unico file:

Exam_Manual.md


Il sistema deve funzionare:

- completamente in locale;
- senza servizi cloud obbligatori;
- usando Ollama come motore AI.


==================================================
ARCHITETTURA GENERALE
==================================================

Il progetto finale avrà questa struttura:

ExamBookGenerator/

│
├── main.py
├── requirements.txt
├── config.yaml
├── template.md
│
├── core/
│
├── parsers/
│
├── pipeline/
│
├── llm/
│
├── storage/
│
├── utils/
│
├── tests/
│
└── output/


==================================================
STEP CORRENTE
==================================================

STEP 1

Creazione struttura base del progetto.


==================================================
OBIETTIVO DELLO STEP
==================================================

Creare lo scheletro iniziale funzionante.

Il progetto deve avere:

- struttura cartelle;
- configurazione globale;
- sistema logging;
- requirements;
- file principali vuoti ma correttamente collegati.


==================================================
FILE DA CREARE
==================================================

Creare:

requirements.txt

config.yaml

main.py

`config.yaml` deve includere fin da subito, con valori di default, questi campi [v2]:

```yaml
output:
  language: "en"          # lingua fissa del manuale generato, indipendente dai documenti sorgente
  filename: "Exam_Manual.md"

generation:
  depth_level: 5           # 1-10: 1 = riassunto minimo (ma copre tutti gli argomenti), 10 = massimo dettaglio
  length_mode: "topic_driven"  # la lunghezza NON è mai fissa, dipende dal numero di topic/subtopic

images:
  extract: true             # estrarre immagini incorporate da PDF/DOCX/PPTX
  match_to_chapters: true   # usare un modello vision per decidere dove inserirle
  vision_model: "llava"     # nome modello Ollama multimodale
  assets_dir: "output/assets/images"
````

Creare cartelle:

core/

parsers/

pipeline/

llm/

storage/

utils/

tests/

output/

Creare anche:

utils/logger.py

utils/config.py

# ================================================== REQUISITI TECNICI

Usare:

Python 3.12+

Il codice deve avere:

- type hints;
- docstring;
- gestione errori;
- logging professionale.

La configurazione deve essere caricata da YAML.

Non inserire ancora:

- parser;
- AI;
- Ollama;
- GUI.

Questo step deve creare solamente la base.

# ================================================== COMPORTAMENTO RICHIESTO

Quando eseguo:

python main.py

deve:

- caricare configurazione;
- inizializzare logging;
- mostrare un messaggio che il sistema è avviato.

# ================================================== OUTPUT RICHIESTO

Fornisci:

1. breve spiegazione architetturale;
    
2. struttura cartelle;
    
3. codice completo di ogni file creato;
    
4. istruzioni installazione;
    
5. comando per testare.
    

Non fornire pseudocodice.

Non lasciare TODO.

Non scrivere parti mancanti.

# ================================================== FILE DA ALLEGARE DAGLI STEP PRECEDENTI

Nessuno.

Questo è il primo modulo del progetto.

````

---

## STEP 2 — Gestione configurazione

```text
Sei un software engineer senior Python.

Stai lavorando al progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE
==================================================

ExamBookGenerator è un software locale che riceve una cartella piena di materiale universitario e genera automaticamente un manuale Markdown approfondito.

Il progetto deve essere:

- modulare;
- offline;
- Python 3.12+;
- compatibile con Ollama;
- facilmente estendibile.


==================================================
STRUTTURA PROGETTO PREVISTA
==================================================

ExamBookGenerator/

├── main.py
├── config.yaml

├── core/

├── parsers/

├── pipeline/

├── llm/

├── storage/

├── utils/

│   ├── config.py
│   └── logger.py

├── tests/


==================================================
STEP CORRENTE
==================================================

STEP 2

Sistema gestione configurazione.


==================================================
OBIETTIVO
==================================================

Creare un sistema centralizzato per leggere e validare:

config.yaml


Il resto dell'applicazione non deve leggere direttamente il file YAML.


==================================================
FILE DA CREARE/MODIFICARE
==================================================

Creare o completare:

utils/config.py


Aggiornare:

config.yaml


==================================================
REQUISITI
==================================================

Il modulo deve:

- caricare YAML;
- validare valori;
- fornire accesso tipizzato;
- gestire configurazione mancante;
- gestire errori;
- permettere override futuri.

Validazioni aggiuntive richieste [v2]:

- `generation.depth_level` deve essere un intero tra 1 e 10 incluso; se fuori range, sollevare `ConfigValidationError` con messaggio esplicativo, oppure clampare al valore più vicino (scegliere un comportamento e documentarlo).
- `output.language` deve essere presente (default `"en"`); se il progetto è usato solo in inglese, la validazione deve comunque accettare il campo senza richiedere logica di traduzione.
- `images.vision_model` è opzionale: se il modello indicato non è disponibile su Ollama, il resto della pipeline deve continuare a funzionare (fallback: nessuna immagine inserita, solo warning nel log), non deve essere un errore bloccante.


Creare una classe:

ConfigManager


con metodi chiari.


==================================================
OUTPUT ATTESO
==================================================

Esempio:

config = ConfigManager()

model = config.get("model.name")


==================================================
VINCOLI
==================================================

Non modificare altri moduli.

Non implementare AI.

Non implementare parser.


==================================================
OUTPUT RICHIESTO
==================================================

Fornisci:

- spiegazione;
- codice completo;
- esempio utilizzo;
- test minimo.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI
==================================================

Necessari:

config.yaml

utils/logger.py

main.py

Se non disponibili, crea compatibilità senza rompere il progetto.
````

---

## STEP 3 — Sistema logging professionale

```text
Sei un software engineer senior Python.

Stai sviluppando il progetto:

ExamBookGenerator


==================================================
CONTESTO PROGETTO
==================================================

ExamBookGenerator è un'applicazione locale che trasforma una cartella disordinata di materiale universitario in un manuale Markdown completo e approfondito.

Il software deve:

- leggere PDF, DOCX, PPTX, TXT, Markdown e immagini;
- organizzare automaticamente gli argomenti;
- usare Ollama come modello AI locale;
- generare un file finale output/Exam_Manual.md, sempre in lingua inglese, con lunghezza non fissa (dipende dai topic/subtopic) e livello di dettaglio controllato da `generation.depth_level` (1-10) [v2].


Il progetto deve essere:

- Python 3.12+
- modulare
- offline
- manutenibile
- professionale


==================================================
STRUTTURA PROGETTO
==================================================

ExamBookGenerator/

├── main.py
├── config.yaml

├── core/

├── parsers/

├── pipeline/

├── llm/

├── storage/

├── utils/

│   ├── logger.py
│   └── config.py

├── tests/


==================================================
STEP CORRENTE
==================================================

STEP 3

Sistema logging professionale.


==================================================
OBIETTIVO
==================================================

Creare un sistema centralizzato di logging utilizzabile da tutti i moduli.


==================================================
FILE DA CREARE/MODIFICARE
==================================================

Creare o completare:

utils/logger.py


Eventualmente aggiornare:

config.yaml


==================================================
REQUISITI
==================================================

Il logger deve:

- usare logging standard Python;
- supportare livelli INFO, DEBUG, WARNING, ERROR;
- scrivere su console;
- scrivere su file;
- creare automaticamente cartella logs;
- evitare duplicazione degli handler;
- essere importabile da qualsiasi modulo.


Creare funzione:

get_logger(nome_modulo)


Esempio:

logger = get_logger(__name__)

logger.info("Avvio parser PDF")


==================================================
VINCOLI
==================================================

Non creare GUI.

Non creare parser.

Non creare moduli AI.


==================================================
OUTPUT RICHIESTO
==================================================

Fornisci:

1. spiegazione;
2. codice completo;
3. esempio utilizzo;
4. test minimo.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI
==================================================

Necessari:

utils/config.py

config.yaml

main.py
```

---

## STEP 4 — Modelli dati centrali

```text
Sei un software architect senior Python.

Devi creare un modulo del progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE
==================================================

Il programma finale deve prendere una cartella con materiale universitario disordinato e produrre un manuale Markdown enorme.

Il sistema deve gestire:

- file originali;
- testo estratto;
- documenti normalizzati;
- argomenti;
- capitoli;
- risultati generati.


Tutti i moduli devono comunicare usando strutture dati comuni.


==================================================
STRUTTURA PROGETTO
==================================================

ExamBookGenerator/

├── core/

│   └── models.py


├── parsers/

├── pipeline/

├── llm/

├── storage/

├── utils/


==================================================
STEP CORRENTE
==================================================

STEP 4

Creazione modelli dati centrali.


==================================================
OBIETTIVO
==================================================

Creare le classi dati condivise dall'intero programma.


==================================================
FILE DA CREARE
==================================================

Creare:

core/models.py


==================================================
MODELLI NECESSARI
==================================================

Creare almeno:


Document

Rappresenta un documento estratto.

Campi:

- id
- title
- source_path
- file_type
- content
- metadata
- images: list[str] = [] — id delle ExtractedImage trovate in questo documento [v2]


Chunk

Rappresenta una porzione di documento.

Campi:

- id
- document_id
- content
- position


Topic

Rappresenta un argomento.

Campi:

- name
- description
- related_documents
- subtopic_count: int = 0 — usato per stimare la lunghezza del capitolo generato [v2]


Chapter

Rappresenta un capitolo generato.

Campi:

- title
- content
- order
- images: list[str] = [] — id delle ExtractedImage effettivamente inserite nel capitolo [v2]


ExtractedImage [v2]

Rappresenta un'immagine estratta da un documento sorgente (non generata dall'IA).

Campi:

- id
- source_document_id
- file_path — percorso in output/assets/images/
- page_or_slide: int | None
- caption: str | None — didascalia originale se presente nel documento
- ai_description: str | None — descrizione generata dal modello vision (popolata da image_matcher)
- width, height


==================================================
REQUISITI TECNICI
==================================================

Usare:

- dataclasses oppure pydantic;
- type hints;
- validazione;
- docstring.


Il modulo non deve conoscere:

- AI;
- parser;
- filesystem.


==================================================
OUTPUT RICHIESTO
==================================================

Fornisci:

- spiegazione;
- codice completo;
- esempio utilizzo;
- test.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI
==================================================

Necessari:

utils/config.py

utils/logger.py
```

---

## STEP 5 — Scanner cartella documenti

```text
Sei un software engineer senior Python.

Stai lavorando al progetto:

ExamBookGenerator


==================================================
CONTESTO
==================================================

Il software deve analizzare una cartella completamente disordinata contenente materiale universitario.


Esempio:

Materiale/

lezione.pdf

vecchi_appunti/

foto.png

capitolo.docx

copia_finale.pdf


Il sistema deve trovare tutto automaticamente.


==================================================
STEP CORRENTE
==================================================

STEP 5

Scanner filesystem.


==================================================
OBIETTIVO
==================================================

Creare un modulo che:

- riceve una cartella;
- esplora ricorsivamente;
- trova file supportati;
- crea inventario.


==================================================
FILE DA CREARE
==================================================

Creare:

pipeline/scanner.py


==================================================
FUNZIONI RICHIESTE
==================================================

Implementare:


scan_directory(path)

Restituisce lista documenti trovati.


detect_file_type(path)

Determina:

- pdf
- docx
- pptx
- txt
- md
- image


generate_inventory()

Salva:

storage/inventory.json


==================================================
REQUISITI

Deve gestire:

- cartelle inesistenti;
- permessi mancanti;
- file corrotti;
- estensioni sconosciute.


Usare:

- pathlib;
- logging.


==================================================
OUTPUT RICHIESTO

Fornisci:

- spiegazione;
- codice completo;
- esempio comando;
- test.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI
==================================================

Necessari:

core/models.py

utils/logger.py

utils/config.py
```

---

## STEP 6 — Estrazione immagini dai documenti [v2]

```text
Sei un software engineer senior Python specializzato in elaborazione documentale.

Progetto:

ExamBookGenerator


==================================================
CONTESTO
==================================================

I parser PDF, DOCX e PPTX (STEP 7, 8, 9, ancora da creare) dovranno estrarre anche le immagini incorporate nei documenti, non solo il testo. Per evitare di duplicare la stessa logica di salvataggio/deduplicazione/filtraggio in ognuno dei tre parser, questo modulo condiviso viene creato PRIMA dei parser stessi: i parser successivi dovranno limitarsi a passare i bytes grezzi delle immagini trovate a `pipeline.image_extractor.save_extracted_image()` invece di reimplementare questa logica.

Il manuale non deve mai generare immagini nuove: può solo riutilizzare immagini realmente presenti nel materiale universitario originale (diagrammi, grafici, foto di slide/lavagne, screenshot).


==================================================
STEP CORRENTE
==================================================

STEP 6

Gestione centralizzata immagini estratte.


==================================================
OBIETTIVO
==================================================

Creare un modulo che riceve immagini grezze (bytes) dai parser e produce oggetti ExtractedImage salvati su disco in modo consistente.


==================================================
FILE DA CREARE
==================================================

Creare:

pipeline/image_extractor.py


==================================================
FUNZIONI RICHIESTE
==================================================

save_extracted_image(raw_bytes, source_document_id, page_or_slide, caption=None) -> ExtractedImage

- calcola hash dei bytes (per deduplicare immagini identiche riutilizzate più volte, es. loghi ripetuti su ogni slide);
- scarta immagini sotto una soglia minima di dimensioni (default 100x100 px, configurabile);
- salva il file in `output/assets/images/` con nome basato sull'hash;
- restituisce un oggetto ExtractedImage popolato.

deduplicate_images(images: list[ExtractedImage]) -> list[ExtractedImage]

- rimuove ExtractedImage con hash identico, mantenendo solo la prima occorrenza e aggiornando i riferimenti.


==================================================
REQUISITI TECNICI
==================================================

Usare:

- Pillow per leggere dimensioni/validità immagine;
- hashlib per il deduplicating hash;
- core.models.ExtractedImage;
- logging.

Gestire:

- immagini corrotte o illeggibili (scartare con warning nel log, non bloccare la pipeline);
- cartella assets mancante (crearla automaticamente).


==================================================
OUTPUT RICHIESTO
==================================================

Fornisci:

- spiegazione;
- codice completo;
- esempio di utilizzo che i futuri parsers/pdf_parser.py, docx_parser.py e pptx_parser.py (STEP 7, 8, 9) dovranno seguire per chiamare questo modulo;
- test.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI
==================================================

Necessari:

core/models.py

utils/logger.py

utils/config.py
```

---

## STEP 7 — Parser PDF

```text
Sei un software engineer senior Python.

Stai sviluppando:

ExamBookGenerator


==================================================
CONTESTO
==================================================

Il programma deve estrarre testo da qualsiasi materiale universitario.


==================================================
STEP CORRENTE
==================================================

STEP 7

Parser PDF.


==================================================
OBIETTIVO

Creare un modulo che legge PDF normali.


==================================================
FILE DA CREARE

Creare:

parsers/pdf_parser.py


==================================================
REQUISITI

Usare:

PyMuPDF


Il parser deve:

- aprire PDF;
- leggere tutte le pagine;
- estrarre testo;
- restituire Document;
- salvare metadata;
- estrarre anche le immagini incorporate in ogni pagina (non solo il testo): per ciascuna immagine grezza trovata, chiamare `pipeline.image_extractor.save_extracted_image(raw_bytes, source_document_id, page_or_slide, caption)` (STEP 6, già disponibile) invece di reimplementare salvataggio/deduplicazione/filtraggio; la funzione restituisce già un `ExtractedImage` pronto con `page_or_slide` valorizzato e, se disponibile, il testo immediatamente vicino come `caption` [v2].


Gestire:

- PDF vuoti;
- PDF corrotti;
- errori lettura;
- pagine senza immagini (nessun errore, semplicemente nessuna ExtractedImage prodotta) [v2].


==================================================
REQUISITI TECNICI AGGIUNTIVI [v2]

Usare `page.get_images()` di PyMuPDF per estrarre i bytes grezzi delle immagini incorporate pagina per pagina, poi passarli a `pipeline.image_extractor.save_extracted_image()` (STEP 6): è quella funzione, non questo parser, a occuparsi dello scarto delle immagini troppo piccole (soglia di default 100x100 px) e della deduplicazione.


==================================================
INTERFACCIA

Funzione:


parse_pdf(path) -> Document

Document.images contiene gli id delle ExtractedImage trovate; le ExtractedImage stesse vengono restituite insieme al Document (es. tramite una tupla `(Document, list[ExtractedImage])`).


==================================================
OUTPUT RICHIESTO

Fornisci:

- spiegazione;
- requirements aggiuntivi;
- codice completo;
- esempio utilizzo;
- test.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI

Necessari:

core/models.py

utils/logger.py

pipeline/image_extractor.py (STEP 6)
```

---

# PARTE 2/4

## STEP 8 — Parser DOCX

```text
Sei un software engineer senior Python.

Stai sviluppando il progetto:

ExamBookGenerator


==================================================
CONTESTO PROGETTO
==================================================

ExamBookGenerator è un'applicazione locale che trasforma materiale universitario disordinato in un manuale Markdown completo.

Il software deve leggere diversi formati:

- PDF
- DOCX
- PPTX
- TXT
- Markdown
- immagini


Ogni parser deve convertire il proprio formato in un oggetto Document comune.


==================================================
STEP CORRENTE
==================================================

STEP 8

Parser documenti DOCX.


==================================================
OBIETTIVO
==================================================

Creare un modulo capace di leggere file Word e trasformarli in Document.


==================================================
FILE DA CREARE

Creare:

parsers/docx_parser.py


==================================================
REQUISITI TECNICI

Usare:

python-docx


Il parser deve:

- aprire file DOCX;
- estrarre paragrafi;
- estrarre testo dalle tabelle;
- mantenere ordine del contenuto;
- raccogliere metadata;
- estrarre le immagini incorporate (accessibili tramite le relationship del pacchetto DOCX, `document.part.related_parts`) e passarle, come bytes grezzi, a `pipeline.image_extractor.save_extracted_image()` (STEP 6, già disponibile), che si occupa di salvataggio, scarto immagini sotto 100x100 px e deduplicazione [v2].


Output:

Document (con Document.images valorizzato), più la lista di ExtractedImage prodotte [v2]


Interfaccia:

parse_docx(path) -> tuple[Document, list[ExtractedImage]]


==================================================
GESTIONE ERRORI

Gestire:

- file inesistente;
- DOCX corrotto;
- documento vuoto.


Usare:

- logging;
- type hints.


==================================================
OUTPUT RICHIESTO

Fornisci:

1. spiegazione;
2. requirements necessari;
3. codice completo;
4. esempio utilizzo;
5. test minimo.


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI

Necessari:

core/models.py

utils/logger.py

pipeline/image_extractor.py (STEP 6)
```

---

## STEP 9 — Parser PPTX

```text
Sei un software engineer senior Python.

Stai lavorando su:

ExamBookGenerator


==================================================
CONTESTO

Il sistema deve trasformare slide universitarie in testo utilizzabile per generare un manuale.


==================================================
STEP CORRENTE

STEP 9

Parser PowerPoint.


==================================================
OBIETTIVO

Creare parser per file PPTX.


==================================================
FILE DA CREARE

Creare:

parsers/pptx_parser.py


==================================================
REQUISITI

Usare:

python-pptx


Il parser deve estrarre:

- titolo slide;
- testo;
- note se disponibili;
- numero slide;
- immagini incorporate in ciascuna slide (`slide.shapes`, filtrando `shape.shape_type == PICTURE`): per ciascuna, chiamare `pipeline.image_extractor.save_extracted_image()` (STEP 6, già disponibile) passando `page_or_slide` = numero slide; è quella funzione a occuparsi di salvataggio, scarto immagini sotto 100x100 px e deduplicazione [v2].


Restituire:

Document (con Document.images valorizzato), più la lista di ExtractedImage prodotte [v2]


Interfaccia:

parse_pptx(path) -> tuple[Document, list[ExtractedImage]]


==================================================
GESTIONE ERRORI

Gestire:

- file danneggiati;
- slide vuote;
- contenuti mancanti.


==================================================
OUTPUT

Fornire:

- spiegazione;
- codice completo;
- esempio;
- test.


==================================================
FILE DA ALLEGARE

Necessari:

core/models.py

utils/logger.py

pipeline/image_extractor.py (STEP 6)
```

---

## STEP 10 — Parser TXT e Markdown

```text
Sei un software engineer senior Python.


Progetto:

ExamBookGenerator


==================================================
STEP CORRENTE

STEP 10

Parser testo semplice.


==================================================
OBIETTIVO

Creare parser per:

- TXT
- MD


==================================================
FILE DA CREARE

Creare:

parsers/text_parser.py


==================================================
REQUISITI

Il parser deve:

- leggere encoding diversi;
- gestire UTF-8;
- preservare Markdown;
- creare Document.


Interfaccia:

parse_text(path) -> Document


Gestire:

- file vuoti;
- caratteri corrotti;
- encoding errato.


==================================================
OUTPUT

Fornisci:

- codice completo;
- esempio;
- test.


==================================================
FILE PRECEDENTI NECESSARI

core/models.py

utils/logger.py
```

---

## STEP 11 — OCR immagini e PDF scannerizzati

```text
Sei un software engineer senior specializzato in sistemi documentali.


Progetto:

ExamBookGenerator


==================================================
CONTESTO

Alcuni materiali universitari saranno:

- foto di lavagne;
- scansioni;
- PDF senza testo.


Serve OCR.


==================================================
STEP CORRENTE

STEP 11

Sistema OCR.


==================================================
OBIETTIVO

Creare modulo che converte immagini in testo.


==================================================
FILE DA CREARE

Creare:

parsers/ocr_parser.py


==================================================
TECNOLOGIE

Usare:

- pytesseract
- Pillow


Supportare:

- PNG
- JPG
- JPEG


==================================================
FUNZIONAMENTO

Input:

immagine


Output:

Document


Estrarre:

- testo OCR;
- nome file;
- metadata.


==================================================
GESTIONE ERRORI

Gestire:

- OCR non installato;
- immagine corrotta;
- testo vuoto.


==================================================
OUTPUT

Fornire:

- codice completo;
- installazione dipendenze;
- esempio;
- test.


==================================================
FILE PRECEDENTI NECESSARI

core/models.py

utils/logger.py
```

---

## STEP 12 — Normalizzazione documenti

```text
Sei un software architect senior Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO

I parser producono documenti provenienti da fonti diverse.

Prima di qualsiasi analisi devono essere uniformati.


==================================================
STEP CORRENTE

STEP 12

Normalizzazione.


==================================================
OBIETTIVO

Creare modulo che converte qualsiasi output parser in Document standard.


==================================================
FILE DA CREARE

Creare:

pipeline/normalizer.py


==================================================
INPUT

Lista di documenti grezzi:

[
{
filename:"",
text:""
}
]


==================================================
OUTPUT

Lista:

Document


==================================================
FUNZIONALITA'

Implementare:

- pulizia testo;
- rimozione spazi inutili;
- normalizzazione caratteri;
- creazione titolo automatico;
- metadata.


==================================================
REQUISITI

Usare:

- core.models.Document
- logging


==================================================
OUTPUT RICHIESTO

Fornire:

- codice completo;
- esempio;
- test.


==================================================
FILE PRECEDENTI NECESSARI

core/models.py

parsers/*

utils/logger.py
```

---

## STEP 13 — Storage e cache locale

```text
Sei un software engineer senior Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO

Il programma lavorerà su migliaia di pagine.

Non deve ricominciare ogni volta da zero.


==================================================
STEP CORRENTE

STEP 13

Sistema storage locale.


==================================================
OBIETTIVO

Creare sistema per salvare:

- documenti elaborati;
- risultati parser;
- cache AI.


==================================================
FILE DA CREARE

Creare:

storage/database.py

storage/cache.py


==================================================
REQUISITI

Usare:

SQLite


Funzioni richieste:


save_document()

load_document()

save_cache()

get_cache()


Il sistema deve:

- creare database automaticamente;
- evitare duplicazioni;
- gestire errori.


==================================================
OUTPUT

Fornire:

- codice completo;
- schema database;
- esempio utilizzo;
- test.


==================================================
FILE DA ALLEGARE

Necessari:

core/models.py

utils/logger.py
```

---

## STEP 14 — Sistema di deduplicazione documenti

```text
Sei un software engineer senior Python specializzato in sistemi documentali.

Progetto:

ExamBookGenerator


==================================================
CONTESTO PROGETTO
==================================================

Il software deve prendere una cartella disordinata con materiale universitario.

È normale trovare:

- copie dello stesso PDF;
- versioni diverse;
- appunti duplicati;
- file rinominati.


Prima della generazione del manuale bisogna eliminare ridondanze.


==================================================
STEP CORRENTE

STEP 14

Sistema deduplicazione.


==================================================
OBIETTIVO

Creare un modulo che identifica documenti duplicati.


==================================================
FILE DA CREARE

Creare:

pipeline/deduplicator.py


==================================================
REQUISITI

Implementare:

- hash file;
- confronto contenuto;
- similarità testo;
- scelta documento migliore.


Strategia:

1. eliminare duplicati identici tramite hash;
2. confrontare documenti simili tramite similarità;
3. mantenere il documento più completo.


==================================================
INPUT

Lista:

Document


==================================================
OUTPUT

Lista:

Document senza duplicati.


==================================================
TECNICHE

Usare:

- hashlib;
- difflib oppure cosine similarity.


==================================================
OUTPUT RICHIESTO

Fornire:

- spiegazione;
- codice completo;
- esempio;
- test.


==================================================
FILE PRECEDENTI NECESSARI

core/models.py

pipeline/normalizer.py

storage/database.py

utils/logger.py
```

---

# PARTE 3/4

## STEP 15 — Chunking intelligente per LLM

```text
Sei un software engineer senior Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO

I documenti possono essere enormi.

Un libro PDF da 800 pagine non può essere inviato interamente al modello AI.


Serve dividere il contenuto in blocchi intelligenti.


==================================================
STEP CORRENTE

STEP 15

Sistema chunking.


==================================================
OBIETTIVO

Creare modulo che divide Document in Chunk.


==================================================
FILE DA CREARE

pipeline/chunker.py


==================================================
REQUISITI

La divisione deve rispettare:

- paragrafi;
- titoli;
- sezioni;
- limite token.


Non deve tagliare frasi a metà.


==================================================
FUNZIONE

create_chunks(document, max_tokens)


Output:

list[Chunk]


==================================================
REQUISITI TECNICI

Usare:

- tiktoken oppure tokenizer locale;
- core.models.


==================================================
OUTPUT

Codice completo.

Test incluso.


==================================================
FILE NECESSARI

core/models.py

pipeline/normalizer.py
```

---

## STEP 16 — Client Ollama locale

```text
Sei un software engineer esperto di AI locali.


Progetto:

ExamBookGenerator


==================================================
CONTESTO

Il sistema usa modelli AI locali tramite Ollama, sia per la generazione testuale (topic, outline, capitoli) sia, più avanti (STEP 21, image matcher), per la descrizione di immagini tramite un modello vision (es. `llava`). Per evitare di dover riaprire e modificare questo modulo più tardi, il supporto multimodale va previsto fin da subito in questo step.

Non devono essere usate API cloud.


==================================================
STEP CORRENTE

STEP 16

Client Ollama.


==================================================
FILE DA CREARE

Creare:

llm/ollama_client.py


==================================================
OBIETTIVO

Creare interfaccia Python per comunicare con Ollama, sia in modalità solo testo sia in modalità testo+immagine.


==================================================
FUNZIONI

Implementare:


generate(prompt)

chat(messages)

check_connection()

generate_with_image(prompt: str, image_path: str) -> str — [v2] invia un prompt testuale insieme a un'immagine (letta da `image_path`) a un modello vision Ollama (nome modello preso da `config.images.vision_model`, es. `llava`); usata a partire dallo STEP 21 (image matcher) per descrivere e valutare la pertinenza delle immagini estratte. Se il modello vision configurato non è disponibile, deve sollevare un'eccezione specifica e gestibile (es. `VisionModelUnavailableError`) invece di un errore generico, così che i moduli chiamanti (STEP 21) possano attivare un fallback senza bloccare la pipeline.


==================================================
REQUISITI

Gestire:

- Ollama spento;
- timeout;
- errori risposta;
- retry;
- modello vision non installato/non disponibile (solo per `generate_with_image`, non deve mai bloccare l'esecuzione delle funzioni solo testo) [v2].


Configurazione modello presa da config.yaml (inclusi `images.vision_model` per `generate_with_image`) [v2].


==================================================
OUTPUT

Fornire:

- codice;
- esempio;
- test.


==================================================
FILE NECESSARI

utils/config.py

utils/logger.py

config.yaml
```

---

## STEP 17 — Gestore prompt AI

```text
Sei un prompt engineer e software engineer Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO

Il progetto utilizza AI per:

- classificare argomenti;
- creare indice;
- generare capitoli.


I prompt devono essere centralizzati.


==================================================
STEP CORRENTE

STEP 17

Prompt manager.


==================================================
FILE DA CREARE

Creare:

llm/prompt_manager.py


==================================================
OBIETTIVO

Gestire template prompt.


==================================================
REQUISITI

Creare funzioni:


build_topic_prompt()

build_outline_prompt()

build_chapter_prompt(topic, chunks, depth_level: int)

build_image_caption_prompt(image_context: str) — [v2] chiede al modello vision di descrivere in inglese, in 1-2 frasi, cosa mostra un'immagine e perché potrebbe essere rilevante didatticamente

build_image_relevance_prompt(chapter_content: str, image_description: str) — [v2] chiede al modello se una data immagine è pertinente per un capitolo e dove inserirla (inizio/dopo una sezione specifica/non inserire)


I prompt devono essere modificabili senza cambiare codice.


Supportare:

- variabili;
- template;
- contesto documenti.


Requisiti [v2]:

- Ogni prompt che produce testo destinato al manuale finale (`build_outline_prompt`, `build_chapter_prompt`) deve contenere esplicitamente l'istruzione "Always write the output in English, regardless of the language of the source material", indipendentemente dalla lingua dei documenti originali.
- `build_chapter_prompt` deve tradurre `depth_level` (1-10) in un'istruzione testuale concreta per il modello, ad esempio tramite una mappa:
  - 1-2: "Produce a concise summary. Cover every topic and subtopic listed, but keep explanations brief (1-3 sentences each). No topic may be omitted."
  - 3-5: "Provide a clear, moderately detailed explanation of each topic, including short examples where useful."
  - 6-8: "Provide a thorough, well-structured explanation of each topic, including derivations, examples, and common exam pitfalls."
  - 9-10: "Provide an exhaustive, highly detailed explanation. Do not omit any nuance, edge case, formula derivation, or example present in the source material."
- Nessun prompt deve richiedere una lunghezza fissa in parole/pagine: la lunghezza deriva da quanti topic/subtopic e chunk vengono passati nel contesto, non da un target numerico imposto dal prompt.


==================================================
OUTPUT

Codice completo.

Esempio utilizzo.


==================================================
FILE NECESSARI

llm/ollama_client.py

config.yaml
```

---

## STEP 18 — Analisi argomenti automatica

```text
Sei un software engineer senior esperto in LLM.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese, indipendentemente dalla lingua dei documenti sorgente.
2. La lunghezza del manuale NON è fissa: dipende dal numero di topic/subtopic individuati e dalla quantità di chunk associati a ciascuno (`generation.length_mode: "topic_driven"`).
3. Un parametro `generation.depth_level` (intero 1-10, da config.yaml o CLI `--depth`) controlla il livello di dettaglio di ogni capitolo: 1 = riassunto minimo ma completo di tutti gli argomenti, 10 = massimo dettaglio.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali, salvate in `output/assets/images/`; `pipeline/image_matcher.py` (STEP 21) decide, tramite un modello vision Ollama (es. `llava`), se e quale immagine inserire in ciascun capitolo.


==================================================
STEP CORRENTE

STEP 18

Topic analyzer.


==================================================
OBIETTIVO

Usare AI per analizzare documenti e creare struttura argomenti.


==================================================
FILE DA CREARE

pipeline/topic_analyzer.py


==================================================
INPUT

Lista:

Chunk


==================================================
OUTPUT

topics.json


contenente:

- argomento;
- descrizione;
- fonti;
- ordine;
- subtopic_count: numero di sotto-argomenti/chunk rilevanti individuati per questo topic — usato a valle da chapter_generator per determinare la lunghezza del capitolo, che deve crescere proporzionalmente a questo numero invece di essere fissa [v2].


==================================================
REQUISITI

Usare:

ollama_client

prompt_manager


==================================================
OUTPUT

Codice completo.

Test.


==================================================
FILE NECESSARI

llm/ollama_client.py

llm/prompt_manager.py

core/models.py
```

---

## STEP 19 — Generazione indice manuale

```text
Sei un software engineer senior.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese, indipendentemente dalla lingua dei documenti sorgente.
2. La lunghezza del manuale NON è fissa: dipende dal numero di topic/subtopic individuati da `pipeline/topic_analyzer.py` (STEP 17) e dalla quantità di chunk associati a ciascuno (`generation.length_mode: "topic_driven"`). Questo modulo genera l'outline, quindi deve riflettere questa proporzionalità già nella struttura (più subtopic → più sezioni nell'outline per quel topic).
3. Un parametro `generation.depth_level` (intero 1-10) controlla il livello di dettaglio di ogni capitolo: 1 = riassunto minimo ma completo di tutti gli argomenti, 10 = massimo dettaglio.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali; `pipeline/image_matcher.py` (STEP 21) decide se e quale immagine inserire in ciascun capitolo.


==================================================
STEP CORRENTE

STEP 19

Generazione outline.


==================================================
OBIETTIVO

Creare automaticamente:

outline.md


contenente:

- capitoli;
- sottocapitoli;
- ordine logico.


==================================================
FILE

pipeline/outline_generator.py


==================================================
INPUT

topics.json


==================================================
OUTPUT

outline.md


==================================================
REQUISITI

Usare AI locale.

Validare Markdown.


==================================================
FILE NECESSARI

pipeline/topic_analyzer.py

llm/ollama_client.py
```

---

## STEP 20 — Sistema template Markdown

```text
Sei un software engineer Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese: il template e le variabili che espone devono essere pensati per contenuto in inglese.
2. La lunghezza del manuale NON è fissa: dipende dai topic/subtopic individuati, non da un target di parole del template.
3. Un parametro `generation.depth_level` (1-10) controlla il livello di dettaglio dei capitoli generati che il template dovrà accogliere.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali; il template deve prevedere un blocco opzionale per queste immagini, popolato solo quando `pipeline/image_matcher.py` (STEP 21) ne trova di pertinenti.


==================================================
STEP CORRENTE

STEP 20

Template engine.


==================================================
OBIETTIVO

Permettere all'utente di definire struttura manuale tramite:

template.md


==================================================
FILE

pipeline/template_engine.py


==================================================
FUNZIONI

load_template()

apply_template()


==================================================
REQUISITI

Supportare:

variabili (in inglese, dato che il manuale finale è sempre in inglese) [v2]:

{{title}}

{{content}}

{{sources}}

{{images}} — [v2] blocco opzionale, popolato solo se il capitolo ha immagini associate; ogni immagine viene resa come `![{{caption}}]({{path}})` seguita, se disponibile, dalla `ai_description` come didascalia estesa in corsivo.


==================================================
OUTPUT

Codice completo.


==================================================
FILE NECESSARI

template.md
```

---

## STEP 21 — Image matcher e inserimento nel manuale [v2]

```text
Sei un software engineer senior specializzato in sistemi RAG multimodali.

Progetto:

ExamBookGenerator


==================================================
CONTESTO
==================================================

Dopo l'estrazione (STEP 6, e la sua integrazione nei parser agli STEP 7-9), abbiamo una libreria di ExtractedImage per documento sorgente, ma non sappiamo ancora quali siano effettivamente utili a spiegare un concetto e in quale capitolo inserirle.

Serve un modulo che, per ogni capitolo generato, decida se una o più immagini tra quelle disponibili nei documenti sorgente di quel topic vadano effettivamente inserite, usando un modello Ollama multimodale (es. llava) per "guardare" l'immagine e valutarne la pertinenza.

Se nessun modello vision è disponibile, il sistema deve degradare in modo controllato (fallback euristico o nessuna immagine), senza bloccare la generazione del manuale.


==================================================
STEP CORRENTE
==================================================

STEP 21

Image matcher.


==================================================
OBIETTIVO
==================================================

Creare il modulo che seleziona e posiziona le immagini pertinenti in ciascun capitolo.


==================================================
FILE DA CREARE
==================================================

Creare:

pipeline/image_matcher.py


==================================================
FUNZIONI RICHIESTE
==================================================

describe_image(image: ExtractedImage) -> str

- invia l'immagine al modello vision configurato (`config.images.vision_model`) tramite `llm.ollama_client.generate_with_image()` (STEP 16), passando il prompt costruito da `build_image_caption_prompt()` (STEP 17);
- restituisce una breve descrizione in inglese, salvata anche in image.ai_description.

select_images_for_chapter(topic: Topic, chapter_draft: str, candidate_images: list[ExtractedImage], max_images: int = 3) -> list[tuple[ExtractedImage, str]]

- filtra le candidate_images ai soli documenti sorgente collegati al topic;
- per ciascuna candidata (fino a un limite ragionevole per non appesantire troppo l'inferenza), chiama describe_image() se non già descritta;
- usa build_image_relevance_prompt(chapter_draft, image.ai_description) per chiedere al modello se e dove inserirla;
- restituisce al massimo max_images coppie (immagine, posizione suggerita nel testo, es. "after introduction" o "after section: <titolo>").

fallback_heuristic_match(topic: Topic, candidate_images: list[ExtractedImage]) -> list[ExtractedImage]

- usata se il modello vision non è disponibile (check_connection() fallita o modello mancante);
- euristica semplice basata su corrispondenza tra parole chiave del topic/caption originale delle immagini e nessun uso del modello AI;
- se non c'è alcun segnale di pertinenza, non restituisce nulla (meglio nessuna immagine che un'immagine sbagliata).


==================================================
REQUISITI TECNICI
==================================================

Usare:

- llm/ollama_client.py, che dallo STEP 16 espone già `generate_with_image(prompt, image_path)` per l'invio di immagini oltre che di solo testo;
- llm/prompt_manager.py;
- core/models.py;
- logging.

Gestire:

- modello vision non installato/non risponde: log di warning, fallback automatico a fallback_heuristic_match, mai un errore bloccante;
- timeout su immagini singole (skip immagine, continua con le altre);
- nessuna immagine candidata per un topic (ritorna lista vuota, non è un errore).


==================================================
OUTPUT RICHIESTO
==================================================

Fornisci:

- spiegazione dell'approccio (in particolare come si integra con chapter_generator.py dello STEP 22);
- codice completo;
- esempio di utilizzo end-to-end (topic + candidate images → capitolo con immagini inserite);
- test (incluso un test del percorso di fallback senza modello vision disponibile).


==================================================
FILE DA ALLEGARE DAGLI STEP PRECEDENTI
==================================================

Necessari:

core/models.py

llm/ollama_client.py

llm/prompt_manager.py

pipeline/image_extractor.py

utils/logger.py

utils/config.py
```

---

# PARTE 4/4

## STEP 22 — Generatore capitoli

```text
Sei un software engineer senior specializzato in sistemi RAG.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese, indipendentemente dalla lingua dei documenti sorgente. Questo modulo genera i singoli capitoli, quindi è il punto in cui questa regola conta di più.
2. La lunghezza del manuale NON è fissa: ogni capitolo deve scalare con `topic.subtopic_count` e con il numero/dimensione dei chunk collegati, non con un target di parole fisso.
3. Un parametro `generation.depth_level` (intero 1-10) controlla il livello di dettaglio: 1 = riassunto minimo ma completo di tutti gli argomenti, 10 = massimo dettaglio.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali, salvate in `output/assets/images/`; `pipeline/image_matcher.py` (STEP 21) decide se e quale immagine inserire in ciascun capitolo.


==================================================
STEP CORRENTE

STEP 22

Generazione capitoli.


==================================================
OBIETTIVO

Creare modulo che genera ogni capitolo del manuale.


==================================================
FILE

pipeline/chapter_generator.py


==================================================
INPUT

- topic (incluso subtopic_count)
- chunk collegati
- template
- depth_level: int (1-10), da config.yaml o CLI [v2]
- candidate_images: list[ExtractedImage] pertinenti ai documenti sorgente del topic [v2]


==================================================
OUTPUT

chapter.md (sempre in lingua inglese, indipendentemente dalla lingua dei documenti sorgente) [v2]


==================================================
REQUISITI

Il capitolo deve avere:

- introduzione;
- spiegazione;
- approfondimenti;
- esempi;
- domande esame.

Requisiti aggiuntivi [v2]:

- Il capitolo è generato interamente in inglese tramite `build_chapter_prompt(topic, chunks, depth_level)`.
- La lunghezza del capitolo NON è fissata: deve scalare naturalmente con `topic.subtopic_count` e con il numero/dimensione dei chunk collegati. Un topic con pochi chunk produce un capitolo corto; un topic con molti chunk produce un capitolo lungo. Questo vale a qualunque `depth_level`: a parità di depth_level, più argomenti = manuale più lungo, non capitoli più lunghi artificialmente.
- Se `config.images.match_to_chapters` è true, il modulo chiama `image_matcher.select_images_for_chapter(topic, chapter_draft, candidate_images)` (STEP 21) per ottenere 0-3 immagini rilevanti, e le inserisce nel Markdown con sintassi `![caption](../assets/images/<file>)` nei punti del capitolo indicati dal matcher (es. subito dopo la sezione a cui si riferiscono). Se non ci sono immagini pertinenti, il capitolo viene generato normalmente senza immagini: non è un errore.


==================================================
FILE NECESSARI

llm/ollama_client.py

llm/prompt_manager.py

core/models.py

pipeline/image_matcher.py (STEP 21) [v2]
```

---

## STEP 23 — Validatore output

```text
Sei un software engineer senior.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale deve essere SEMPRE in lingua inglese: questo modulo deve poter segnalare capitoli che non rispettano questa regola.
2. La lunghezza del manuale NON è fissa: il validatore non deve mai penalizzare capitoli "troppo corti o troppo lunghi" in assoluto, solo verificare coerenza strutturale (sezioni mancanti, Markdown rotto, duplicazioni).
3. Un parametro `generation.depth_level` (1-10) controlla il livello di dettaglio; non è compito del validatore imporlo, solo verificare che il capitolo sia comunque completo.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali, salvate in `output/assets/images/`; questo modulo deve verificare che ogni riferimento immagine nel Markdown punti a un file realmente esistente.


==================================================
STEP CORRENTE

STEP 23

Validator.


==================================================
OBIETTIVO

Controllare qualità manuale generato.


==================================================
FILE

pipeline/validator.py


==================================================
CONTROLLI

Verificare:

- sezioni mancanti;
- Markdown errato;
- capitoli vuoti;
- duplicazioni;
- [v2] ogni riferimento immagine `![...](path)` nel Markdown punta a un file effettivamente esistente in `output/assets/images/` (nessun link rotto);
- [v2] euristica di rilevamento lingua (es. tramite `langdetect` o controllo di parole chiave comuni) per segnalare in `validation.json` eventuali capitoli che sembrano non essere in inglese, così da poterli rigenerare.


==================================================
OUTPUT

Report:

validation.json


==================================================
FILE NECESSARI

chapter_generator.py
```

---

## STEP 24 — Merge manuale finale

```text
Sei un software engineer senior.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese: il file unito deve chiamarsi esattamente `Exam_Manual.md`.
2. La lunghezza del manuale NON è fissa: il merge deve semplicemente concatenare tutti i capitoli generati, qualunque sia il loro numero o lunghezza, senza troncare nulla.
3. Un parametro `generation.depth_level` (1-10) ha già determinato il livello di dettaglio dei singoli capitoli a monte; questo step non deve modificarlo.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali, salvate in `output/assets/images/`; questo step deve assicurarsi che tutti i path relativi alle immagini restino validi dopo l'unione dei capitoli in un unico file.


==================================================
STEP CORRENTE

STEP 24

Assemblatore finale.


==================================================
OBIETTIVO

Unire tutti i capitoli.


==================================================
FILE

pipeline/merge.py


==================================================
INPUT

capitoli/*.md


==================================================
OUTPUT

output/Exam_Manual.md


==================================================
REQUISITI

Creare:

- indice;
- numerazione;
- metadata.

[v2] Copiare/consolidare `output/assets/images/` accanto al file finale, verificando che i path relativi usati nel Markdown (`../assets/images/...` o `assets/images/...` a seconda della posizione finale del file) restino validi dopo l'unione dei capitoli in un unico file.


==================================================
FILE NECESSARI

chapter_generator.py

pipeline/image_matcher.py [v2]
```

---

## STEP 25 — CLI completa

```text
Sei un software engineer senior.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese, indipendentemente dalla lingua dei documenti sorgente.
2. La lunghezza del manuale NON è fissa: dipende dai topic/subtopic individuati, non da un flag CLI.
3. Un parametro `generation.depth_level` (1-10) controlla il livello di dettaglio; la CLI deve esporne un override tramite `--depth`.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali; la CLI deve poter disattivare questa funzionalità con `--no-images`.


==================================================
STEP CORRENTE

STEP 25

Interfaccia terminale.


==================================================
OBIETTIVO

Creare comando:


python main.py --input cartella


==================================================
FUNZIONI

Supportare:

--input

--template

--model

--output

--depth (int, 1-10, default da config.yaml) — override runtime di `generation.depth_level` [v2]

--no-images — disattiva l'estrazione/inserimento immagini per questa esecuzione [v2]


==================================================
FILE

main.py


==================================================
FILE NECESSARI

tutti i moduli precedenti
```

---

## STEP 26 — GUI

```text
Sei un software engineer senior Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese, indipendentemente dalla lingua dei documenti sorgente.
2. La lunghezza del manuale NON è fissa: dipende dai topic/subtopic individuati, non da un'impostazione dell'utente.
3. Un parametro `generation.depth_level` (1-10) controlla il livello di dettaglio; la GUI deve esporlo come slider.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali; la GUI deve permettere di attivare/disattivare questa funzionalità.


==================================================
STEP CORRENTE

STEP 26

Interfaccia grafica.


==================================================
OBIETTIVO

Creare GUI desktop.


==================================================
TECNOLOGIA

Usare:

PySide6


==================================================
FUNZIONI

La GUI deve avere:

- selezione cartella;
- selezione template;
- scelta modello;
- barra progresso;
- log;
- [v2] slider "Depth level" da 1 a 10 (con etichette agli estremi: "1 = minimal summary" / "10 = maximum detail"), collegato a `generation.depth_level`;
- [v2] checkbox "Include images from source material" collegata a `images.match_to_chapters`.


==================================================
FILE

gui/app.py


==================================================
FILE NECESSARI

main.py

pipeline completa
```

---

## STEP 27 — Test automatici

```text
Sei un software engineer senior.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale deve essere SEMPRE in lingua inglese: aggiungere almeno un test che verifichi questo comportamento (es. su chapter_generator).
2. La lunghezza del manuale NON è fissa: i test non devono assumere un numero fisso di parole/pagine, ma verificare la proporzionalità tra numero di chunk/topic e lunghezza generata.
3. Un parametro `generation.depth_level` (1-10) deve essere testato ai valori limite (1 e 10) e con un valore fuori range.
4. Il sistema riutilizza (non genera mai) immagini reali; includere test per il percorso con e senza modello vision disponibile (fallback euristico).


==================================================
STEP CORRENTE

STEP 27

Testing.


==================================================
OBIETTIVO

Creare test automatici.


==================================================
FILE

tests/


==================================================
TESTARE

- parser;
- scanner;
- storage;
- chunking;
- generazione.


Usare:

pytest


==================================================
OUTPUT

Codice test completo.
```

---

## STEP 28 — Packaging finale

```text
Sei un software engineer senior DevOps Python.


Progetto:

ExamBookGenerator


==================================================
CONTESTO GENERALE DEL PROGETTO [v2]
==================================================

ExamBookGenerator è un'applicazione locale (Python 3.12+, Ollama, nessuna API cloud) che trasforma una cartella disordinata di materiale universitario in un unico manuale finale:

output/Exam_Manual.md

Regole valide per l'intero progetto, applicabili anche a questo step se pertinenti:

1. Il manuale finale è SEMPRE in lingua inglese, indipendentemente dalla lingua dei documenti sorgente — il README generato deve documentarlo.
2. La lunghezza del manuale NON è fissa: dipende dai topic/subtopic individuati nel materiale fornito dall'utente.
3. Un parametro `generation.depth_level` (1-10, in `config.yaml` di esempio) controlla il livello di dettaglio; il README deve spiegarlo al nuovo utente.
4. Il sistema riutilizza (non genera mai) immagini reali estratte dai documenti originali; il README deve menzionare i requisiti opzionali per la vision (es. `ollama pull llava`).


==================================================
STEP CORRENTE

STEP 28

Preparazione distribuzione.


==================================================
OBIETTIVO

Rendere il progetto installabile.


==================================================
CREARE

- README.md
- requirements.txt finale
- script installazione
- configurazione esempio


==================================================
REQUISITI

Un nuovo utente deve poter fare:

pip install -r requirements.txt

python main.py


==================================================
OUTPUT

Documentazione completa.
```

---

---

## Note finali

Con questi 28 step puoi aprire **28 chat diverse con 28 modelli diversi** e ogni modello saprà:

- che progetto sta costruendo;
- quale pezzo deve creare;
- quali file servono;
- quali interfacce rispettare.

Segui gli step nell'ordine numerico 1 → 28: la numerazione ora rispecchia le reali dipendenze tra i moduli (un modulo non richiede mai un file prodotto da uno step successivo), quindi arrivato allo STEP 28 il progetto è davvero completo.

L'unica cosa che devi fare durante lo sviluppo è conservare i file generati e allegare quelli indicati nella sezione **"FILE DA ALLEGARE DAGLI STEP PRECEDENTI"** quando un modulo ne ha bisogno.