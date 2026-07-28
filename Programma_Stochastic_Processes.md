# Stochastic Processes for Data Science 2025-26
## Programma del corso

---

### 1. Introduzione e limiti della previsione
- Panoramica generale del modulo e dei temi trattati
- Introduzione euristica ai limiti della previsione ("Predicting the future from the past")

### 2. Processi scambiabili (exchangeable processes)
- Definizione di processo stocastico
- Tre esempi di processi scambiabili: distribuzione binomiale, ipergeometrica e di Polya
- Probabilità predittive e loro relazione con le distribuzioni finito-dimensionali
- Derivazione delle probabilità predittive per il processo di Polya
- Derivazione delle distribuzioni finito-dimensionali per il processo di Polya
- Simulazione della passeggiata aleatoria di Polya
- Teorema di rappresentazione di de Finetti (versione finita)
- Teorema di rappresentazione di de Finetti (versione infinita)
- Teorema di rappresentazione di Johnson
- Cenni sulle reti Bayesiane

### 3. Catene di Markov a tempo discreto
- Definizione di catena di Markov e catena di Markov omogenea nel tempo
- Matrice di transizione e vettore delle probabilità iniziali
- Derivazione delle distribuzioni finito-dimensionali da matrice di transizione e vettore iniziale
- Equazioni di Chapman-Kolmogorov (condizioni necessarie per le catene di Markov)
- Equazione master e relazione con le equazioni differenziali (caso tempo continuo)
- Distribuzione invariante: bilancio globale (global balance) e bilancio dettagliato (detailed balance)
- Esempio: distribuzione invariante per catena a due stati con transizioni simmetriche
- Espansione agli autovalori delle probabilità di transizione in n passi (caso simmetrico a due stati)
- Modello dell'urna di Ehrenfest
- Potenze della matrice di transizione: caso periodico e aperiodico
- Simulazione Monte Carlo del processo di Ehrenfest
- Classificazione degli stati delle catene di Markov (con esempi)

### 4. Tempi di arresto e proprietà di Markov
- Tempi di colpimento (hitting times), probabilità di colpimento e tempo atteso di colpimento per un sottoinsieme dello spazio degli stati
- Sistemi di equazioni lineari per probabilità e tempi di colpimento; soluzione minimale
- Concetto di tempo di arresto (stopping time), con esempi
- Proprietà forte di Markov
- Stati ricorrenti e transienti

### 5. Ricorrenza, transienza e distribuzioni invarianti
- Teoremi su ricorrenza e transienza, criteri per stabilirle
- Ricorrenza delle passeggiate aleatorie simmetriche in 1D, transienza di quelle polarizzate
- Esistenza di una misura invariante per catene irriducibili e ricorrenti
- Unicità della misura invariante
- Classificazione degli stati ricorrenti: ricorrenti positivi e ricorrenti nulli
- Legge forte dei grandi numeri per catene di Markov
- Catene aperiodiche
- Teorema limite per catene irriducibili, aperiodiche e positive ricorrenti (dimostrazione completa)

### 6. Modello di Ising e MCMC
- Modello di Ising e distribuzione di Gibbs (introduzione in vista degli algoritmi MCMC)
- Esempio di tre spin su grafo triangolare completamente connesso
- Calcolo della funzione di partizione "by brute force"
- Grandezze termodinamiche: energia interna, energia libera, entropia
- Seminario (Lorenzo Facciaroni): metodi gruppo-teoretici per i tempi di mixing delle catene di Markov — esempio della passeggiata aleatoria sull'ipercubo

### 7. Passeggiate aleatorie a tempo continuo (CTRW)
- Definizione di continuous time random walk
- Derivazione dell'equazione di Montroll-Weiss
- Processi di conteggio di tipo renewal

### 8. Vettori e processi Gaussiani
- Variabili aleatorie Gaussiane (normali) e vettori Gaussiani
- Funzione caratteristica dei vettori Gaussiani
- Dimostrazione: variabili Gaussiane incorrelate sono indipendenti
- Densità di probabilità congiunta per un vettore Gaussiano (con esempio a due componenti)
- Definizione di processo Gaussiano e teorema di caratterizzazione
- Teorema di esistenza per i processi Gaussiani
- Proprietà del cammino: teorema di Kolmogorov-Chentsov
- Proprietà di Markov: teorema di Doob
- Equivalenza tra stazionarietà debole e forte per processi Gaussiani
- Costruzione dello spazio di Hilbert a nucleo riproducente (RKHS)
- Espansione di Karhunen-Loève per un processo Gaussiano centrato

### 9. Moto Browniano
- Definizione del moto Browniano come processo Gaussiano
- Dimostrazione delle proprietà di Markov e di martingala
