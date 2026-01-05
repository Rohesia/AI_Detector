<p align="center">
  <img src="assets/architecture.avif" alt="Human vs AI Concept" width="800">
</p>



# 🧠 AI Detector – Hybrid BERT & Stylometric Analysis

## 📌 Overview
Questo progetto implementa un **sistema end-to-end per il rilevamento di testi generati da Intelligenza Artificiale**, combinando modelli NLP moderni e tecniche di analisi stilometrica.  
L'obiettivo è distinguere testi **scritti da esseri umani** da testi **generati da modelli AI**, adottando un approccio sia **predittivo** sia **interpretabile**.

Il sistema è stato progettato come applicazione **full-stack containerizzata**, includendo backend AI, frontend web e orchestrazione tramite Docker.

---

<p align="center">
  <img src="assets/ai_ex.gif" width="380"/>
  <img src="assets/hum.gif" width="380"/>
</p>


## 🏗️ Architettura del Sistema

```
┌────────────┐  REST API  ┌──────────────┐
│  React UI  │ ─────────► │  FastAPI ML  │
│  Frontend  │            │   Backend    │
└────────────┘            └──────┬───────┘
                                 │
                         ┌───────▼────────┐
                         │   Modelli ML   │
                         │  BERT (.pth)   │
                         │ Signature(.pkl)│
                         └────────────────┘
```

- **Frontend**: React + TailwindCSS  
- **Backend**: Python + FastAPI  
- **Modelli**: BERT + feature stilometriche  
- **Deployment**: Docker & Docker Compose  

---

## 📁 Struttura del Progetto

```
bert_ai_detector/
│
├── app.py                  # Backend FastAPI
├── requirements.txt        # Dipendenze Python
├── Dockerfile              # Backend container
│
├── react-app/              # Frontend React
│   ├── Dockerfile
│   ├── src/
│   │   └── components/
│   │       └── AIDetectorInterface.jsx
│   └── ...
│
├── pth/                    # Modelli deep learning
│   └── best_bert.pth
│
├── pkl/                    # Modelli ML / signature
├── txt/                    # File di supporto
│
├── *.ipynb                 # Notebook (EDA, training, analisi)
├── *.csv                   # Dataset
│
├── docker-compose.yml
└── README.md
```

I notebook Jupyter sono volutamente esclusi dai container per separare la fase di **training** da quella di **inference**.

---

## 🔬 Metodologia

### 1️⃣ Exploratory Data Analysis (EDA)
- Analisi statistica dei testi AI vs Human
- Studio di lunghezza, variabilità e struttura
- Supporto alle decisioni di feature engineering

### 2️⃣ Signature Stilometriche
- Diversità lessicale
- Ripetitività strutturale
- Burstiness
- Lunghezza media delle frasi

Queste feature forniscono un livello di **interpretabilità** complementare ai modelli deep learning.

### 3️⃣ Modello Ibrido (BERT + Feature Stilometriche)
- Embedding contestuali ottenuti tramite BERT
- Integrazione con feature linguistiche manuali
- Migliore robustezza e generalizzazione rispetto ad approcci singoli

---

## 📊 Risultati

| Architettura              | Accuracy (Test) | Punti di forza |
|--------------------------|----------------|---------------|
| **Hybrid (BERT + Style)** | **97.83%**     | Unisce contesto semantico e impronta stilistica |
| LSTM (Recurrent)         | 97.03%         | Cattura dipendenze sequenziali |
| BERT Fine-tuned          | 96.75%         | Comprensione semantica profonda |
| CNN (Convolutional)      | 92.50%         | Ottimo nel rilevare pattern locali (n-gram) |
| Baseline (Style Only)    | 86.72%         | Interpretazione delle abitudini di scrittura |


## ⚙️ Backend – FastAPI

Il backend espone un'API REST per l'analisi dei testi.

### Endpoint
`POST /analyze`

### Input
```json
{
  "text": "Testo da analizzare"
}
```

### Output
```json
{
  "isAI": true,
  "confidence": 87.3,
  "metrics": {
    "lexical_diversity": 0.42,
    "burstiness": 3.1,
    "avg_sentence_length": 18.7
  }
}
```

---

## 🎨 Frontend – React App

Il frontend fornisce un'interfaccia web interattiva per:

- Inserimento del testo
- Validazione dell'input
- Visualizzazione dei risultati e delle metriche
- Comunicazione diretta con il backend tramite REST API

---

## 🐳 Containerizzazione con Docker

Il sistema è completamente containerizzato tramite Docker Compose, che orchestra:

- Backend AI (FastAPI + modelli ML)
- Frontend React

### Avvio del progetto
```bash
docker-compose build
docker-compose up
```

| Servizio | | Descrizione     |
|----------| |-----------------|
| Backend  | | API AI Detector |
| Frontend | | Interfaccia Web |

La comunicazione tra frontend e backend avviene tramite service name Docker, garantendo portabilità e riproducibilità.

---

## 🎓 Scelte Progettuali

- **Separazione tra training e inference**
- **Approccio ibrido** per bilanciare performance e interpretabilità
- **Containerizzazione** per:
  - Riproducibilità degli esperimenti
  - Isolamento dell'ambiente
  - Semplicità di deploy
- **Interfaccia grafica** come strumento di analisi e non solo demo

---
## 🗄️ Persistenza dei Dati con SQLite3

<p align="center">
  <img src="assets/tab_pred.png" width="800">
  <img src="assets/tab_seq.png" width="400">
</p>



Per completare il sistema di AI Detection, non ci siamo limitati alla sola predizione in tempo reale, ma abbiamo introdotto un livello di **persistenza dei dati**, fondamentale per garantire tracciabilità, analisi e validazione dei risultati.

A questo scopo è stato utilizzato **SQLite3**, un database relazionale embedded, leggero e privo di dipendenze esterne.

---

### 🎯 Perché SQLite3

SQLite3 è particolarmente adatto a questo tipo di progetto per diversi motivi:

- Non richiede un server dedicato
- È immediatamente integrabile in applicazioni Python
- Funziona tramite un singolo file `.db`
- È ideale per ambienti containerizzati
- Garantisce semplicità, portabilità e affidabilità

Essendo il progetto di natura accademica e orientato all’analisi, SQLite rappresenta una scelta progettuale equilibrata tra semplicità ed efficacia.

---



### 🧠 Ruolo del Database nel Sistema

Il database non influisce sul processo decisionale del modello, ma svolge un ruolo chiave nel **monitoraggio delle predizioni**.

In particolare, consente di:

- Salvare ogni predizione effettuata dal sistema
- Conservare informazioni su confidenza e modello utilizzato
- Analizzare il comportamento del detector nel tempo
- Supportare future estensioni (dashboard, statistiche, auditing)

Ogni chiamata all’endpoint `/predict` genera automaticamente una nuova entry nel database.

---

### 🧱 Struttura del Database

Il database contiene una singola tabella chiamata `predictions`, progettata per essere semplice ma estendibile.

I campi principali includono:
- Un identificatore univoco
- Timestamp della predizione
- Etichetta finale (AI o Human)
- Confidenza associata
- Metriche stilometriche
- Versione del modello utilizzato

Questa struttura permette di mantenere uno storico completo delle analisi.

---

### ⚙️ Inizializzazione Automatica

Il database viene inizializzato automaticamente all’avvio dell’applicazione backend.

Se la tabella esiste già, il sistema lo rileva e non interviene.
Se la tabella non esiste, viene creata automaticamente.

Questo approccio garantisce:
- Robustezza
- Assenza di configurazioni manuali
- Compatibilità con Docker e deploy automatico

---

### 🔁 Integrazione con il Backend

Dal punto di vista architetturale, la gestione del database è isolata nel file `db.py`.

Il backend **FastAPI**:
1. Riceve il testo dall’utente
2. Esegue la predizione tramite il modello AI
3. Restituisce il risultato al frontend
4. Salva automaticamente i dati nel database

Il salvataggio è completamente trasparente per l’utente finale.

---

### 🔍 Ispezione e Debug

Il file del database può essere aperto tramite strumenti grafici come **DB Browser for SQLite**, consentendo:

- Verifica immediata dei record
- Controllo della correttezza delle predizioni
- Analisi manuale dei risultati

Questo è particolarmente utile in fase di testing e validazione del sistema.

---

### 🧩 Estensioni Future

L’introduzione di SQLite apre la strada a possibili sviluppi futuri, tra cui:

- Analisi statistiche delle predizioni
- Dashboard di monitoraggio
- Migrazione verso database più complessi (PostgreSQL)
- Logging avanzato e audit trail

In questo modo, il sistema non è solo un detector, ma una piattaforma analizzabile e tracciabile.

---



L’integrazione di SQLite3 completa il progetto dal punto di vista ingegneristico, trasformando il modello AI in un sistema reale e persistente.

La scelta di un database embedded riflette una progettazione consapevole, orientata alla semplicità, alla riproducibilità e alla qualità del software.



## ⚠️ Limiti e Sviluppi Futuri

- Generalizzazione rispetto a modelli AI futuri
- Integrazione del supporto GPU (CUDA)
- Valutazione cross-domain
- Logging e monitoring delle predizioni
- Supporto per analisi batch

---

## 👤 Autore

Progetto sviluppato come lavoro accademico nell'ambito di Machine Learning e Natural Language Processing.

---

## 🏁 Conclusione

Il progetto dimostra come sia possibile costruire un AI Detector moderno e completo, combinando:

- Analisi statistica
- Modelli deep learning
- Interpretabilità linguistica
- Ingegneria del software

Il risultato è un sistema modulare, riproducibile e pronto al deploy.