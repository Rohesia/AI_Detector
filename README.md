# 🧠 AI Detector – Hybrid BERT & Stylometric Analysis

## 📌 Overview
Questo progetto implementa un **sistema end-to-end per il rilevamento di testi generati da Intelligenza Artificiale**, combinando modelli NLP moderni e tecniche di analisi stilometrica.  
L’obiettivo è distinguere testi **scritti da esseri umani** da testi **generati da modelli AI**, adottando un approccio sia **predittivo** sia **interpretabile**.

Il sistema è stato progettato come applicazione **full-stack containerizzata**, includendo backend AI, frontend web e orchestrazione tramite Docker.

---

## 🏗️ Architettura del Sistema

┌────────────┐ REST API ┌──────────────┐
│ React UI │ ───────────────▶ │ FastAPI ML │
│ Frontend │ │ Backend │
└────────────┘ └──────┬───────┘
│
┌─────────▼─────────┐
│ Modelli ML │
│ BERT (.pth) │
│ Signature (.pkl) │
└───────────────────┘


- **Frontend**: React + TailwindCSS  
- **Backend**: Python + FastAPI  
- **Modelli**: BERT + feature stilometriche  
- **Deployment**: Docker & Docker Compose  

---

## 📁 Struttura del Progetto

bert_ai_detector/
│
├── app.py # Backend FastAPI
├── requirements.txt # Dipendenze Python
├── Dockerfile # Backend container
│
├── react-app/ # Frontend React
│ ├── Dockerfile
│ ├── src/
│ │ └── components/
│ │ └── AIDetectorInterface.jsx
│ └── ...
│
├── pth/ # Modelli deep learning
│ └── best_bert.pth
│
├── pkl/ # Modelli ML / signature
├── txt/ # File di supporto
│
├── *.ipynb # Notebook (EDA, training, analisi)
├── *.csv # Dataset
│
├── docker-compose.yml
└── README.md


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

## ⚙️ Backend – FastAPI

Il backend espone un’API REST per l’analisi dei testi.

### Endpoint
`POST /analyze`

### Input
```json
{
  "text": "Testo da analizzare"
}
