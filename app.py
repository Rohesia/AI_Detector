import os
# Forza l'uso di PyTorch ed evita conflitti con TensorFlow (causa dell'errore DLL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
CORS(app)

# --- CONFIGURAZIONE PERCORSI ---
# Coerente con la tua struttura cartelle: Progetto/pth/best_bert_final.pth
MODEL_PATH = os.path.join("pth", "best_bert_final.pth")
MODEL_NAME = "bert-base-uncased" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNZIONE DI PULIZIA (Fondamentale per la coerenza) ---
def clean_text_simple(text):
    """
    Simula la pulizia fatta nel notebook (text_cleaned_lem).
    Il modello si aspetta testo senza caratteri speciali e minuscolo.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Rimuove punteggiatura
    text = re.sub(r'\d+', '', text)      # Rimuove numeri
    return text.strip()

# --- CARICAMENTO MODELLO ---
print(f"Caricamento modello su: {device}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

try:
    if os.path.exists(MODEL_PATH):
        # Carica i pesi salvati nel notebook
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Modello BERT caricato con successo!")
    else:
        print(f"❌ ERRORE: File {MODEL_PATH} non trovato!")
except Exception as e:
    print(f"❌ Errore durante il caricamento: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_text = data.get('text', '')

    if not raw_text or len(raw_text) < 10:
        return jsonify({"error": "Testo troppo breve"}), 400

    # 1. Pulizia del testo per coerenza con 'text_cleaned_lem'
    cleaned_text = clean_text_simple(raw_text)

    # 2. Tokenizzazione (Usa lo stesso MAX_LENGTH del notebook, solitamente 512 o meno)
    inputs = tokenizer(
        cleaned_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Estrazione probabilità
        prob_human = probs[0][0].item()
        prob_ai = probs[0][1].item()
        
        # Debug nel terminale per vedere i valori reali
        print(f"DEBUG -> Raw Score - Human: {prob_human:.4f}, AI: {prob_ai:.4f}")
        
        prediction = torch.argmax(probs, dim=1).item()
        confidence = max(prob_human, prob_ai)

    # 3. Mappatura Label coerente con: human=0, ai=1
    label = "AI Generated" if prediction == 1 else "Human Written"
    
    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "human": round(prob_human * 100, 2),
            "ai": round(prob_ai * 100, 2)
        }
    })

if __name__ == '__main__':
    print("Server avviato su http://localhost:5000")
    app.run(port=5000, debug=False) # Debug False per evitare doppi caricamenti pesanti di BERT