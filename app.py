from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
# CORS permette al frontend (React) di comunicare con questo server Python
CORS(app)

# --- CONFIGURAZIONE MODELLO ---
# Assicurati che il percorso del file .pth sia corretto rispetto a dove si trova app.py
MODEL_PATH = "pth/best_bert_final.pth" 
# Usa lo stesso modello base che hai usato durante il training
MODEL_NAME = "bert-base-uncased" 

# Imposta il dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Caricamento modello su: {device}...")

# Carichiamo il tokenizer e la struttura del modello
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Carichiamo i pesi dal tuo file .pth
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Imposta il modello in modalità valutazione
    print("Modello caricato con successo!")
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")

# --- ROTTA PER LA PREDIZIONE ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Nessun testo fornito"}), 400

    # Tokenizzazione del testo ricevuto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Otteniamo la classe con la probabilità più alta
        prediction = torch.argmax(outputs.logits, dim=1).item()
        # Calcoliamo la percentuale di confidenza
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probs).item()

    # Definiamo i label (controlla se nel tuo training 1 è AI e 0 è Human)
    label = "AI Generated" if prediction == 1 else "Human Written"
    
    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == '__main__':
    # Il server girerà su http://localhost:5000
    app.run(port=5000, debug=True)