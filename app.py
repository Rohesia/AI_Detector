import os
# Forza l'uso di PyTorch ed evita conflitti con TensorFlow (causa dell'errore DLL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

"""
AI Text Detector - Flask Backend API (Enhanced Edition)
Handles predictions from React frontend with disguised AI detection
"""
import os
# Forza l'uso di PyTorch ed evita conflitti con TensorFlow (causa dell'errore DLL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

"""
AI Text Detector - Flask Backend API (ULTRA-Enhanced Edition)
Handles predictions with aggressive disguised AI detection
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import spacy
from scipy.stats import entropy
from collections import Counter
from itertools import tee

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
MIN_TEXT_LENGTH = 50

print("\n" + "="*70)
print(" 🤖 AI DETECTOR BACKEND - ULTRA-ENHANCED - STARTING...")
print("="*70 + "\n")

# ============================================================================
# LOAD MODELS & RESOURCES
# ============================================================================

print("📦 Loading models...")

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print("✅ spaCy loaded")
except:
    print("⚠️  Installing spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

# Load BERT
try:
    tokenizer = BertTokenizer.from_pretrained('./bert_ai_detector')
    model_bert = BertForSequenceClassification.from_pretrained('./bert_ai_detector').to(DEVICE)
    model_bert.eval()
    print("✅ BERT model loaded")
    BERT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  BERT not found: {e}")
    model_bert = None
    tokenizer = None
    BERT_AVAILABLE = False

# Load Hybrid model
try:
    rf_hybrid = joblib.load('./pkl/rf_hybrid_model.pkl')
    scaler = joblib.load('./pkl/feature_scaler.pkl')
    print("✅ Hybrid model loaded")
    HYBRID_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Hybrid model not found: {e}")
    rf_hybrid = None
    scaler = None
    HYBRID_AVAILABLE = False

# Load feature list
try:
    with open('./txt/stylometric_features.txt', 'r') as f:
        STYLOMETRIC_FEATURES = [line.strip() for line in f.readlines()]
    print(f"✅ {len(STYLOMETRIC_FEATURES)} features loaded")
except:
    print("⚠️  Using default features")
    STYLOMETRIC_FEATURES = [
        'sentence_length_cv', 'burstiness_index', 'pos_bigram_entropy',
        'dependency_depth_mean', 'lexical_compression_ratio',
        'function_word_ratio', 'sentence_similarity_drift',
        'structural_redundancy', 'sentiment_variance',
        'readability_oscillation', 'clause_density',
        'hapax_density', 'template_bias_score'
    ]

# Load config
try:
    with open('./pkl/config.json', 'r') as f:
        config = json.load(f)
        MAX_LENGTH = config.get('max_length', 512)
except:
    pass

print(f"🖥️  Device: {DEVICE}")
print(f"📏 Max length: {MAX_LENGTH}")
print("🔍 ULTRA-AGGRESSIVE disguise detection: ACTIVE")
print("\n" + "="*70 + "\n")

# ============================================================================
# STYLOMETRIC FEATURE EXTRACTION
# ============================================================================

def pairwise(iterable):
    """Generate consecutive pairs"""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def safe_entropy(counter: Counter) -> float:
    """Calculate entropy safely"""
    values = np.array(list(counter.values()), dtype=float)
    if values.sum() == 0:
        return 0.0
    probs = values / values.sum()
    return entropy(probs)

def coefficient_of_variation(x):
    """CV = std/mean"""
    mu = np.mean(x)
    return np.std(x) / mu if mu > 0 else 0.0

def burstiness_index(x):
    """Burstiness = (σ - μ) / (σ + μ)"""
    mu, sigma = np.mean(x), np.std(x)
    return (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0.0

def extract_stylometric_signature(text):
    """Extract all 13 stylometric features"""
    try:
        text = str(text)[:10000]
        doc = nlp(text)
        
        features = {}
        
        # === R - Rhythmic Control ===
        sent_lengths = np.array([len(sent) for sent in doc.sents if len(sent) > 0])
        
        if len(sent_lengths) > 0:
            features['sentence_length_cv'] = coefficient_of_variation(sent_lengths)
            features['burstiness_index'] = burstiness_index(sent_lengths)
        else:
            features['sentence_length_cv'] = 0.0
            features['burstiness_index'] = 0.0
        
        # === S - Syntactic Entropy ===
        pos_tags = [token.pos_ for token in doc]
        
        if len(pos_tags) >= 2:
            bigrams = list(pairwise(pos_tags))
            counts = Counter(bigrams)
            features['pos_bigram_entropy'] = safe_entropy(counts)
        else:
            features['pos_bigram_entropy'] = 0.0
        
        # Dependency depth
        depths = []
        for sent in doc.sents:
            for token in sent:
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                depths.append(depth)
        features['dependency_depth_mean'] = np.mean(depths) if depths else 0.0
        
        # === L - Lexical Efficiency ===
        tokens = [t for t in doc if t.is_alpha]
        
        if tokens:
            lemmas = [t.lemma_ for t in tokens]
            features['lexical_compression_ratio'] = len(set(lemmas)) / len(tokens)
            
            function_words = [t for t in tokens if t.pos_ in {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ"}]
            features['function_word_ratio'] = len(function_words) / len(tokens)
        else:
            features['lexical_compression_ratio'] = 0.0
            features['function_word_ratio'] = 0.0
        
        # === D - Discourse Regularization ===
        if len(list(doc.sents)) >= 2:
            vectors = np.array([sent.vector for sent in doc.sents])
            
            sims = []
            for i in range(len(vectors) - 1):
                v1, v2 = vectors[i], vectors[i+1]
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(v1, v2) / (norm1 * norm2)
                    sims.append(sim)
            
            features['sentence_similarity_drift'] = float(np.mean(sims)) if sims else 0.0
        else:
            features['sentence_similarity_drift'] = 0.0
        
        # Structural redundancy
        patterns = []
        for sent in doc.sents:
            pattern = tuple(tok.dep_ for tok in sent)
            patterns.append(pattern)
        
        if patterns:
            counts = Counter(patterns)
            repeated = sum(c for c in counts.values() if c > 1)
            features['structural_redundancy'] = repeated / len(patterns)
        else:
            features['structural_redundancy'] = 0.0
        
        # === E - Emotional Variance ===
        features['sentiment_variance'] = 0.15
        
        # === C - Cognitive Load ===
        features['readability_oscillation'] = 0.5
        
        sub_clauses = sum(1 for tok in doc if tok.dep_ in {"advcl", "ccomp", "xcomp", "relcl"})
        sentences_count = len(list(doc.sents))
        features['clause_density'] = sub_clauses / sentences_count if sentences_count > 0 else 0.0
        
        # === Additional Features ===
        words = [t.text.lower() for t in doc if t.is_alpha]
        if words:
            word_counts = Counter(words)
            hapax_count = sum(1 for w in word_counts if word_counts[w] == 1)
            features['hapax_density'] = hapax_count / len(words)
        else:
            features['hapax_density'] = 0.0
        
        # Template bias
        score = 0.0
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in ['in conclusion', 'overall', 'to summarize']):
            score += 1.2
        
        connectors = ['furthermore', 'moreover', 'additionally', 'consequently']
        connector_count = sum(1 for c in connectors if c in text_lower)
        if connector_count >= 2:
            score += 1.0
        
        features['template_bias_score'] = score
        
        return features
    
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return {feat: 0.0 for feat in STYLOMETRIC_FEATURES}

# ============================================================================
# ULTRA-AGGRESSIVE DISGUISED AI DETECTION
# ============================================================================

def calculate_artificial_informality(text):
    """Rileva quando l'AI finge di essere informale"""
    score = 0
    
    # Tutto minuscolo
    if text.islower():
        score += 3
        
        # MA struttura grammaticalmente perfetta
        try:
            doc = nlp(text)
            sents = list(doc.sents)
            if len(sents) > 5:
                # Conta errori grammaticali reali
                errors = 0
                for sent in sents:
                    # Un vero testo informale ha ripetizioni, frasi incomplete
                    if len(sent) > 20:  # Frasi lunghe in testo informale = sospetto
                        errors += 1
                
                if errors < 2:  # Troppo perfetto
                    score += 4
        except:
            pass
    
    # Nessuna punteggiatura
    if text.count('.') < 2 and text.count(',') < 2:
        score += 3
        
        # MA il testo è lungo e strutturato
        if len(text) > 200:
            score += 3
    
    return min(score, 10)

def detect_generic_motivation(text):
    """Rileva linguaggio motivazionale generico - ULTRA AGGRESSIVO"""
    
    # Database completo di frasi motivazionali AI
    motivation_phrases = [
        'dont worry', 'keep going', 'small steps', 'be patient',
        'stay positive', 'you got this', 'believe in yourself',
        'take it easy', 'one day at a time', 'progress not perfection',
        'be consistent', 'reach your goals', 'you will get better',
        'dont give up', 'stay focused', 'trust the process',
        'mistakes are normal', 'ask questions', 'practice when you can',
        'be patient with yourself', 'with time and effort', 'just be',
        'still matter', 'help you grow', 'you dont need', 'just keep',
        'do your best', 'every day', 'dont need to be perfect',
        'will get better', 'reach your goals', 'help you'
    ]
    
    motivation_words = [
        'consistent', 'patient', 'practice', 'effort', 'grow', 
        'better', 'goals', 'improve', 'achieve', 'progress',
        'worry', 'perfect', 'questions', 'matter', 'normal'
    ]
    
    text_lower = text.lower()
    
    # Conta frasi (peso massimo)
    phrase_matches = sum(1 for phrase in motivation_phrases if phrase in text_lower)
    
    # Conta parole
    word_matches = sum(1 for word in motivation_words if f' {word} ' in f' {text_lower} ' or text_lower.startswith(word + ' ') or text_lower.endswith(' ' + word))
    
    # Score con penalità esponenziale per molti match
    base_score = (phrase_matches * 4) + (word_matches * 0.8)
    
    # BONUS: Se ci sono 5+ frasi motivazionali in un testo corto = SICURO AI
    if phrase_matches >= 5 and len(text) < 500:
        base_score *= 1.5
    
    
    return min(base_score, 20)

def calculate_lexical_genericity(text):
    """Misura vocabolario generico - ULTRA AGGRESSIVO"""
    
    generic_words = {
        'important', 'help', 'good', 'better', 'great', 'nice',
        'thing', 'things', 'matter', 'reach', 'achieve', 'improve',
        'time', 'effort', 'practice', 'grow', 'learn', 'try', 'goals',
        'just', 'still', 'every', 'day', 'need', 'get', 'make',
        'can', 'will', 'dont', 'best', 'much', 'too', 'perfect',
        'yourself', 'questions', 'patient', 'worry', 'normal',
        'going', 'steps', 'small', 'you', 'be', 'and', 'the'
    }
    
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return 0
    
    generic_count = sum(1 for w in words if w in generic_words)
    ratio = generic_count / len(words)
    
    # Penalità ESPONENZIALE per alto rapporto
    if ratio > 0.6:  # 60%+ parole generiche
        return min(ratio * 25, 20)
    elif ratio > 0.5:
        return min(ratio * 18, 15)
    else:
        return ratio * 10
    


def detect_repetitive_structure(text):
    """Rileva struttura artificialmente ripetitiva"""
    
    # Separa in "pseudo-frasi" anche senza punteggiatura
    words = text.split()
    if len(words) < 10:
        return 0
    
    # Analizza pattern di lunghezza
    chunk_size = 5
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    if len(chunks) < 3:
        return 0
    
    chunk_lengths = [len(c.split()) for c in chunks]
    
    # Se le lunghezze sono troppo uniformi = AI
    std_dev = np.std(chunk_lengths)
    if std_dev < 1.5:  # Molto uniforme
        return 5
    elif std_dev < 2.5:
        return 3
    
    return 0

def detect_perfect_grammar_without_punctuation(text):
    # Se non c'è punteggiatura ma il testo è lungo
    if text.count('.') < 1 and len(text.split()) > 20:
        doc = nlp(text)
        sents = list(doc.sents)
        
        # Se spaCy identifica chiaramente più di 3 frasi logiche 
        # in un testo che non ha punti, è un segnale enorme di IA.
        if len(sents) >= 3:
            # Verifichiamo la perfezione: ogni "pseudo-frase" ha Soggetto + Verbo?
            perfect_units = 0
            for sent in sents:
                has_subj = any(t.dep_ in ('nsubj', 'nsubjpass') for t in sent)
                has_verb = any(t.pos_ == 'VERB' for t in sent)
                if has_subj and has_verb:
                    perfect_units += 1
            
            if perfect_units >= 3:
                return 10 # Punteggio massimo di sospetto
    return 0

def extract_enhanced_features(text):
    """Estrae features base + features anti-disguise ULTRA"""
    
    features = extract_stylometric_signature(text)
    
    # === CALCOLO DISGUISE SCORE AGGIORNATO ===
    disguise_score = (
        features.get('artificial_informality', 0) * 0.25 + # +5%
        features.get('generic_motivation_score', 0) * 0.35 + # +5%
        features.get('perfect_grammar_no_punct', 0) * 0.30 + # +10%
        features.get('lexical_genericity', 0) * 0.10
    )
    
    # TRIGGER AGGRESSIVO: Se mancano punti E ci sono frasi motivazionali
    if text.count('.') == 0 and features.get('generic_motivation_score', 0) > 10:
        disguise_score += 3.0 # Boost immediato per smascheramento
            
    # Punctuation ratio
    punct_count = text.count('.') + text.count(',') + text.count('!') + text.count('?')
    features['punctuation_ratio'] = punct_count / max(len(text), 1)
    
    # Capitalization variance
    words = text.split()
    if words:
        features['capitalization_variance'] = sum(1 for w in words if w and w[0].isupper()) / len(words)
    else:
        features['capitalization_variance'] = 0
    
    # Lexical genericity
    features['lexical_genericity'] = calculate_lexical_genericity(text)
    
    return features

def preprocess_and_validate(text):
    """Valida il testo e rileva anomalie"""
    
    char_count = len(text)
    word_count = len(text.split())
    
    anomalies = {
        'too_short': char_count < 200,
        'no_punctuation': (text.count('.') + text.count('!') + text.count('?')) < 2,
        'all_lowercase': text.islower(),
        'no_capitals': text[0].islower() if text else True,
        'high_repetition': len(set(text.split())) / max(len(text.split()), 1) < 0.5
    }
    
    reliability = 100 - (sum(anomalies.values()) * 15)
    reliability = max(reliability, 30)
    
    return text, reliability, anomalies

# ============================================================================
# PREDICTION FUNCTION (ULTRA-ENHANCED)
# ============================================================================

def predict_text_backend(text):
    """
    Main prediction logic con rilevamento ULTRA-AGGRESSIVO
    """
    
    processed_text, reliability, anomalies = preprocess_and_validate(text)
    features = extract_enhanced_features(processed_text)
    
    # === CALCOLO DISGUISE SCORE ULTRA-AGGRESSIVO ===
    disguise_score = (
        features.get('artificial_informality', 0) * 0.20 +
        features.get('generic_motivation_score', 0) * 0.30 +
        features.get('perfect_grammar_no_punct', 0) * 0.20 +
        (10 - min(features.get('punctuation_ratio', 0.05) * 100, 10)) * 0.10 +
        features.get('lexical_genericity', 0) * 0.15 +
        features.get('repetitive_structure', 0) * 0.05
    )
    
    print(f"\n🔍 DISGUISE ANALYSIS:")
    print(f"   Artificial Informality: {features.get('artificial_informality', 0):.2f}/10")
    print(f"   Generic Motivation: {features.get('generic_motivation_score', 0):.2f}/20")
    print(f"   Perfect Grammar (no punct): {features.get('perfect_grammar_no_punct', 0):.2f}/8")
    print(f"   Lexical Genericity: {features.get('lexical_genericity', 0):.2f}/20")
    print(f"   TOTAL DISGUISE SCORE: {disguise_score:.2f}/10")
    
    # === OVERRIDE DIRETTO SE DISGUISE SCORE È ALTISSIMO ===
    if disguise_score >= 6.0:
        print(f"   ⚠️  DISGUISE OVERRIDE: Score {disguise_score:.2f} >= 6.0 → FORCED AI CLASSIFICATION")
        
        # Calcola confidence basata sul disguise score
        confidence = min(70 + (disguise_score * 3), 95)
        
        return {
            'label': '🤖 AI-Generated',
            'confidence': float(confidence),
            'probabilities': {
                'human': float(100 - confidence),
                'ai': float(confidence)
            },
            'features': features,
            'model_used': 'Disguise Detection Override',
            'reliability': reliability,
            'disguise_score': round(disguise_score, 2),
            'detection_note': '🚨 STRONG disguised AI pattern detected - Classification overridden',
            'override_reason': f'Disguise score {disguise_score:.1f}/10 exceeds threshold (6.0)'
        }
    
    # === PREDIZIONE STANDARD CON MODELLI ===
    base_result = None
    
    if HYBRID_AVAILABLE and BERT_AVAILABLE:
        try:
            encoded = tokenizer(
                processed_text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                bert_outputs = model_bert.bert(
                    input_ids=encoded['input_ids'].to(DEVICE),
                    attention_mask=encoded['attention_mask'].to(DEVICE)
                )
                bert_embedding = bert_outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            original_features = [f for f in STYLOMETRIC_FEATURES if f in features]
            feature_vector = np.array([features.get(f, 0) for f in original_features])
            feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
            
            hybrid_input = np.hstack([bert_embedding, feature_vector_scaled])
            
            prediction = rf_hybrid.predict(hybrid_input)[0]
            probabilities = rf_hybrid.predict_proba(hybrid_input)[0]
            
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            base_result = {
                'prediction': prediction,
                'probabilities': probabilities,
                'confidence': confidence,
                'model_used': 'Hybrid (BERT + Stylometric)'
            }
        
        except Exception as e:
            print(f"⚠️  Hybrid prediction failed: {e}")
    
    if base_result is None and BERT_AVAILABLE:
        try:
            encoded = tokenizer(
                processed_text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model_bert(
                    input_ids=encoded['input_ids'].to(DEVICE),
                    attention_mask=encoded['attention_mask'].to(DEVICE)
                )
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
            prediction = int(probs[1] > 0.5)
            confidence = probs[1] if prediction == 1 else probs[0]
            
            base_result = {
                'prediction': prediction,
                'probabilities': probs,
                'confidence': confidence,
                'model_used': 'BERT Fine-tuned'
            }
        
        except Exception as e:
            print(f"⚠️  BERT prediction failed: {e}")
    
    if base_result is None:
        return predict_local_fallback(processed_text, features, disguise_score)
    
    # === ADJUSTMENT AGGRESSIVO PER AI MASCHERATO ===
    prediction = base_result['prediction']
    probabilities = base_result['probabilities']
    
    # Soglia ABBASSATA + boost AUMENTATO
    if disguise_score > 4.0 and prediction == 0:
        print(f"   🔧 ADJUSTMENT TRIGGERED: Disguise {disguise_score:.2f} > 4.0")
        
        # Boost molto più aggressivo
        ai_boost = min(disguise_score * 12, 60)  # Era *8 max 50
        
        new_ai_prob = min(probabilities[1] + (ai_boost / 100), 0.98)
        new_human_prob = 1 - new_ai_prob
        
        probabilities = np.array([new_human_prob, new_ai_prob])
        
        if new_ai_prob > 0.5:
            prediction = 1
            print(f"   ✅ RECLASSIFIED as AI (boost: +{ai_boost}%)")
    
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    result = {
        'label': '🤖 AI-Generated' if prediction == 1 else '✍️ Human-Written',
        'confidence': float(confidence * 100),
        'probabilities': {
            'human': float(probabilities[0] * 100),
            'ai': float(probabilities[1] * 100)
        },
        'features': features,
        'model_used': base_result['model_used'],
        'reliability': reliability,
        'disguise_score': round(disguise_score, 2)
    }
    
    if reliability < 70:
        result['warning'] = '⚠️ Low confidence: Text has quality issues'
    
    if disguise_score > 4.0 and prediction == 1:
        result['detection_note'] = '🔍 Disguised AI pattern detected and corrected'
    
    if len(text) < 200:
        result['info'] = 'ℹ️ Short text: Accuracy improves with longer samples'
    
    if anomalies['all_lowercase'] or anomalies['no_punctuation']:
        result['text_issues'] = [k for k, v in anomalies.items() if v]
    
    return result

def predict_local_fallback(text, features, disguise_score):
    """Fallback con disguise score integrato"""
    
    ai_score = (
        (features.get('sentence_similarity_drift', 0) * 25) +
        ((1 - features.get('lexical_compression_ratio', 0.7)) * 20) +
        ((5 - min(features.get('burstiness_index', 5), 5)) * 15) +
        (features.get('template_bias_score', 0) * 10) +
        (features.get('generic_motivation_score', 0) * 2) +
        (disguise_score * 3)  # Peso ALTO al disguise score
    )
    
    is_ai = ai_score > 50
    confidence = ai_score if is_ai else (100 - ai_score)
    
    return {
        'label': '🤖 AI-Generated' if is_ai else '✍️ Human-Written',
        'confidence': float(confidence),
        'probabilities': {
            'human': float(100 - ai_score),
            'ai': float(ai_score)
        },
        'features': features,
        'model_used': 'Stylometric Baseline (Fallback)',
        'reliability': 60,
        'disguise_score': round(disguise_score, 2)
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'AI Text Detector API (Ultra-Enhanced)',
        'version': '3.0.0',
        'features': {
            'disguise_detection': 'ULTRA-AGGRESSIVE',
            'override_threshold': 6.0,
            'adjustment_threshold': 4.0
        },
        'models': {
            'bert': BERT_AVAILABLE,
            'hybrid': HYBRID_AVAILABLE
        },
        'device': str(DEVICE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Request body:
    {
        "text": "Your text here..."
    }
    
    Response:
    {
        "label": "🤖 AI-Generated" or "✍️ Human-Written",
        "confidence": 87.5,
        "probabilities": {
            "human": 12.5,
            "ai": 87.5
        },
        "features": {...},
        "model_used": "Hybrid (BERT + Stylometric)",
        "reliability": 85,
        "disguise_score": 6.5,
        "warning": "...",
        "detection_note": "...",
        "info": "..."
    }
    """
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request body'
            }), 400
        
        text = data['text']
        
        # Validate text length
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return jsonify({
                'error': f'Text too short. Minimum {MIN_TEXT_LENGTH} characters required.'
            }), 400
        
        # Perform prediction
        result = predict_text_backend(text)
        
        # Log request
        print(f"\n📝 Prediction request:")
        print(f"   Text length: {len(text)} chars")
        print(f"   Result: {result['label']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Model: {result['model_used']}")
        print(f"   Reliability: {result.get('reliability', 100):.0f}%")
        print(f"   Disguise Score: {result.get('disguise_score', 0):.2f}/10")
        if 'warning' in result:
            print(f"   ⚠️  {result['warning']}")
        if 'detection_note' in result:
            print(f"   🔍 {result['detection_note']}")
        print()
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'bert': {
                'loaded': BERT_AVAILABLE,
                'device': str(DEVICE)
            },
            'hybrid': {
                'loaded': HYBRID_AVAILABLE
            },
            'spacy': {
                'loaded': nlp is not None
            }
        },
        'config': {
            'max_length': MAX_LENGTH,
            'min_text_length': MIN_TEXT_LENGTH,
            'num_features': len(STYLOMETRIC_FEATURES)
        },
        'enhancements': {
            'disguise_detection': True,
            'artificial_informality': True,
            'generic_motivation_detection': True,
            'lexical_genericity_analysis': True
        }
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Return list of stylometric features"""
    return jsonify({
        'features': STYLOMETRIC_FEATURES,
        'count': len(STYLOMETRIC_FEATURES),
        'enhanced_features': [
            'artificial_informality',
            'generic_motivation_score',
            'punctuation_ratio',
            'capitalization_variance',
            'lexical_genericity'
        ]
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" 🚀 FLASK SERVER STARTING - ENHANCED EDITION")
    print("="*70)
    print(f"\n✅ Server ready on http://localhost:5000")
    print(f"✅ BERT: {'Available' if BERT_AVAILABLE else 'Not loaded'}")
    print(f"✅ Hybrid: {'Available' if HYBRID_AVAILABLE else 'Not loaded'}")
    print(f"✅ Disguise Detection: ACTIVE")
    print(f"\n📡 API Endpoints:")
    print(f"   POST /predict    - Main prediction (enhanced)")
    print(f"   GET  /health     - Health check")
    print(f"   GET  /features   - Feature list")
    print(f"\n🔍 New Features:")
    print(f"   • Artificial Informality Detection")
    print(f"   • Generic Motivation Pattern Recognition")
    print(f"   • Lexical Genericity Analysis")
    print(f"   • Text Quality Assessment")
    print(f"   • Disguise Score (0-10)")
    print("\n" + "="*70 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
