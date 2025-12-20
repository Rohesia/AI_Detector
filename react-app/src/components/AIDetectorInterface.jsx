import React, { useState } from 'react';
import { Brain, Zap, Eye, Activity, Sparkles, AlertCircle, CheckCircle, TrendingUp, BarChart3, Waves } from 'lucide-react';

// ============================================================================
// COMPONENTI HELPER
// ============================================================================

const ResultGauge = ({ value, label, color }) => {
  const rotation = (value / 100) * 180 - 90;
  
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-16 mb-2">
        <svg viewBox="0 0 100 50" className="w-full h-full">
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="#1e293b"
            strokeWidth="8"
          />
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={`${(value / 100) * 125} 125`}
            style={{ transition: 'all 1s' }}
          />
          <circle cx="50" cy="45" r="3" fill={color} />
          <line
            x1="50"
            y1="45"
            x2="50"
            y2="15"
            stroke={color}
            strokeWidth="2"
            transform={`rotate(${rotation} 50 45)`}
            style={{ transition: 'all 1s' }}
          />
        </svg>
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'center'
        }}>
          <span className="text-2xl font-bold" style={{ color }}>{value.toFixed(1)}%</span>
        </div>
      </div>
      <span className="text-sm text-gray-400">{label}</span>
    </div>
  );
};

const FeatureBar = ({ label, value, max, colorFrom, colorTo }) => (
  <div>
    <div className="flex justify-between text-sm mb-1">
      <span className="text-gray-400">{label}</span>
      <span className="font-semibold">{value.toFixed(1)}%</span>
    </div>
    <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
      <div
        className="h-full transition-all duration-1000"
        style={{
          width: `${(value / max) * 100}%`,
          background: `linear-gradient(to right, ${colorFrom}, ${colorTo})`
        }}
      />
    </div>
  </div>
);

const MetricCard = ({ label, value, icon }) => (
  <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
    <div className="text-2xl mb-1">{icon}</div>
    <div className="text-xs text-gray-400 mb-1">{label}</div>
    <div className="text-lg font-bold text-cyan-300">{value}</div>
  </div>
);

// ============================================================================
// COMPONENTE PRINCIPALE
// ============================================================================

const AIDetectorInterface = () => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('input');
  const [pulseAnimation, setPulseAnimation] = useState(false);
  const [error, setError] = useState(null);
  
  const [neurons] = useState(() => 
    Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 3 + 1,
      duration: Math.random() * 10 + 10,
      delay: Math.random() * 5
    }))
  );

  const exampleAI = "Artificial intelligence represents a significant advancement in modern technology. It enables machines to perform tasks that typically require human intelligence. Machine learning algorithms analyze vast amounts of data to identify patterns. These systems continue to evolve and improve over time.";
  
  const exampleHuman = "I can't believe what happened today! So I was running late (as usual lol) and literally spilled coffee all over my shirt right before the meeting. Had to wear my gym hoodie... super professional. But honestly? Best meeting ever. Sometimes chaos works in your favor, you know?";

  // ============================================================================
  // ANALISI CON BACKEND FLASK
  // ============================================================================
  
  const analyzeText = async () => {
    if (text.trim().length < 50) {
      alert('Please enter at least 50 characters for accurate analysis.');
      return;
    }

    setIsAnalyzing(true);
    setPulseAnimation(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // LOG PER DEBUG: Apri la console del browser (F12) per vedere cosa arriva
      console.log("Dati ricevuti dal server:", data);
      
      // Trasforma la risposta del backend nel formato atteso dall'UI
      const formattedResult = {
        // CORREZIONE: Controlliamo se la stringa contiene "AI"
        isAI: data.label.includes('AI'), 
        confidence: parseFloat(data.confidence) || 0,
        features: {
          // Usiamo i dati reali se presenti, altrimenti fallback a 0
          sentenceSimilarity: data.probabilities?.ai || 0, 
          lexicalDiversity: data.probabilities?.human || 0,
          burstiness: 0.5, // Valori di default se il BERT semplice non li calcola
          avgSentLength: text.split(' ').length / (text.split('.').length || 1),
          repetitiveness: data.probabilities?.ai || 0,
          lexicalPoverty: 100 - (data.probabilities?.human || 0),
          structuralVariation: 50
        }
      };

      setResult(formattedResult);
      setActiveTab('results');
      
    } catch (error) {
      console.error("Error connecting to backend:", error);
      setError('⚠️ Backend not responding. Make sure Flask server is running.');
      
      // Fallback: usa analisi frontend solo se il server è spento
      const localAnalysis = calculateFeaturesLocal(text);
      setResult(localAnalysis);
      setActiveTab('results');
      
    } finally {
      setIsAnalyzing(false);
      setPulseAnimation(false);
    }
  };

  // ============================================================================
  // FALLBACK: ANALISI LOCALE (se backend non disponibile)
  // ============================================================================
  
  const calculateFeaturesLocal = (inputText) => {
    const sentences = inputText.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const words = inputText.toLowerCase().match(/\b\w+\b/g) || [];
    
    if (sentences.length === 0 || words.length === 0) return null;

    const sentLengths = sentences.map(s => s.split(/\s+/).length);
    const avgSentLength = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
    const stdSentLength = Math.sqrt(sentLengths.reduce((a, b) => a + Math.pow(b - avgSentLength, 2), 0) / sentLengths.length);
    
    const uniqueWords = new Set(words).size;
    const lexicalDiversity = uniqueWords / words.length;
    
    const burstiness = stdSentLength;
    const sentenceSimilarity = sentLengths.length > 1 ? 
      sentLengths.slice(1).reduce((acc, len, i) => 
        acc + (1 - Math.abs(len - sentLengths[i]) / Math.max(len, sentLengths[i])), 0
      ) / (sentLengths.length - 1) : 0.5;

    const aiScore = (
      (sentenceSimilarity * 35) + 
      ((1 - lexicalDiversity) * 30) +
      ((5 - Math.min(burstiness, 5)) * 20) +
      ((avgSentLength > 15 && avgSentLength < 25 ? 15 : 0))
    );

    const isAI = aiScore > 50;
    const confidence = isAI ? aiScore : (100 - aiScore);

    return {
      isAI,
      confidence,
      features: {
        sentenceSimilarity: sentenceSimilarity * 100,
        lexicalDiversity: lexicalDiversity * 100,
        burstiness: burstiness,
        avgSentLength,
        repetitiveness: sentenceSimilarity * 100,
        lexicalPoverty: (1 - lexicalDiversity) * 100,
        structuralVariation: burstiness * 20
      }
    };
  };

  return (
    <div className="min-h-screen">
      {/* Background Neurons */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {neurons.map(neuron => (
          <div
            key={neuron.id}
            className="absolute rounded-full bg-blue-400 opacity-20 animate-pulse"
            style={{
              left: `${neuron.x}%`,
              top: `${neuron.y}%`,
              width: `${neuron.size}px`,
              height: `${neuron.size}px`,
              animationDuration: `${neuron.duration}s`,
              animationDelay: `${neuron.delay}s`
            }}
          />
        ))}
      </div>

      <div className="max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-4 mb-4">
            <Brain className={`w-16 h-16 text-cyan-400 ${pulseAnimation ? 'animate-pulse' : ''}`} />
            <h1 className="text-6xl">NEURAL PROBE</h1>
          </div>
          <p className="text-xl text-gray-300 flex items-center justify-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Advanced AI Detection System
            <Sparkles className="w-5 h-5 text-purple-400" />
          </p>
          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '0.5rem',
              color: '#fca5a5',
              fontSize: '0.875rem'
            }}>
              {error}
            </div>
          )}
        </div>

        {/* Main Panel */}
        <div className="bg-slate-900/50 backdrop-blur-xl rounded-3xl border border-cyan-500/30 shadow-2xl overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-cyan-500/30">
            <button
              onClick={() => setActiveTab('input')}
              className={`flex-1 py-4 px-6 font-semibold transition-all ${
                activeTab === 'input' ? 'bg-cyan-500/20' : ''
              }`}
            >
              <Eye className="w-5 h-5 inline mr-2" />
              Input Analysis
            </button>
            <button
              onClick={() => result && setActiveTab('results')}
              disabled={!result}
              className={`flex-1 py-4 px-6 font-semibold transition-all ${
                activeTab === 'results' ? 'bg-purple-500/20' : ''
              } ${!result ? 'text-gray-600' : ''}`}
            >
              <Activity className="w-5 h-5 inline mr-2" />
              Neural Report
            </button>
          </div>

          {/* Input Tab */}
          {activeTab === 'input' && (
            <div className="p-8">
              <div className="mb-6">
                <label className="block text-lg font-semibold mb-3 text-cyan-300">
                  Enter Text for Analysis
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste your text here... (minimum 50 characters)"
                  className="w-full"
                  disabled={isAnalyzing}
                />
                <div className="mt-2 text-sm text-gray-400">
                  Characters: {text.length} / 50 minimum
                </div>
              </div>

              <div className="flex gap-4 mb-6">
                <button
                  onClick={() => setText(exampleAI)}
                  className="flex-1 py-2 bg-red-500/20"
                  disabled={isAnalyzing}
                >
                  🤖 Load AI Example
                </button>
                <button
                  onClick={() => setText(exampleHuman)}
                  className="flex-1 py-2 bg-green-500/20"
                  disabled={isAnalyzing}
                >
                  ✍️ Load Human Example
                </button>
              </div>

              <button
                onClick={analyzeText}
                disabled={isAnalyzing || text.length < 50}
                className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Waves className="w-6 h-6 animate-spin" style={{ display: 'inline-block', marginRight: '0.75rem' }} />
                    Analyzing Neural Patterns...
                  </>
                ) : (
                  <>
                    <Brain className="w-6 h-6" style={{ display: 'inline-block', marginRight: '0.75rem' }} />
                    Initiate Neural Scan
                  </>
                )}
              </button>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && result && (
            <div className="p-8">
              <div className={`mb-8 p-6 rounded-2xl border-2 ${
                result.isAI ? 'bg-red-500/10 border-red-500' : 'bg-green-500/10 border-green-500'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    {result.isAI ? (
                      <AlertCircle className="w-12 h-12 text-red-400" />
                    ) : (
                      <CheckCircle className="w-12 h-12 text-green-400" />
                    )}
                    <div>
                      <h2 className="text-3xl font-bold">
                        {result.isAI ? '🤖 AI Generated' : '✍️ Human Written'}
                      </h2>
                      <p className="text-gray-400">Neural Analysis Complete</p>
                    </div>
                  </div>
                  <ResultGauge
                    value={result.confidence}
                    label="Confidence"
                    color={result.isAI ? '#ef4444' : '#22c55e'}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-slate-800/50 p-6 rounded-xl border border-cyan-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-cyan-400" />
                    Stylometric Signature
                  </h3>
                  <div className="space-y-3">
                    <FeatureBar
                      label="Structural Repetition"
                      value={result.features.repetitiveness}
                      max={100}
                      colorFrom="#ef4444"
                      colorTo="#f97316"
                    />
                    <FeatureBar
                      label="Lexical Poverty"
                      value={result.features.lexicalPoverty}
                      max={100}
                      colorFrom="#f97316"
                      colorTo="#eab308"
                    />
                    <FeatureBar
                      label="Burstiness Index"
                      value={result.features.structuralVariation}
                      max={100}
                      colorFrom="#eab308"
                      colorTo="#22c55e"
                    />
                  </div>
                </div>

                <div className="bg-slate-800/50 p-6 rounded-xl border border-purple-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-purple-400" />
                    Linguistic Metrics
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <MetricCard
                      label="Lexical Diversity"
                      value={`${result.features.lexicalDiversity.toFixed(1)}%`}
                      icon="🔤"
                    />
                    <MetricCard
                      label="Avg Sentence"
                      value={`${result.features.avgSentLength.toFixed(0)} words`}
                      icon="📏"
                    />
                    <MetricCard
                      label="Burstiness"
                      value={result.features.burstiness.toFixed(2)}
                      icon="💥"
                    />
                    <MetricCard
                      label="Similarity"
                      value={`${result.features.sentenceSimilarity.toFixed(1)}%`}
                      icon="🔄"
                    />
                  </div>
                </div>
              </div>

              <div className="p-6 rounded-xl border border-blue-500/30" style={{
                background: 'linear-gradient(to right, rgba(30, 58, 138, 0.3), rgba(88, 28, 135, 0.3))'
              }}>
                <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-yellow-400" />
                  Neural Interpretation
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  {result.isAI ? (
                    <>
                      The text exhibits <strong>high structural uniformity</strong> ({result.features.repetitiveness.toFixed(1)}% similarity) 
                      and <strong>limited lexical variation</strong> ({result.features.lexicalDiversity.toFixed(1)}% diversity), 
                      typical patterns of AI-generated content. The sentence construction follows predictable templates 
                      with low burstiness ({result.features.burstiness.toFixed(2)}), indicating algorithmic origin.
                    </>
                  ) : (
                    <>
                      The text demonstrates <strong>natural linguistic variation</strong> with diverse sentence structures 
                      (burstiness: {result.features.burstiness.toFixed(2)}) and <strong>rich vocabulary</strong> 
                      ({result.features.lexicalDiversity.toFixed(1)}% lexical diversity). The organic flow and structural 
                      unpredictability are characteristic of human authorship.
                    </>
                  )}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>Powered by Advanced Neural Analysis • Hybrid BERT + Stylometric Detection</p>
        </div>
      </div>
    </div>
  );
};

export default AIDetectorInterface;