import React, { useState } from 'react';
import { Brain, Zap, Eye, Activity, Sparkles, AlertCircle, CheckCircle, TrendingUp, BarChart3, Waves } from 'lucide-react';
import jsPDF from 'jspdf';

// ============================================================================
// COMPONENTI HELPER
// ============================================================================

/* 
const response = await fetch("http://backend:8000/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text })
});

 */

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
  
  const exampleHuman = "Today it's a beautiful day for all of us, because we are alive and we have to be happy";

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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Dati ricevuti dal server:", data);

      const formattedResult = {
        isAI: data.label.includes('AI'),
        confidence: parseFloat(data.confidence) || 0,
        features: {
          sentenceSimilarity: data.probabilities?.ai || 0,
          lexicalDiversity: data.probabilities?.human || 0,
          burstiness: 0.5,
          avgSentLength: text.split(' ').length / (text.split('.').length || 1),
          repetitiveness: data.probabilities?.ai || 0,
          lexicalPoverty: 100 - (data.probabilities?.human || 0),
          structuralVariation: 50
        }
      };

      setResult(formattedResult);
      setActiveTab('results');

    } catch (err) {
      console.error("Error connecting to backend:", err);
      setError('⚠️ Backend not responding. Using local analysis fallback.');
      const localAnalysis = calculateFeaturesLocal(text);
      setResult(localAnalysis);
      setActiveTab('results');
    } finally {
      setIsAnalyzing(false);
      setPulseAnimation(false);
    }
  };

  // ============================================================================  
  // ANALISI LOCALE
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

  // ============================================================================  
  // DOWNLOAD REPORT PDF
  // ============================================================================

  /* const downloadReport = () => {
    if (!result) return;
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const isAI = result.isAI;
    
    // ============ PALETTE COLORI CYBERPUNK ============
    const colors = {
      bg: [10, 14, 26],
      cyan: [0, 240, 255],
      purple: [191, 0, 255],
      magenta: [255, 0, 170],
      lime: [0, 255, 136],
      orange: [255, 170, 0],
      red: [255, 0, 85],
      gray: [100, 116, 139],
      lightGray: [203, 213, 225]
    };

    // ============ SFONDO PRINCIPALE ============
    doc.setFillColor(...colors.bg);
    doc.rect(0, 0, pageWidth, pageHeight, 'F');

    // ============ WATERMARK PATTERN ============
    doc.setTextColor(30, 41, 59);
    doc.setFontSize(60);
    doc.setFont("helvetica", "bold");
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        doc.text("NP", 20 + i * 60, 60 + j * 70, { angle: 45, opacity: 0.03 });
      }
    }

    // ============ HEADER GRADIENT SIMULATO ============
    for (let i = 0; i < 35; i++) {
      const ratio = i / 35;
      const r = Math.floor(colors.cyan[0] + (colors.purple[0] - colors.cyan[0]) * ratio);
      const g = Math.floor(colors.cyan[1] + (colors.purple[1] - colors.cyan[1]) * ratio);
      const b = Math.floor(colors.cyan[2] + (colors.purple[2] - colors.cyan[2]) * ratio);
      doc.setFillColor(r, g, b);
      doc.rect(0, i, pageWidth, 1, 'F');
    }

    // ============ LOGO + TITOLO ============
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(32);
    doc.setFont("helvetica", "bold");
    doc.text("NEURAL PROBE", pageWidth / 2, 18, { align: 'center' });
    
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.text("AI DETECTION ANALYSIS REPORT", pageWidth / 2, 25, { align: 'center' });
    
    // Report ID e Timestamp
    const reportId = `NP-${Date.now().toString(36).toUpperCase()}`;
    const timestamp = new Date().toLocaleString('en-US', { 
      dateStyle: 'medium', 
      timeStyle: 'short',
      timeZone: 'UTC'
    });
    doc.setTextColor(...colors.lightGray);
    doc.setFontSize(7);
    doc.text(`Report ID: ${reportId}`, pageWidth / 2, 30, { align: 'center' });
    doc.text(`Generated: ${timestamp} UTC`, pageWidth / 2, 33, { align: 'center' });

    // ============ RESULT CARD CON GLOW EFFECT ============
    const cardY = 42;
    const cardHeight = 32;
    
    // Glow simulato (ombre multiple)
    const glowColor = isAI ? colors.red : colors.lime;
    for (let i = 3; i > 0; i--) {
      doc.setDrawColor(glowColor[0], glowColor[1], glowColor[2]);
      doc.setLineWidth(i * 0.5);
      doc.roundedRect(15 - i, cardY - i, pageWidth - 30 + i * 2, cardHeight + i * 2, 4, 4, 'S');
    }
    
    // Card principale
    const cardGradientSteps = 20;
    for (let i = 0; i < cardGradientSteps; i++) {
      const ratio = i / cardGradientSteps;
      const r = isAI ? 
        Math.floor(colors.red[0] + (colors.magenta[0] - colors.red[0]) * ratio) :
        Math.floor(colors.lime[0] + (colors.cyan[0] - colors.lime[0]) * ratio);
      const g = isAI ?
        Math.floor(colors.red[1] + (colors.magenta[1] - colors.red[1]) * ratio) :
        Math.floor(colors.lime[1] + (colors.cyan[1] - colors.lime[1]) * ratio);
      const b = isAI ?
        Math.floor(colors.red[2] + (colors.magenta[2] - colors.red[2]) * ratio) :
        Math.floor(colors.lime[2] + (colors.cyan[2] - colors.lime[2]) * ratio);
      doc.setFillColor(r, g, b);
      const stepHeight = cardHeight / cardGradientSteps;
      doc.rect(15, cardY + i * stepHeight, pageWidth - 30, stepHeight, 'F');
    }
    
    // Bordo card
    doc.setDrawColor(255, 255, 255);
    doc.setLineWidth(0.5);
    doc.roundedRect(15, cardY, pageWidth - 30, cardHeight, 4, 4, 'S');

    // Icona e testo risultato
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(40);
    doc.text(isAI ? "⚠" : "✓", 25, cardY + 22);
    
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text(isAI ? "AI GENERATED" : "HUMAN WRITTEN", pageWidth / 2, cardY + 15, { align: 'center' });
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text(`Confidence: ${result.confidence.toFixed(1)}%`, pageWidth / 2, cardY + 23, { align: 'center' });
    
    // Risk Badge
    const riskLevel = result.confidence > 80 ? "HIGH" : result.confidence > 60 ? "MEDIUM" : "LOW";
    const badgeColor = result.confidence > 80 ? colors.red : result.confidence > 60 ? colors.orange : colors.lime;
    doc.setFillColor(...badgeColor);
    doc.roundedRect(pageWidth / 2 - 15, cardY + 26, 30, 6, 2, 2, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(7);
    doc.setFont("helvetica", "bold");
    doc.text(`${riskLevel} RISK`, pageWidth / 2, cardY + 30, { align: 'center' });

    // ============ CIRCULAR CONFIDENCE GAUGE ============
    const gaugeX = pageWidth - 35;
    const gaugeY = cardY + 16;
    const gaugeRadius = 12;
    
    // Sfondo gauge
    doc.setDrawColor(30, 41, 59);
    doc.setLineWidth(3);
    doc.circle(gaugeX, gaugeY, gaugeRadius, 'S');
    
    // Arc riempito
    doc.setDrawColor(...glowColor);
    doc.setLineWidth(3);
    const arcAngle = (result.confidence / 100) * 360;
    for (let i = 0; i < arcAngle; i += 5) {
      const rad1 = ((i - 90) * Math.PI) / 180;
      const rad2 = ((i - 85) * Math.PI) / 180;
      const x1 = gaugeX + gaugeRadius * Math.cos(rad1);
      const y1 = gaugeY + gaugeRadius * Math.sin(rad1);
      const x2 = gaugeX + gaugeRadius * Math.cos(rad2);
      const y2 = gaugeY + gaugeRadius * Math.sin(rad2);
      doc.line(x1, y1, x2, y2);
    }
    
    doc.setTextColor(...glowColor);
    doc.setFontSize(9);
    doc.setFont("helvetica", "bold");
    doc.text(`${result.confidence.toFixed(0)}%`, gaugeX, gaugeY + 1, { align: 'center' });

    // ============ LAYOUT A 3 COLONNE ============
    const colY = 82;
    
    // COLONNA 1: QUICK STATS (SIDEBAR)
    doc.setFillColor(15, 23, 42);
    doc.roundedRect(10, colY, 45, 120, 3, 3, 'F');
    doc.setDrawColor(...colors.cyan);
    doc.setLineWidth(0.3);
    doc.roundedRect(10, colY, 45, 120, 3, 3, 'S');
    
    doc.setTextColor(...colors.cyan);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.text("QUICK STATS", 32.5, colY + 8, { align: 'center' });
    
    const stats = [
      { label: "Words", value: text.split(/\s+/).length },
      { label: "Sentences", value: text.split(/[.!?]+/).filter(s => s.trim()).length },
      { label: "Avg/Sentence", value: result.features.avgSentLength.toFixed(0) },
      { label: "Unique Words", value: new Set(text.toLowerCase().match(/\b\w+\b/g) || []).size }
    ];
    
    stats.forEach((stat, i) => {
      const y = colY + 20 + i * 22;
      doc.setFillColor(30, 41, 59);
      doc.roundedRect(13, y, 39, 18, 2, 2, 'F');
      doc.setTextColor(...colors.gray);
      doc.setFontSize(7);
      doc.text(stat.label.toUpperCase(), 32.5, y + 6, { align: 'center' });
      doc.setTextColor(...colors.cyan);
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text(String(stat.value), 32.5, y + 14, { align: 'center' });
    });

    // COLONNA 2: MAIN CONTENT
    const mainX = 60;
    const mainWidth = 90;
    
    // Stylometric Radar Chart (simulato con poligono)
    doc.setTextColor(...colors.purple);
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("STYLOMETRIC SIGNATURE", mainX, colY + 5);
    
    const radarCenterX = mainX + mainWidth / 2;
    const radarCenterY = colY + 35;
    const radarRadius = 25;
    const metrics = [
      { label: "Rep", value: result.features.repetitiveness },
      { label: "Lex", value: result.features.lexicalPoverty },
      { label: "Bur", value: result.features.structuralVariation },
      { label: "Sim", value: result.features.sentenceSimilarity },
      { label: "Div", value: result.features.lexicalDiversity },
      { label: "Avg", value: Math.min(result.features.avgSentLength * 4, 100) }
    ];
    
    // Griglia radar
    doc.setDrawColor(51, 65, 85);
    doc.setLineWidth(0.2);
    for (let r = radarRadius; r > 0; r -= radarRadius / 4) {
      for (let i = 0; i < 6; i++) {
        const angle1 = (i * 60 - 90) * Math.PI / 180;
        const angle2 = ((i + 1) * 60 - 90) * Math.PI / 180;
        const x1 = radarCenterX + r * Math.cos(angle1);
        const y1 = radarCenterY + r * Math.sin(angle1);
        const x2 = radarCenterX + r * Math.cos(angle2);
        const y2 = radarCenterY + r * Math.sin(angle2);
        doc.line(x1, y1, x2, y2);
      }
    }
    
    // Assi
    metrics.forEach((_, i) => {
      const angle = (i * 60 - 90) * Math.PI / 180;
      const x = radarCenterX + radarRadius * Math.cos(angle);
      const y = radarCenterY + radarRadius * Math.sin(angle);
      doc.line(radarCenterX, radarCenterY, x, y);
    });
    
    // Poligono dati
    doc.setDrawColor(...colors.magenta);
    doc.setFillColor(...colors.magenta);
    doc.setLineWidth(0.5);
    const points = metrics.map((m, i) => {
      const angle = (i * 60 - 90) * Math.PI / 180;
      const r = (m.value / 100) * radarRadius;
      return {
        x: radarCenterX + r * Math.cos(angle),
        y: radarCenterY + r * Math.sin(angle)
      };
    });
    
    points.forEach((p, i) => {
      const next = points[(i + 1) % points.length];
      doc.line(p.x, p.y, next.x, next.y);
      doc.circle(p.x, p.y, 1, 'F');
    });
    
    // Labels
    doc.setTextColor(...colors.lightGray);
    doc.setFontSize(6);
    metrics.forEach((m, i) => {
      const angle = (i * 60 - 90) * Math.PI / 180;
      const x = radarCenterX + (radarRadius + 8) * Math.cos(angle);
      const y = radarCenterY + (radarRadius + 8) * Math.sin(angle);
      doc.text(m.label, x, y, { align: 'center' });
    });

    // Linguistic Heatmap
    doc.setTextColor(...colors.cyan);
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("LINGUISTIC HEATMAP", mainX, colY + 75);
    
    const heatmapMetrics = [
      { label: "Diversity", value: result.features.lexicalDiversity, icon: "D" },
      { label: "Burstiness", value: result.features.burstiness * 10, icon: "B" },
      { label: "Similarity", value: result.features.sentenceSimilarity, icon: "S" },
      { label: "Variation", value: result.features.structuralVariation, icon: "V" }
    ];
    
    heatmapMetrics.forEach((m, i) => {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const x = mainX + col * 43;
      const y = colY + 83 + row * 24;
      
      // Colore basato sul valore
      const intensity = Math.min(m.value / 100, 1);
      const r = Math.floor(255 * intensity);
      const g = Math.floor(255 * (1 - intensity));
      const b = 100;
      
      doc.setFillColor(r, g, b);
      doc.roundedRect(x, y, 40, 20, 2, 2, 'F');
      
      doc.setDrawColor(51, 65, 85);
      doc.setLineWidth(0.3);
      doc.roundedRect(x, y, 40, 20, 2, 2, 'S');
      
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      doc.text(m.icon, x + 8, y + 13);
      
      doc.setFontSize(7);
      doc.text(m.label.toUpperCase(), x + 16, y + 8);
      doc.setFontSize(10);
      doc.setFont("helvetica", "bold");
      doc.text(`${m.value.toFixed(0)}%`, x + 16, y + 16);
    });

    // COLONNA 3: INSIGHTS (SIDEBAR DESTRA)
    const insightX = 155;
    doc.setFillColor(15, 23, 42);
    doc.roundedRect(insightX, colY, 45, 120, 3, 3, 'F');
    doc.setDrawColor(...colors.purple);
    doc.setLineWidth(0.3);
    doc.roundedRect(insightX, colY, 45, 120, 3, 3, 'S');
    
    doc.setTextColor(...colors.purple);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.text("KEY INSIGHTS", insightX + 22.5, colY + 8, { align: 'center' });
    
    const insights = [
      { icon: "📊", label: "Pattern", value: isAI ? "Uniform" : "Varied" },
      { icon: "🎯", label: "Risk", value: riskLevel },
      { icon: "🔍", label: "Model", value: "BERT v2" },
      { icon: "⚡", label: "Speed", value: "Fast" }
    ];
    
    insights.forEach((ins, i) => {
      const y = colY + 20 + i * 22;
      doc.setFillColor(30, 41, 59);
      doc.roundedRect(insightX + 3, y, 39, 18, 2, 2, 'F');
      doc.setFontSize(14);
      doc.text(ins.icon, insightX + 8, y + 12);
      doc.setTextColor(...colors.gray);
      doc.setFontSize(7);
      doc.text(ins.label.toUpperCase(), insightX + 16, y + 7);
      doc.setTextColor(...colors.purple);
      doc.setFontSize(9);
      doc.setFont("helvetica", "bold");
      doc.text(ins.value, insightX + 16, y + 14);
    });

    // ============ NEURAL INTERPRETATION ============
    const interpretY = 210;
    doc.setFillColor(88, 28, 135);
    doc.roundedRect(10, interpretY, pageWidth - 20, 35, 3, 3, 'F');
    
    doc.setDrawColor(...colors.purple);
    doc.setLineWidth(0.5);
    doc.roundedRect(10, interpretY, pageWidth - 20, 35, 3, 3, 'S');
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.text("⚡ NEURAL INTERPRETATION", 15, interpretY + 8);
    
    doc.setFontSize(8);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...colors.lightGray);
    const interpretation = isAI 
      ? `High structural uniformity (${result.features.repetitiveness.toFixed(1)}%) and limited lexical variation (${result.features.lexicalDiversity.toFixed(1)}% diversity) indicate algorithmic origin. The predictable sentence construction (burstiness: ${result.features.burstiness.toFixed(2)}) and consistent patterns are characteristic of AI-generated content.`
      : `Natural linguistic variation with diverse structures (burstiness: ${result.features.burstiness.toFixed(2)}) and rich vocabulary (${result.features.lexicalDiversity.toFixed(1)}% diversity) demonstrate human authorship. The organic flow and unpredictability are incompatible with current AI models.`;
    
    const splitInterpret = doc.splitTextToSize(interpretation, pageWidth - 30);
    doc.text(splitInterpret, 15, interpretY + 16);

    // ============ FOOTER ============
    doc.setFillColor(30, 41, 59);
    doc.rect(0, 255, pageWidth, 42, 'F');
    
    // Logo mini
    doc.setTextColor(...colors.cyan);
    doc.setFontSize(8);
    doc.setFont("helvetica", "bold");
    doc.text("NP", 15, 265);
    
    doc.setTextColor(...colors.gray);
    doc.setFontSize(7);
    doc.setFont("helvetica", "normal");
    doc.text("POWERED BY NEURAL PROBE", 15, 270);
    doc.text("Hybrid BERT + Stylometric Detection Engine", 15, 275);
    doc.text("Model v2.1 | Accuracy: 94.3%", 15, 280);
    
    // Legal
    doc.setFontSize(6);
    doc.setTextColor(100, 100, 100);
    const legal = "This report is generated by automated AI detection systems and should be used as a guide only. For critical decisions, manual review is recommended.";
    const splitLegal = doc.splitTextToSize(legal, 110);
    doc.text(splitLegal, 100, 268);
    
    // QR Code placeholder
    doc.setDrawColor(...colors.cyan);
    doc.setLineWidth(1);
    doc.rect(pageWidth - 30, 260, 20, 20, 'S');
    doc.setTextColor(...colors.cyan);
    doc.setFontSize(5);
    doc.text("VERIFY", pageWidth - 20, 282, { align: 'center' });

    doc.save(`NeuralProbe_${reportId}.pdf`);
  }; */

  // ============================================================================
// FIX: NEURAL PROBE REPORT v2.6 (Fixed Bezier & Enhanced Layout)
// ============================================================================

const downloadReport = () => {
  if (!result) return;
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const isAI = result.isAI;
  
  // Palette Colori Raffinata
  const theme = {
    primary: isAI ? [220, 38, 38] : [22, 163, 74], 
    accent: [79, 70, 229], 
    dark: [15, 23, 42],
    slate: [71, 85, 105],
    light: [248, 250, 252],
    white: [255, 255, 255]
  };

  // --- SFONDO E HEADER ---
  doc.setFillColor(...theme.light);
  doc.rect(0, 0, pageWidth, pageHeight, 'F');
  
  doc.setFillColor(...theme.dark);
  doc.rect(0, 0, pageWidth, 45, 'F');
  
  doc.setTextColor(...theme.white);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(22);
  doc.text("NEURAL PROBE", 15, 20);
  
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(160, 160, 160);
  doc.text("CERTIFIED AI CONTENT ANALYSIS", 15, 27);

  // Badge Risultato
  const statusLabel = isAI ? "AI GENERATED" : "HUMAN AUTHORED";
  doc.setFillColor(...theme.primary);
  doc.roundedRect(pageWidth - 65, 15, 50, 12, 1, 1, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(10);
  doc.text(statusLabel, pageWidth - 40, 23, { align: 'center' });

  // --- SCORE SECTION (Donut) ---
  const centerX = 40;
  const centerY = 85;
  const radius = 20;

  doc.setDrawColor(230, 230, 230);
  doc.setLineWidth(3);
  doc.circle(centerX, centerY, radius, 'S');

  doc.setDrawColor(...theme.primary);
  const segments = (result.confidence / 100) * 40; 
  for (let i = 0; i < segments; i++) {
    const angle = (i * (360 / 40) - 90) * Math.PI / 180;
    const x1 = centerX + radius * Math.cos(angle);
    const y1 = centerY + radius * Math.sin(angle);
    const x2 = centerX + (radius) * Math.cos(angle + 0.1);
    const y2 = centerY + (radius) * Math.sin(angle + 0.1);
    doc.line(x1, y1, x2, y2);
  }

  doc.setTextColor(...theme.dark);
  doc.setFontSize(20);
  doc.text(`${result.confidence.toFixed(0)}%`, centerX, centerY + 2, { align: 'center' });
  doc.setFontSize(7);
  doc.text("CONFIDENCE", centerX, centerY + 9, { align: 'center' });

  // --- SUMMARY CARD ---
  const cardX = 75;
  doc.setFillColor(...theme.white);
  doc.roundedRect(cardX, 60, pageWidth - cardX - 15, 50, 2, 2, 'F');
  doc.setDrawColor(230, 230, 230);
  doc.setLineWidth(0.2);
  doc.rect(cardX, 60, pageWidth - cardX - 15, 50, 'S');

  doc.setTextColor(...theme.dark);
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.text("Executive Forensic Summary", cardX + 5, 70);
  
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...theme.slate);
  const summaryText = isAI 
    ? "Analysis identified high structural uniformity and low lexical variation. The linguistic 'DNA' matches patterns typically found in generative pre-trained transformers (GPT)."
    : "Analysis revealed high natural variance (burstiness) and complex sentence structure. The content demonstrates cognitive flexibility unique to human authorship.";
  const splitText = doc.splitTextToSize(summaryText, pageWidth - cardX - 25);
  doc.text(splitText, cardX + 5, 78);

  // --- LINGUISTIC METRICS (Bar Chart) ---
  const chartY = 130;
  doc.setTextColor(...theme.dark);
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("Linguistic Metrics", 15, chartY);

  const metrics = [
    { label: "Lexical Diversity", val: result.features.lexicalDiversity },
    { label: "Syntactic Complexity", val: result.features.structuralVariation },
    { label: "Burstiness (Entropy)", val: result.features.burstiness * 10 },
    { label: "Probability Score", val: result.confidence }
  ];

  metrics.forEach((m, i) => {
    const yPos = chartY + 15 + (i * 12);
    doc.setFontSize(8);
    doc.setTextColor(...theme.slate);
    doc.text(m.label, 15, yPos);
    
    // Bar Track
    doc.setFillColor(235, 235, 235);
    doc.rect(55, yPos - 3, 100, 3, 'F');
    
    // Bar Fill
    const barWidth = Math.min(Math.max(m.val, 2), 100);
    doc.setFillColor(...(m.val > 70 && isAI ? theme.primary : theme.accent));
    doc.rect(55, yPos - 3, barWidth, 3, 'F');
    
    doc.setTextColor(...theme.dark);
    doc.text(`${m.val.toFixed(1)}%`, 160, yPos);
  });

  // --- DISTRIBUTION ANALYSIS (Simulated Violin/Wave) ---
  const distY = 200;
  doc.setTextColor(...theme.dark);
  doc.setFontSize(11);
  doc.text("Stylometric Distribution", 15, distY);
  
  // Creiamo una "forma" di distribuzione usando rettangoli sottili per massima compatibilità
  for (let i = 0; i < 40; i++) {
    const h = 5 + Math.sin(i * 0.2) * 15 + (Math.random() * 5);
    doc.setFillColor(theme.accent[0], theme.accent[1], theme.accent[2], 0.5);
    doc.rect(15 + (i * 2), distY + 25 - (h/2), 1.5, h, 'F');
  }
  doc.setDrawColor(...theme.primary);
  doc.line(15, distY + 25, 95, distY + 25); // Baseline

  // --- DATA GRID ---
  const gridX = 110;
  doc.setFillColor(240, 240, 245);
  doc.roundedRect(gridX, distY + 5, 85, 35, 2, 2, 'F');
  
  const stats = [
    ["Tokens Analyzed", text.split(/\s+/).length],
    ["Sentence Mean", `${result.features.avgSentLength.toFixed(1)}`],
    ["Unique Lemmas", new Set(text.toLowerCase().match(/\b\w+\b/g)).size]
  ];

  stats.forEach((s, i) => {
    doc.setFontSize(8);
    doc.setTextColor(...theme.slate);
    doc.text(s[0], gridX + 5, distY + 15 + (i * 8));
    doc.setTextColor(...theme.dark);
    doc.text(String(s[1]), gridX + 80, distY + 15 + (i * 8), { align: 'right' });
  });

  // --- FOOTER ---
  const footerY = 275;
  doc.setDrawColor(200, 200, 200);
  doc.setLineWidth(0.1);
  doc.line(15, footerY, pageWidth - 15, footerY);
  
  doc.setFontSize(7);
  doc.setTextColor(150, 150, 150);
  doc.text(`REPORT ID: NP-${Math.random().toString(36).substr(2, 9).toUpperCase()}`, 15, footerY + 8);
  doc.text("Generated by Neural Probe AI Forensic Engine v2.6", 15, footerY + 13);
  
  doc.text("Page 1 of 1", pageWidth - 15, footerY + 8, { align: 'right' });

  doc.save("NeuralProbe_Analysis_Report.pdf");
};

  // ============================================================================  
  // RENDER
  // ============================================================================

  return (
    <div className="min-h-screen relative">
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

      <div className="max-w-7xl mx-auto relative z-10">
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
            <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded text-red-600 text-sm">
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
              className={`flex-1 py-4 px-6 font-semibold transition-all ${activeTab === 'input' ? 'bg-cyan-500/20' : ''}`}
            >
              <Eye className="w-5 h-5 inline mr-2" />
              Input Analysis
            </button>
            <button
              onClick={() => result && setActiveTab('results')}
              disabled={!result}
              className={`flex-1 py-4 px-6 font-semibold transition-all ${activeTab === 'results' ? 'bg-purple-500/20' : ''} ${!result ? 'text-gray-600' : ''}`}
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
                  className="w-full p-3 rounded bg-slate-800 text-white"
                  disabled={isAnalyzing}
                  rows={6}
                />
                <div className="mt-2 text-sm text-gray-400">
                  Characters: {text.length} / 50 minimum
                </div>
              </div>

              <div className="flex gap-4 mb-6">
                <button
                  onClick={() => setText(exampleAI)}
                  className="flex-1 py-2 bg-red-500/20 rounded"
                  disabled={isAnalyzing}
                >
                  🤖 Load AI Example
                </button>
                <button
                  onClick={() => setText(exampleHuman)}
                  className="flex-1 py-2 bg-green-500/20 rounded"
                  disabled={isAnalyzing}
                >
                  ✍️ Load Human Example
                </button>
              </div>

              <button
                onClick={analyzeText}
                disabled={isAnalyzing || text.length < 50}
                className="w-full py-3 rounded-xl bg-cyan-500/20 border border-cyan-400/40 text-cyan-300 font-semibold tracking-wide hover:bg-cyan-400/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Waves className="w-6 h-6 animate-spin inline-block mr-3" />
                    Analyzing Neural Patterns...
                  </>
                ) : (
                  <>
                    <Brain className="w-6 h-6 inline-block mr-3" />
                    Initiate Neural Scan
                  </>
                )}
              </button>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && result && (
            <div className="p-8">
              {/* Summary */}
              <div className={`mb-8 p-6 rounded-2xl border-2 ${result.isAI ? 'bg-red-500/10 border-red-500' : 'bg-green-500/10 border-green-500'}`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    {result.isAI ? (
                      <AlertCircle className="w-12 h-12 text-red-400" />
                    ) : (
                      <CheckCircle className="w-12 h-12 text-green-400" />
                    )}
                    <div>
                      <h2 className="text-3xl font-bold">{result.isAI ? '🤖 AI Generated' : '✍️ Human Written'}</h2>
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

              {/* Feature Panels */}
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

              {/* Neural Interpretation */}
              <div className="p-6 rounded-xl border border-blue-500/30 mb-6" style={{
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

              {/* Download Button */}
              <button
                onClick={downloadReport}
                className="mt-6 w-full py-3 rounded-xl bg-cyan-500/20 border border-cyan-400/40 text-cyan-300 font-semibold tracking-wide hover:bg-cyan-400/30 transition-all"
              >
                ⬇ Download Neural Report
              </button>
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

