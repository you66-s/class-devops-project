import React, { useState } from 'react';
import EmailInput from './components/EmailInput';
import ResultCard from './components/ResultCard';
import History from './components/History';
import './App.css';

// Mock API function
const analyzeEmail = (emailText) => {
  // Simulate API delay
  return new Promise((resolve) => {
    setTimeout(() => {
      const isSpam = Math.random() > 0.5;
      const confidence = Math.floor(Math.random() * 40) + 60; // 60-99%
      resolve({ isSpam, confidence });
    }, 500);
  });
};

function App() {
  const [history, setHistory] = useState([
    {
      id: 1,
      email: "Vous avez été sélectionné comme gagnant d'un iPhone 15 GRATUIT.",
      isSpam: true,
      confidence: 95,
      timestamp: Date.now() - 86400000 * 3 // 3 days ago
    },
    {
      id: 2,
      email: "N'oubliez pas notre réunion lundi à 10h pour discuter de l'avancement du projet.",
      isSpam: false,
      confidence: 78,
      timestamp: Date.now() - 86400000 * 2 // 2 days ago
    },
    {
      id: 3,
      email: "Travaillez depuis chez vous et gagnez jusqu'à 5000€ par semaine.",
      isSpam: true,
      confidence: 92,
      timestamp: Date.now() - 86400000 // 1 day ago
    }
  ]);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (emailText) => {
    setLoading(true);
    try {
      const result = await analyzeEmail(emailText);
      setCurrentResult(result);
      const newEntry = {
        id: Date.now(),
        email: emailText,
        ...result,
        timestamp: Date.now()
      };
      setHistory(prev => [newEntry, ...prev]);
    } catch (error) {
      console.error('Error analyzing email:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Spam Detection System</h1>
      <div className="container">
        <EmailInput onAnalyze={handleAnalyze} />
        {loading && <p>Analyzing...</p>}
        <ResultCard result={currentResult} />
        <History history={history} />
      </div>
    </div>
  );
}

export default App;
