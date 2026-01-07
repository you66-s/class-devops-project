import React from 'react';

const ResultCard = ({ result }) => {
  if (!result) {
    return <div>No result yet. Analyze an email to see the result.</div>;
  }

  return (
    <div>
      <h2>Analysis Result</h2>
      <p><strong>Classification:</strong> {result.isSpam ? 'Spam' : 'Non-Spam'}</p>
      <p><strong>Confidence Score:</strong> {result.confidence}%</p>
    </div>
  );
};

export default ResultCard;
