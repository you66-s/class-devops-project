import React, { useState } from 'react';

const EmailInput = ({ onAnalyze }) => {
  const [emailText, setEmailText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (emailText.trim()) {
      onAnalyze(emailText);
      setEmailText('');
    }
  };

  return (
    <div>
      <h2>Email Input</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={emailText}
          onChange={(e) => setEmailText(e.target.value)}
          placeholder="Enter email content here..."
          rows={10}
          cols={50}
          required
        />
        <br />
        <button type="submit">Analyze</button>
      </form>
    </div>
  );
};

export default EmailInput;
