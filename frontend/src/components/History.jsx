import React from 'react';

const History = ({ history }) => {
  return (
    <div>
      <h2>Prediction History</h2>
      {history.length === 0 ? (
        <p>No predictions yet.</p>
      ) : (
        <table border="1">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Email (truncated)</th>
              <th>Classification</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item) => (
              <tr key={item.id}>
                <td>{new Date(item.timestamp).toLocaleString()}</td>
                <td>{item.email.length > 50 ? item.email.substring(0, 50) + '...' : item.email}</td>
                <td>{item.isSpam ? 'Spam' : 'Non-Spam'}</td>
                <td>{item.confidence}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default History;
