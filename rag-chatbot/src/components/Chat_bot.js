import React, { useState } from 'react';
import axios from 'axios';

function Chat_bot() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [method, setMethod] = useState('self_query');  // New state for method selection

  const handleMessageChange = (e) => {
    setMessage(e.target.value);
  };

  const handleMethodChange = (e) => {
    setMethod(e.target.value);
  };

  const handleChat = async () => {
    if (!message.trim()) {
      setError('Message cannot be empty.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const res = await axios.post('http://localhost:5000/chat', { 
        message,
        method  // Send the selected method to the backend
      });
      setResponse(res.data.response);
    } catch (error) {
      console.error('Error getting chat response:', error);
      setError('Failed to get response.');
      setResponse('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Chat with Bot</h2>
      <select value={method} onChange={handleMethodChange}>
        <option value="self_query">Self Query</option>
        <option value="query_expansion">Query Expansion</option>
      </select>
      <input type="text" value={message} onChange={handleMessageChange} />
      <button onClick={handleChat} disabled={loading}>
        {loading ? 'Sending...' : 'Send'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {response && <p>{response}</p>}
    </div>
  );
}

export default Chat_bot;
