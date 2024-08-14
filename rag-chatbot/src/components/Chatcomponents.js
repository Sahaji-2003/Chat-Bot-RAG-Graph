import React, { useState } from 'react';

const ChatComponent = () => {
    const [userMessage, setUserMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [retrievalMethod, setRetrievalMethod] = useState('self-query');
    const [loading, setLoading] = useState(false);

    const handleInputChange = (e) => {
        setUserMessage(e.target.value);
    };

    const handleSendMessage = async () => {
        if (userMessage.trim() === '' || ['exit', 'quit'].includes(userMessage.toLowerCase())) {
            setChatHistory(prev => [...prev, { sender: 'Bot', message: 'Goodbye!' }]);
            setUserMessage('');
            return;
        }

        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ message: userMessage, retrieval_method: retrievalMethod }),
            });

            const data = await response.json();
            const botResponse = data.response || 'An error occurred while processing your request.';

            setChatHistory(prev => [...prev, { sender: 'You', message: userMessage }, { sender: 'Bot', message: botResponse }]);
        } catch (error) {
            setChatHistory(prev => [...prev, { sender: 'You', message: userMessage }, { sender: 'Bot', message: 'An error occurred while processing your request.' }]);
        } finally {
            setUserMessage('');
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    };

    const handleMethodChange = (method) => {
        setRetrievalMethod(method);
    };

    return (
        <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
            <h1>Chat with Bot</h1>
            <div style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', height: '300px', overflowY: 'scroll' }}>
                {chatHistory.map((chat, index) => (
                    <div key={index} style={{ marginBottom: '10px' }}>
                        <strong>{chat.sender}:</strong> <span>{chat.message}</span>
                    </div>
                ))}
                {loading && <div>Loading...</div>}
            </div>
            <div style={{ marginBottom: '10px' }}>
                <button onClick={() => handleMethodChange('self-query')} style={{ marginRight: '10px' }}>
                    Self-Query
                </button>
                <button onClick={() => handleMethodChange('query-expansion')}>
                    Query Expansion
                </button>
            </div>
            <input
                type="text"
                value={userMessage}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder="Type your message and press Enter..."
                style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
            />
            <button onClick={handleSendMessage} style={{ width: '100%', padding: '10px' }}>
                Send
            </button>
        </div>
    );
};

export default ChatComponent;


