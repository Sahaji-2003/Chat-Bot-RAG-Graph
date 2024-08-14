// // src/components/Chatbot.js
// import React, { useState } from 'react';
// import axios from 'axios';

// const Chatbot = () => {
//     const [messages, setMessages] = useState([]);
//     const [input, setInput] = useState('');

//     const sendMessage = async () => {
//         const response = await axios.post('http://localhost:5000/chat', { message: input });
//         setMessages([...messages, { text: input, user: true }, { text: response.data.response, user: false }]);
//         setInput('');
//     };

//     return (
//         <div>
//             <div>
//                 {messages.map((msg, index) => (
//                     <div key={index} className={msg.user ? 'user-message' : 'bot-message'}>
//                         {msg.text}
//                     </div>
//                 ))}
//             </div>
//             <input value={input} onChange={(e) => setInput(e.target.value)} />
//             <button onClick={sendMessage}>Send</button>
//         </div>
//     );
// };

// export default Chatbot;



// import React, { useState, useEffect } from 'react';
// import axios from 'axios';
// import './Chatbot.css';

// const Chatbot = () => {
//     const [messages, setMessages] = useState([]);
//     const [input, setInput] = useState('');
//     const [typingMessage, setTypingMessage] = useState(null);

//     const sendMessage = async () => {
//         if (input.trim() === '') return;
//         const userMessage = { text: input, user: true };
//         setMessages([...messages, userMessage]);
//         setInput('');

//         try {
//             const response = await axios.post('http://localhost:5000/chat', { message: input });
//             displayTypingEffect(response.data.response);
//         } catch (error) {
//             console.error("Error sending message:", error);
//         }
//     };

//     const displayTypingEffect = (text) => {
//         let index = 0;
//         const intervalId = setInterval(() => {
//             setTypingMessage(text.slice(0, index));
//             index++;
//             if (index > text.length) {
//                 clearInterval(intervalId);
//                 setMessages(prevMessages => [...prevMessages, { text, user: false }]);
//                 setTypingMessage(null);
//             }
//         }, 50); // Adjust the typing speed here
//     };

//     const formatMessage = (text) => {
//         const lines = text.split('\n').map((line, index) => {
//             const parts = line.split(/(\*\*.*?\*\*)/).map((part, i) =>
//                 part.startsWith('**') && part.endsWith('**') ? (
//                     <strong key={i} className="highlight">{part.slice(2, -2)}</strong>
//                 ) : part
//             );

//             if (line.startsWith('* ')) {
//                 return <li key={index}>{parts}</li>;
//             } else if (line.startsWith('## ')) {
//                 return <h2 key={index}>{parts}</h2>;
//             } else {
//                 return <p key={index}>{parts}</p>;
//             }
//         });

//         return <div>{lines}</div>;
//     };

//     return (
//         <div className="chatbot-container">
//             <div className="chatbox">
//                 {messages.map((msg, index) => (
//                     <div key={index} className={msg.user ? 'user-message message' : 'bot-message message'}>
//                         {formatMessage(msg.text)}
//                     </div>
//                 ))}
//                 {typingMessage && (
//                     <div className="bot-message message typing">
//                         {formatMessage(typingMessage)}
//                     </div>
//                 )}
//             </div>
//             <div className="input-container">
//                 <input
//                     value={input}
//                     onChange={(e) => setInput(e.target.value)}
//                     onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
//                     placeholder="Type a message..."
//                     className="input-box"
//                 />
//                 <button onClick={sendMessage} className="send-button">Send</button>
//             </div>
//         </div>
//     );
// };

// export default Chatbot;


// import React, { useState, useEffect } from 'react';
// import axios from 'axios';
// import './Chatbot.css';

// const Chatbot = () => {
//     const [messages, setMessages] = useState([]);
//     const [input, setInput] = useState('');
//     const [typingMessage, setTypingMessage] = useState(null);

//     const sendMessage = async () => {
//         if (input.trim() === '') return;
//         const userMessage = { text: input, user: true };
//         setMessages([...messages, userMessage]);
//         setInput('');

//         try {
//             const response = await axios.post('http://localhost:5000/chat', { message: input });
//             displayTypingEffect(response.data.response);
//         } catch (error) {
//             console.error("Error sending message:", error);
//         }
//     };

//     const displayTypingEffect = (text) => {
//         let index = 0;
//         const intervalId = setInterval(() => {
//             setTypingMessage(text.slice(0, index));
//             index++;
//             if (index > text.length) {
//                 clearInterval(intervalId);
//                 setMessages(prevMessages => [...prevMessages, { text, user: false }]);
//                 setTypingMessage(null);
//             }
//         }, 50); // Adjust the typing speed here
//     };

//     const formatMessage = (text) => {
//         const lines = text.split('\n').map((line, index) => {
//             const parts = line.split(/(\*\*.*?\*\*)/).map((part, i) =>
//                 part.startsWith('**') && part.endsWith('**') ? (
//                     <strong key={i} className="highlight">{part.slice(2, -2)}</strong>
//                 ) : part
//             );

//             if (line.startsWith('* ')) {
//                 return <li key={index}>{parts}</li>;
//             } else if (line.startsWith('## ')) {
//                 return <h2 key={index}>{parts}</h2>;
//             } else {
//                 return <p key={index}>{parts}</p>;
//             }
//         });

//         return <div>{lines}</div>;
//     };

//     return (
//         <div className="chatbot-container">
//             <div className="chatbox">
//                 {messages.map((msg, index) => (
//                     <div key={index} className={msg.user ? 'user-message message' : 'bot-message message'}>
//                         {formatMessage(msg.text)}
//                     </div>
//                 ))}
//                 {typingMessage && (
//                     <div className="bot-message message typing">
//                         {formatMessage(typingMessage)}
//                     </div>
//                 )}
//             </div>
//             <div className="input-container">
//                 <input
//                     value={input}
//                     onChange={(e) => setInput(e.target.value)}
//                     onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
//                     placeholder="Type a message..."
//                     className="input-box"
//                 />
//                 <button onClick={sendMessage} className="send-button">Send</button>
//             </div>
//         </div>
//     );
// };

// export default Chatbot;


import React, { useState } from 'react';
import axios from 'axios';
import './Chatbot.css';

const Chatbot = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [typingMessage, setTypingMessage] = useState(null);
    const [method, setMethod] = useState('self_query');

    const sendMessage = async () => {
        if (input.trim() === '') return;
        const userMessage = { text: input, user: true };
        setMessages([...messages, userMessage]);
        setInput('');
    
        try {
            const response = await axios.post('http://localhost:5000/chat', { 
                message: input,
                method: method // Pass the method choice to the backend
            });
    
            // Log the response to see if it's being received correctly
            console.log('Response from server:', response.data.response);
    
            // Display typing effect with the response
            displayTypingEffect(response.data.response);
        } catch (error) {
            console.error("Error sending message:", error);
        }
    };
    

    const displayTypingEffect = (text) => {
        let index = 0;
        const intervalId = setInterval(() => {
            setTypingMessage(text.slice(0, index));
            index++;
            if (index > text.length) {
                clearInterval(intervalId);
                setMessages(prevMessages => [...prevMessages, { text, user: false }]);
                setTypingMessage(null);
            }
        }, 50); // Adjust the typing speed here
    };

    const formatMessage = (text) => {
        const lines = text.split('\n').map((line, index) => {
            const parts = line.split(/(\*\*.*?\*\*)/).map((part, i) =>
                part.startsWith('**') && part.endsWith('**') ? (
                    <strong key={i} className="highlight">{part.slice(2, -2)}</strong>
                ) : part
            );

            if (line.startsWith('* ')) {
                return <li key={index}>{parts}</li>;
            } else if (line.startsWith('## ')) {
                return <h2 key={index}>{parts}</h2>;
            } else {
                return <p key={index}>{parts}</p>;
            }
        });

        return <div>{lines}</div>;
    };

    return (
        <div className="chatbot-container">
            <div className="chatbox">
                {messages.map((msg, index) => (
                        <div key={index} className={msg.user ? 'user-message message' : 'bot-message message'}>
                        {formatMessage(msg.text)}
                    </div>
                ))}
                {typingMessage && (
                    <div className="bot-message message typing">
                        {formatMessage(typingMessage)}
                    </div>
                )}
            </div>
            <div className="input-container">
                <select
                    value={method}
                    onChange={(e) => setMethod(e.target.value)}
                    className="method-select"
                >
                    <option value="self_query">Self-Query</option>
                    <option value="query_expansion">Query Expansion</option>
                </select>
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Type a message..."
                    className="input-box"
                />
                <button onClick={sendMessage} className="send-button">Send</button>
            </div>
        </div>
    );
};

export default Chatbot;
