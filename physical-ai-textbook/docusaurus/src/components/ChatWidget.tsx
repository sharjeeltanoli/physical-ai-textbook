import React, { useState, useEffect, useRef } from 'react';
import './chat.css'; // Import the CSS file

interface Message {
  text: string;
  sender: 'user' | 'bot';
}

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatBoxRef = useRef<HTMLDivElement>(null); // Ref for the chat box itself

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleSendMessage = async () => {
    const userMessage = inputValue.trim();
    if (userMessage) {
      // Add user message to the chat
      setMessages((prevMessages) => [...prevMessages, { text: userMessage, sender: 'user' }]);
      setInputValue('');
      setIsLoading(true);

      // Show typing indicator
      setTimeout(() => {
        setMessages((prevMessages) => [...prevMessages, { text: '...', sender: 'bot' }]);
      }, 500);


      try {
        const response = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query: userMessage }),
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Remove typing indicator and add bot response
        setMessages((prevMessages) => {
            const newMessages = prevMessages.filter(msg => msg.text !== '...');
            return [...newMessages, { text: data.answer, sender: 'bot' }];
        });

      } catch (error) {
        console.error('Error fetching chat response:', error);
        // Remove typing indicator and show error message
        setMessages((prevMessages) => {
            const newMessages = prevMessages.filter(msg => msg.text !== '...');
            return [...newMessages, { text: 'Sorry, something went wrong. Please try again.', sender: 'bot' }];
        });
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // Prevent new line in input
      handleSendMessage();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Effect to close chatbox when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (chatBoxRef.current && !chatBoxRef.current.contains(event.target as Node)) {
        // Check if the click was not on the chat toggle button itself
        const chatButton = document.getElementById('chat-toggle-button');
        if (chatButton && chatButton.contains(event.target as Node)) {
            return; // Don't close if clicking the toggle button
        }
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    } else {
      document.removeEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <>
      <button id="chat-toggle-button" className="chat-toggle-button" onClick={toggleChat}>
        {isOpen ? '✕' : '💬'}
      </button>

      {isOpen && (
        <div ref={chatBoxRef} className="chat-box">
          <div className="chat-header">
            <h3>AI Assistant</h3>
            <button className="chat-close-button" onClick={toggleChat}>✕</button>
          </div>
          <div className="chat-messages">
            {messages.map((message, index) => (
              <div key={index} className={`chat-bubble chat-${message.sender}`}>
                {message.text}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-area">
            <input
              type="text"
              placeholder="Type your message..."
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
            />
            <button onClick={handleSendMessage}>Send</button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatWidget;
