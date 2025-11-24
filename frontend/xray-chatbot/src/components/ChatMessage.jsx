import React from 'react';

function ChatMessage({ message }) {
  const isUser = message.sender === 'user';
  
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      animation: 'fadeIn 0.3s ease-in'
    }}>
      <div style={{ 
        display: 'flex', 
        gap: '12px',
        maxWidth: '70%',
        flexDirection: isUser ? 'row-reverse' : 'row'
      }}>
        
        {/* Avatar */}
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          background: isUser 
            ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            : 'linear-gradient(135deg, #4b5563 0%, #1f2937 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
        }}>
          <svg style={{ width: '24px', height: '24px', color: 'white' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {isUser ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            )}
          </svg>
        </div>

        {/* Message Content */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{
            background: isUser 
              ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
              : 'white',
            color: isUser ? 'white' : '#1f2937',
            padding: '16px',
            borderRadius: '16px',
            boxShadow: isUser 
              ? '0 4px 15px rgba(102, 126, 234, 0.3)'
              : '0 2px 10px rgba(0,0,0,0.1)',
            border: isUser ? 'none' : '1px solid #e5e7eb'
          }}>
            {message.image && (
              <div style={{ marginBottom: '12px' }}>
                <img 
                  src={message.image} 
                  alt="X-ray" 
                  style={{
                    borderRadius: '12px',
                    maxWidth: '250px',
                    maxHeight: '250px',
                    width: '100%',
                    height: 'auto',
                    objectFit: 'contain',
                    boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
                  }}
                />
              </div>
            )}
            <p style={{ 
              fontSize: '14px', 
              lineHeight: '1.6', 
              margin: 0,
              whiteSpace: 'pre-wrap'
            }}>
              {message.text}
            </p>
          </div>
          
          {/* Timestamp */}
          <div style={{
            fontSize: '12px',
            color: '#9ca3af',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            paddingLeft: isUser ? '0' : '4px',
            paddingRight: isUser ? '4px' : '0'
          }}>
            <span>🕐</span>
            <span>{formatTime(message.timestamp)}</span>
          </div>
        </div>

      </div>
    </div>
  );
}

export default ChatMessage;