import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import UploadBox from './components/UploadBox';
import ChatMessage from './components/ChatMessage';

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('xray');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleUpload = async (file) => {
    const isImage = file.type.startsWith('image/');
    const isPDF = file.type === 'application/pdf';
    const isText = file.type === 'text/plain';

    if (mode === 'xray' && !isImage) {
      alert('Please upload an image file for X-ray analysis');
      return;
    }

    if (mode === 'report' && !(isPDF || isText)) {
      alert('Please upload a PDF or TXT file for report summarization');
      return;
    }

    const userMsg = { 
      sender: 'user', 
      text: `Uploaded: ${file.name}`,
      image: isImage ? URL.createObjectURL(file) : null,
      fileType: mode === 'xray' ? 'image' : 'document',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const formData = new FormData();
      
      if (mode === 'xray') {
        formData.append('image', file);
        const res = await axios.post('http://127.0.0.1:5000/generate_report', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        
        const botMsg = { 
          sender: 'bot', 
          text: res.data.report,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMsg]);
      } else {
        formData.append('file', file);
        const res = await axios.post('http://127.0.0.1:5000/summarize_report', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        
        const botMsg = { 
          sender: 'bot', 
          text: res.data.summary,
          metadata: `Compressed from ${res.data.word_count} words`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMsg]);
      }
    } catch (error) {
      const errMsg = { 
        sender: 'bot', 
        text: '⚠️ Error processing file. Please ensure the backend is running and try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  const handleModeChange = (newMode) => {
    if (messages.length > 0) {
      const confirm = window.confirm('Switching modes will clear the chat. Continue?');
      if (!confirm) return;
    }
    setMode(newMode);
    setMessages([]);
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      <header style={{ 
        background: 'rgba(255, 255, 255, 0.95)', 
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(0,0,0,0.1)',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
      }}>
        <div style={{ 
          maxWidth: '1400px', 
          margin: '0 auto', 
          padding: '20px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              padding: '12px',
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)'
            }}>
              <svg style={{ width: '32px', height: '32px', color: 'white' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h1 style={{ 
                fontSize: '28px', 
                fontWeight: '700', 
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                margin: 0
              }}>
                Medical AI Assistant
              </h1>
              <p style={{ fontSize: '14px', color: '#666', margin: 0 }}>
                X-Ray Analysis & Report Summarization
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '8px', background: '#f3f4f6', padding: '4px', borderRadius: '12px' }}>
              <button
                onClick={() => handleModeChange('xray')}
                style={{
                  padding: '10px 20px',
                  background: mode === 'xray' ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'transparent',
                  color: mode === 'xray' ? 'white' : '#666',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.3s'
                }}
              >
                📷 X-Ray
              </button>
              <button
                onClick={() => handleModeChange('report')}
                style={{
                  padding: '10px 20px',
                  background: mode === 'report' ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'transparent',
                  color: mode === 'report' ? 'white' : '#666',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.3s'
                }}
              >
                📄 Report
              </button>
            </div>

            {messages.length > 0 && (
              <button
                onClick={handleClearChat}
                style={{
                  padding: '10px 20px',
                  background: '#ef4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.3s'
                }}
                onMouseOver={(e) => e.target.style.background = '#dc2626'}
                onMouseOut={(e) => e.target.style.background = '#ef4444'}
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </header>

      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px' }}>
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          
          <aside style={{ flex: '0 0 320px', minWidth: '320px' }}>
            <div style={{ 
              background: 'white', 
              borderRadius: '16px', 
              padding: '24px',
              marginBottom: '16px',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
            }}>
              <h2 style={{ 
                fontSize: '18px', 
                fontWeight: '700', 
                marginBottom: '16px',
                color: '#1f2937'
              }}>
                {mode === 'xray' ? '📷 X-Ray Analysis' : '📄 Report Summary'}
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {mode === 'xray' ? (
                  <>
                    <Step number="1" text="Upload X-ray image (PNG, JPG)" />
                    <Step number="2" text="AI analyzes the image" />
                    <Step number="3" text="Receive medical report" />
                  </>
                ) : (
                  <>
                    <Step number="1" text="Upload report (PDF or TXT)" />
                    <Step number="2" text="AI reads the document" />
                    <Step number="3" text="Receive concise summary" />
                  </>
                )}
              </div>
            </div>

            <div style={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '16px', 
              padding: '24px',
              marginBottom: '16px',
              color: 'white',
              boxShadow: '0 4px 20px rgba(102, 126, 234, 0.3)'
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '12px' }}>
                Statistics
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Processed:</span>
                  <span style={{ fontWeight: '700' }}>
                    {messages.filter(m => m.sender === 'bot' && !m.text.includes('Error')).length}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Uploaded:</span>
                  <span style={{ fontWeight: '700' }}>
                    {messages.filter(m => m.sender === 'user').length}
                  </span>
                </div>
              </div>
            </div>

            <div style={{ 
              background: '#fef3c7',
              border: '2px solid #fbbf24',
              borderRadius: '16px', 
              padding: '16px'
            }}>
              <div style={{ display: 'flex', gap: '12px' }}>
                <span style={{ fontSize: '20px' }}>⚠️</span>
                <div>
                  <p style={{ fontSize: '14px', fontWeight: '700', color: '#78350f', margin: '0 0 4px 0' }}>
                    Medical Disclaimer
                  </p>
                  <p style={{ fontSize: '12px', color: '#92400e', margin: 0 }}>
                    This is an AI tool. Always consult healthcare professionals for diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </aside>

          <div style={{ flex: '1', minWidth: '400px' }}>
            <div style={{ 
              background: 'white', 
              borderRadius: '16px',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden'
            }}>
              
              <div style={{ 
                height: '500px',
                overflowY: 'auto', 
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px'
              }}>
                {messages.length === 0 ? (
                  <EmptyState mode={mode} />
                ) : (
                  <>
                    {messages.map((msg, i) => (
                      <ChatMessage key={i} message={msg} />
                    ))}
                    {loading && <LoadingIndicator mode={mode} />}
                    <div ref={messagesEndRef} />
                  </>
                )}
              </div>

              <div style={{ 
                borderTop: '1px solid #e5e7eb',
                padding: '20px',
                background: '#f9fafb'
              }}>
                <UploadBox onUpload={handleUpload} loading={loading} mode={mode} />
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

function Step({ number, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
      <span style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        width: '24px',
        height: '24px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '12px',
        fontWeight: '700',
        flexShrink: 0
      }}>
        {number}
      </span>
      <span style={{ fontSize: '14px', color: '#4b5563' }}>{text}</span>
    </div>
  );
}

function EmptyState({ mode }) {
  return (
    <div style={{ 
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center',
      padding: '24px'
    }}>
      <div style={{ maxWidth: '400px' }}>
        <div style={{
          width: '80px',
          height: '80px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 20px',
          boxShadow: '0 10px 30px rgba(102, 126, 234, 0.3)'
        }}>
          <svg style={{ width: '40px', height: '40px', color: 'white' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {mode === 'xray' ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            )}
          </svg>
        </div>
        <h3 style={{ fontSize: '24px', fontWeight: '700', color: '#1f2937', marginBottom: '12px' }}>
          {mode === 'xray' ? 'X-Ray Analysis' : 'Report Summarization'}
        </h3>
        <p style={{ fontSize: '16px', color: '#6b7280', marginBottom: '24px' }}>
          {mode === 'xray' 
            ? 'Upload an X-ray image to get AI-powered medical analysis'
            : 'Upload a medical report to get an AI-generated summary'
          }
        </p>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '12px 24px',
          borderRadius: '30px',
          fontSize: '14px',
          fontWeight: '600'
        }}>
          <span>⚡</span>
          <span>Powered by AI</span>
        </div>
      </div>
    </div>
  );
}

function LoadingIndicator({ mode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
      <div style={{
        background: '#f3f4f6',
        borderRadius: '20px',
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', gap: '4px' }}>
          <div style={{
            width: '8px',
            height: '8px',
            background: '#667eea',
            borderRadius: '50%',
            animation: 'bounce 1.4s infinite ease-in-out both'
          }} />
          <div style={{
            width: '8px',
            height: '8px',
            background: '#667eea',
            borderRadius: '50%',
            animation: 'bounce 1.4s infinite ease-in-out both 0.16s'
          }} />
          <div style={{
            width: '8px',
            height: '8px',
            background: '#667eea',
            borderRadius: '50%',
            animation: 'bounce 1.4s infinite ease-in-out both 0.32s'
          }} />
        </div>
        <span style={{ fontSize: '14px', color: '#6b7280' }}>
          {mode === 'xray' ? 'Analyzing image...' : 'Summarizing report...'}
        </span>
      </div>
    </div>
  );
}

export default App;