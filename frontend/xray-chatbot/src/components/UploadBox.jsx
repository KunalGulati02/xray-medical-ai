import React, { useRef, useState } from 'react';

function UploadBox({ onUpload, loading, mode }) {
  const fileInput = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const acceptedTypes = mode === 'xray' ? 'image/*' : '.pdf,.txt';
  const fileTypeText = mode === 'xray' ? 'PNG, JPG, JPEG' : 'PDF, TXT';

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept={acceptedTypes}
        ref={fileInput}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        disabled={loading}
      />
      
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !loading && fileInput.current.click()}
        style={{
          border: dragActive ? '3px dashed #667eea' : '3px dashed #d1d5db',
          borderRadius: '16px',
          padding: '20px',
          textAlign: 'center',
          background: dragActive ? '#f0f4ff' : 'white',
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'all 0.3s',
          opacity: loading ? 0.6 : 1
        }}
        onMouseOver={(e) => {
          if (!loading && !dragActive) {
            e.currentTarget.style.borderColor = '#9ca3af';
            e.currentTarget.style.background = '#f9fafb';
          }
        }}
        onMouseOut={(e) => {
          if (!dragActive) {
            e.currentTarget.style.borderColor = '#d1d5db';
            e.currentTarget.style.background = 'white';
          }
        }}
      >
        <div style={{
          width: '50px',
          height: '50px',
          margin: '0 auto 12px',
          borderRadius: '50%',
          background: dragActive 
            ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            : '#f3f4f6',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <svg 
            style={{ 
              width: '25px', 
              height: '25px', 
              color: dragActive ? 'white' : '#9ca3af'
            }}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
            />
          </svg>
        </div>
        
        <p style={{ 
          fontSize: '14px', 
          fontWeight: '600', 
          color: '#1f2937',
          marginBottom: '4px'
        }}>
          {dragActive ? 'Drop your file here' : 'Click to upload or drag and drop'}
        </p>
        <p style={{ 
          fontSize: '12px', 
          color: '#6b7280',
          marginBottom: '16px'
        }}>
          {fileTypeText} up to 10MB
        </p>

        <button
          type="button"
          disabled={loading}
          onClick={(e) => {
            e.stopPropagation();
            if (!loading) fileInput.current.click();
          }}
          style={{
            padding: '10px 24px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '12px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: loading ? 'not-allowed' : 'pointer',
            boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)',
            transition: 'all 0.3s',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px'
          }}
          onMouseOver={(e) => {
            if (!loading) {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.4)';
            }
          }}
          onMouseOut={(e) => {
            e.target.style.transform = 'translateY(0)';
            e.target.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.3)';
          }}
        >
          <svg style={{ width: '18px', height: '18px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          <span>{loading ? 'Processing...' : mode === 'xray' ? 'Upload X-Ray' : 'Upload Report'}</span>
        </button>
      </div>
    </div>
  );
}

export default UploadBox;