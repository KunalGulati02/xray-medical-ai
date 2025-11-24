# Frontend – X-Ray Medical AI Assistant

This is the frontend interface for the **X-Ray Medical AI Assistant**, built using **React + Vite + TailwindCSS**.  
It provides a clean, modern interface for:
- Uploading X‑ray images  
- Viewing AI‑generated medical summaries  
- Uploading medical reports (TXT/PDF)  
- Displaying chat‑style responses  

---

## 🚀 Tech Stack
- **React**
- **Vite**
- **TailwindCSS**
- **JavaScript**
- **Axios** (API communication)

---

## 📦 How to Run the Frontend

### 1. Install dependencies
```bash
npm install
```

### 2. Start development server
```bash
npm run dev
```

Your frontend will be available at:
```
http://localhost:5173
```

---

## 📂 Frontend Structure
```
frontend/
│── index.html
│── package.json
│── package-lock.json
│── vite.config.js
│── tailwind.config.js
│── postcss.config.js
│
├── public/
│   └── (static assets)
│
└── src/
    ├── App.jsx
    ├── main.jsx
    ├── components/
    │     ├── ChatMessage.jsx
    │     ├── UploadBox.jsx
    ├── index.css
```

---

## 🔗 API Configuration

The frontend expects the backend at:
```
http://localhost:5000
```

If different, update:
```
src/App.jsx
src/components/UploadBox.jsx
```

---

## 🎨 UI Features
- Dual‑mode interface: **X‑Ray** & **Report**
- Drag‑and‑drop upload boxes
- Realtime chat UI
- Smooth transitions & clean design with Tailwind

---

## 🧪 Production Build
```bash
npm run build
```

Output appears in `/dist`.

---

## ⭐ Author
Frontend developed by **Kunal Gulati**.

