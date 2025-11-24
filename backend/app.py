from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import pipeline, AutoModelForSeq2SeqLM
from PIL import Image
import io
import PyPDF2

app = Flask(__name__)
CORS(app)

# Load X-Ray Report Generation Model
print("Loading X-Ray report generation model...")
XRAY_MODEL_PATH = "xray_report_model"

try:
    xray_model = VisionEncoderDecoderModel.from_pretrained(XRAY_MODEL_PATH)
    feature_extractor = ViTImageProcessor.from_pretrained(XRAY_MODEL_PATH)
    xray_tokenizer = AutoTokenizer.from_pretrained(XRAY_MODEL_PATH)

    if xray_tokenizer.pad_token_id is None:
        xray_tokenizer.pad_token_id = xray_tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xray_model.to(device)
    xray_model.eval()
    print("X-Ray model loaded successfully.")
except Exception as e:
    print(f"Error loading X-Ray model: {e}")
    xray_model = None

# Load Summarization Model
print("Loading summarization model...")
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

try:
    summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    summary_model.to(device)
    
    summarizer = pipeline(
        "summarization",
        model=summary_model,
        tokenizer=summary_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("Summarization model loaded successfully.")
except Exception as e:
    print(f"Error loading summarization model: {e}")
    summarizer = None


@app.route("/generate_report", methods=["POST"])
def generate_report():
    if not xray_model:
        return jsonify({"error": "X-Ray model not loaded"}), 500
    
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    encodings = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encodings.pixel_values.to(device)
    attention_mask = torch.ones(pixel_values.shape[:-1], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = xray_model.generate(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            pad_token_id=xray_tokenizer.pad_token_id,
            early_stopping=True
        )

    report = xray_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"report": report})


def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None


def clean_medical_text(text):
    text = " ".join(text.split())
    text = text.replace("XXXX", "[redacted]")
    return text


@app.route('/summarize_report', methods=['POST'])
def summarize_report():
    if not summarizer:
        return jsonify({"error": "Summarization model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
            if not text:
                return jsonify({"error": "Could not extract text from PDF"}), 400
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return jsonify({"error": "Unsupported format. Upload PDF or TXT"}), 400
        
        if not text or len(text.strip()) < 10:
            return jsonify({"error": "File is empty or too short"}), 400
        
        text = clean_medical_text(text)
        words = text.split()
        
        if len(words) > 1024:
            text = " ".join(words[:1024])
        
        if len(words) < 30:
            summary = f"Summary: {text}"
        else:
            result = summarizer(
                text,
                max_length=150,
                min_length=50,
                do_sample=False,
                truncation=True
            )
            summary = result[0]['summary_text']
        
        return jsonify({
            "summary": summary,
            "word_count": len(words),
            "compressed_from": f"{len(words)} words to {len(summary.split())} words"
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "xray_model": xray_model is not None,
        "summarizer": summarizer is not None,
        "device": str(device)
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Medical AI Backend Running")
    print("="*60)
    print(f"Device: {device}")
    print(f"X-Ray Model: {'Loaded' if xray_model else 'Not loaded'}")
    print(f"Summarizer: {'Loaded' if summarizer else 'Not loaded'}")
    print("="*60 + "\n")
    
    app.run(host="127.0.0.1", port=5000, debug=True)