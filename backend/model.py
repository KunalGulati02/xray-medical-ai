import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Path where your fine-tuned model is saved
SAVE_PATH = "xray_report_model"

# Load model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained(SAVE_PATH)
feature_extractor = ViTImageProcessor.from_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)

# Set pad_token_id if missing (important for generation)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load an X-ray image
image_path = r"C:\Users\Kunal Gulati\OneDrive\Desktop\minor project\backend\output_images\test\1_IM-0001-3001.dcm.png"
image = Image.open(image_path).convert("RGB")

# Preprocess image -> pixel values and attention mask
encodings = feature_extractor(images=image, return_tensors="pt")
pixel_values = encodings.pixel_values.to(device)
# Vision models don't create an attention_mask by default, so we create a dummy one of all 1s
attention_mask = torch.ones(pixel_values.shape[:-1], dtype=torch.long).to(device)

# Generate report
outputs = model.generate(
    pixel_values=pixel_values,
    attention_mask=attention_mask,
    max_length=128,
    num_beams=4,
    pad_token_id=tokenizer.pad_token_id,
    early_stopping=True
)

# Decode the generated ids into text
report = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated report:", report)
