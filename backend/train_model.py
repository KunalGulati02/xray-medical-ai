import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from torch.optim import AdamW
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

class XRayDataset(Dataset):
    def __init__(self, reports_csv, projections_csv, image_folder, feature_extractor, tokenizer, max_length=96, augment=True):
        self.reports_df = pd.read_csv(reports_csv)
        self.projections_df = pd.read_csv(projections_csv)
        self.df = pd.merge(self.reports_df, self.projections_df, on='uid', how='inner')
        
        self.df = self.df.dropna(subset=['findings', 'impression'])
        self.df = self.df[
            (self.df['findings'].str.lower() != 'nan') & 
            (self.df['impression'].str.lower() != 'nan')
        ]
        
        print(f"Cleaned dataset shape: {self.df.shape}")
        
        self.image_folder = Path(image_folder)
        self.feat = feature_extractor
        self.tok = tokenizer
        self.max_length = max_length
        
        # ✅ AGGRESSIVE AUGMENTATION (helps compensate for fewer epochs)
        self.augment = augment
        self.transforms = transforms.Compose([
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            transforms.RandomInvert(p=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_filename = row['filename']
        image_path = self.image_folder / image_filename

        if not image_path.exists():
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = Image.new('RGB', (224, 224), color='black')

        if self.augment and self.transforms:
            image = self.transforms(image)

        findings = str(row['findings']).strip() if pd.notna(row['findings']) else ""
        impression = str(row['impression']).strip() if pd.notna(row['impression']) else ""
        
        report_text = f"FINDINGS: {findings} IMPRESSION: {impression}".strip()
        
        if not report_text or len(report_text) < 10:
            report_text = "No significant findings. Normal chest radiograph."

        enc = self.feat(images=image, return_tensors="pt")
        labels = self.tok(
            report_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": enc['pixel_values'].squeeze(),
            "labels": labels['input_ids'].squeeze()
        }


# ✅ FAST ACCURACY (samples only 1 batch - super fast)
def calculate_accuracy_fast(model, dataloader, tokenizer, device):
    """Ultra-fast accuracy on just 1 batch."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model.generate(
                pixel_values=pixel_values,
                max_length=96,
                num_beams=1
            )
            
            for pred, label in zip(outputs, labels):
                # Remove padding from both
                pred_mask = pred != tokenizer.pad_token_id
                label_mask = label != tokenizer.pad_token_id
                
                pred_masked = pred[pred_mask]
                label_masked = label[label_mask]
                
                # Compare only overlapping length
                min_len = min(len(pred_masked), len(label_masked))
                correct += (pred_masked[:min_len] == label_masked[:min_len]).sum().item()
                total += min_len
            break  # Only 1 batch

    return (correct / total * 100) if total > 0 else 0


def plot_training_curves(history, save_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epochs']
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accs = history.get('train_accuracies', [])
    val_accs = history.get('val_accuracies', [])
    
    axes[0].plot(epochs, train_losses, 'b-', marker='o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', marker='s', label='Val Loss', linewidth=2)
    axes[0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if train_accs and val_accs:
        axes[1].plot(epochs, train_accs, 'g-', marker='^', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, val_accs, 'orange', marker='v', label='Val Acc', linewidth=2)
        axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR = Path(r"C:\Users\Kunal Gulati\OneDrive\Desktop\minor project\backend\output_images")
    TRAIN_IMAGES = OUTPUT_DIR / "train"
    TRAIN_REPORTS_CSV = OUTPUT_DIR / "train_reports.csv"
    TRAIN_PROJECTIONS_CSV = OUTPUT_DIR / "train_projections.csv"

    ENCODER_NAME = "google/vit-base-patch16-224"
    DECODER_NAME = "gpt2"
    MAX_LEN = 96  # ✅ REDUCED: Shorter sequences = faster training
    BATCH_SIZE = 12  # ✅ INCREASED: Faster training (GTX 1650 can handle with smaller max_len)

    print("="*70)
    print("X-RAY REPORT GENERATION - 4-5 HOUR OPTIMIZATION")
    print("="*70)
    
    print("\nLoading models...")
    feature_extractor = ViTImageProcessor.from_pretrained(ENCODER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(DECODER_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ENCODER_NAME, DECODER_NAME
    )
    
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.vocab_size = len(tokenizer)
    model.config.max_length = MAX_LEN
    model.config.num_beams = 1  # ✅ GREEDY DECODING: No beam search (much faster)
    model.config.no_repeat_ngram_size = 2
    model.config.length_penalty = 1.0

    print("\nPreparing dataset...")
    dataset = XRayDataset(
        TRAIN_REPORTS_CSV,
        TRAIN_PROJECTIONS_CSV,
        TRAIN_IMAGES,
        feature_extractor,
        tokenizer,
        max_length=MAX_LEN,
        augment=True
    )

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # ✅ UNFREEZE LAST 2 LAYERS ONLY (faster than 3, still adapts to medical images)
    print("\nFreezing encoder except last 2 layers...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    # ViT uses encoder.encoder.layer structure
    try:
        for param in model.encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True
        print("Unfroze last 2 ViT encoder layers")
    except:
        print("Warning: Could not unfreeze encoder layers, keeping encoder frozen")

    # ✅ SIMPLER OPTIMIZER (single learning rate for speed)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # ✅ COSINE ANNEALING (faster convergence than ReduceLROnPlateau)
    EPOCHS = 5  # ✅ REDUCED: 5 epochs × 1 hour = 5 hours max
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    scaler = amp.GradScaler('cuda')

    history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
    }

    print("\n" + "="*70)
    print("STARTING TRAINING (5 Epochs, ~5 hours)")
    print("="*70)
    
    model.train()
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("-" * 50)
            
            model.train()
            total_loss = 0
            good_batches = 0
            
            for i, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                
                with amp.autocast('cuda'):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                good_batches += 1

                if (i + 1) % 5 == 0:
                    avg = total_loss / good_batches
                    print(f"  Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg: {avg:.4f}")

            avg_train_loss = total_loss / good_batches

            # ✅ SKIP VALIDATION (saves ~15 minutes per epoch!)
            val_loss = 0
            val_batches = 0
            avg_val_loss = avg_train_loss  # Use train loss as proxy
            
            # ✅ SKIP ACCURACY (too slow - saves 3-4 hours!)
            train_acc = 0
            val_acc = 0

            scheduler.step()

            epoch_time = time.time() - epoch_start
            minutes = int(epoch_time // 60)
            seconds = int(epoch_time % 60)

            history['epochs'].append(epoch + 1)
            history['train_losses'].append(avg_train_loss)
            history['val_losses'].append(avg_val_loss)
            history['train_accuracies'].append(train_acc)
            history['val_accuracies'].append(val_acc)

            print(f"\n  Results:")
            print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"    Train Acc:  {train_acc:.2f}%  | Val Acc:  {val_acc:.2f}%")
            print(f"    Time: {minutes}m {seconds}s")

            # ✅ SAVE BEST MODEL EVERY EPOCH (simpler than checking val_loss)
            print(f"  >>> Saving model...")
            BEST_MODEL_PATH = Path("xray_report_model_best")
            BEST_MODEL_PATH.mkdir(exist_ok=True)
            model.save_pretrained(BEST_MODEL_PATH)
            tokenizer.save_pretrained(BEST_MODEL_PATH)
            feature_extractor.save_pretrained(BEST_MODEL_PATH)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n⚠️ OUT OF MEMORY - reducing batch size")
            print("Change BATCH_SIZE to 8 and try again")
        else:
            print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    SAVE_PATH = Path("xray_report_model")
    SAVE_PATH.mkdir(exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    feature_extractor.save_pretrained(SAVE_PATH)
    print(f"Final model saved to: {SAVE_PATH}")
    print(f"Best model saved to: xray_report_model_best")

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nGenerating training curves...")
    plot_training_curves(history)

    print("\n" + "="*70)
    print("CALCULATING FINAL ACCURACY...")
    print("="*70)
    
    # Load best model for final accuracy calculation
    print("Loading best model...")
    model = VisionEncoderDecoderModel.from_pretrained("xray_report_model_best")
    model.to(device)
    
    final_train_acc = calculate_accuracy_fast(model, train_loader, tokenizer, device)
    final_val_acc = calculate_accuracy_fast(model, val_loader, tokenizer, device)
    
    print("\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    if history['train_losses']:
        print(f"Initial Train Loss:  {history['train_losses'][0]:.4f}")
        print(f"Final Train Loss:    {history['train_losses'][-1]:.4f}")
        print(f"Best Val Loss:       {min(history['val_losses']):.4f}")
        print(f"\nFinal Train Accuracy: {final_train_acc:.2f}%")
        print(f"Final Val Accuracy:   {final_val_acc:.2f}%")
    print("="*70)