# %%
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# ✅ Configs
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Paths
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ Datasets and Dataloaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"🔍 Classes: {train_dataset.classes}")  # ['graph', 'non_graph']

# ✅ Model: MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(DEVICE)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ Training Loop
print(f"🚀 Starting training on {DEVICE}...\n")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"📘 Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}, Accuracy = {train_acc:.2f}%")

    # ✅ Save model after each epoch
    torch.save(model.state_dict(), f"mobilenetv2_epoch_{epoch+1}.pth")

# ✅ Final Evaluation
print("\n📊 Evaluating on validation set...")
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n🧾 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print raw confusion matrix
print("🧩 Confusion Matrix (raw values):")
print(cm)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# %%
# Inference
import os
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# === Configuration ===
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mobilenetv2_epoch_5.pth"  # or whichever epoch you saved
IMAGE_FOLDER = "inference_images"       # Folder containing new images

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Model ===
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)  # 2 classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Class Names (ensure order matches training)
class_names = ['graph', 'non_graph']

# === Predict on Images ===
results = []

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("⚠️ No valid images found in the folder.")
else:
    for img_name in tqdm(image_files, desc="🔍 Predicting"):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)
                pred_class = class_names[pred.item()]

            results.append({"image": img_name, "prediction": pred_class})
        except Exception as e:
            print(f"❌ Failed to process {img_name}: {e}")

    # === Convert to DataFrame
    df = pd.DataFrame(results)
    print("\n📄 Predictions:")
    print(df)

    # === Save to CSV (optional)
    df.to_csv("predictions.csv", index=False)
    print("\n✅ Results saved to predictions.csv")

# %%
