
import os
import pandas as pd
from PIL import Image
import torch
import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# Creating class for dataset
class ImageTextDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Label and Image paths
csv_file = r"C:\Users\utkar\OneDrive\Desktop\dataset and csv for postive and negative labels\CLIP\csv for bates_CLIP.csv"
img_dir = r"C:\Users\utkar\Downloads\Utkarsh Gupta\NUIG\Semester 2\DA Project\Extracted Dataset updated"

# Defining transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Loading Bates dataset
dataset = ImageTextDataset(csv_file=csv_file, img_dir=img_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loading CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_clip = clip.load("ViT-B/32", device=device)

# Defining Cross Entropy Loss and Adam optimizer
loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop for fine tuning CLIP
for epoch in range(10):
    for images, labels in dataloader:
        images = images.to(device)
        labels = clip.tokenize(labels).to(device)

        optimizer.zero_grad()

        logits_per_image, logits_per_text = model(images, labels)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} / 10, Loss: {total_loss.item()}")

print("Training complete.")

# Saving the fine tuned model
torch.save(model.state_dict(), "clip_model.pth")
