import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT  # Use recommended pre-trained weights
        model = mobilenet_v2(weights=weights)
        self.features = model.features
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        return feat1, feat2

    def extract_features(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        distance = torch.norm(feat1 - feat2, p=2, dim=1)  # Euclidean distance
        loss = label * torch.pow(distance, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()

class SignatureDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.npy')]
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return min(len(self.real_images), len(self.fake_images)) * 2  # Equal number of pos/neg pairs

    def __getitem__(self, index):
        if index % 2 == 0:
            img1_path, img2_path = random.sample(self.real_images, 2)
            label = 1  # Positive pair
        else:
            img1_path = random.choice(self.real_images)
            img2_path = random.choice(self.fake_images)
            label = 0  # Negative pair

        img1 = np.load(img1_path)
        img2 = np.load(img2_path)

        # Convert NumPy arrays to tensors and apply transformations
        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ------------------ 4️⃣ Training Function ------------------
def train_siamese_network(real_dir, fake_dir, num_epochs=10, batch_size=8, lr=0.001):
    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    # Create Dataset and DataLoader
    dataset = SignatureDataset(real_dir, fake_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            feat1, feat2 = model(img1, img2)
            loss = criterion(feat1, feat2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

    # Save the Model
    model_path = "siamese_signature_model.pth"
    torch.save(model.features.state_dict(), model_path)
    print(f"Model saved successfully at {model_path}!")

# ------------------ 5️⃣ Run Training ------------------
if __name__ == '__main__':
    real_dir = "signatures/rickyy"
    fake_dir = "signatures/ricky"
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("Error: One or both signature directories do not exist.")
    else:
        train_siamese_network(real_dir, fake_dir, num_epochs=10, batch_size=8, lr=0.001)
