import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import pdb

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
    def __init__(self, base_dir, transform=None):
        # self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.npy')]
        # self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.npy')]
        self.base_dir = base_dir
        self.users = os.listdir(os.path.join(base_dir, "real"))

        self.real_images = {user: self.get_image_list("real", user) for user in self.users}
        self.fake_images = {user: self.get_image_list("fake", user) for user in self.users}

        self.pairs = self.generate_pairs()
        self.transform = transform

    def get_image_list(self, category, user):
        """ Returns a list of .npy file paths for a given category (real/fake) and user. """
        user_path = os.path.join(self.base_dir, category, user)
        return [os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith(".npy")]

    def generate_pairs(self):
        """ Generates positive and negative pairs based on the dataset structure. """
        print("Generating pos/neg pairs")
        pairs = []

        for user in self.users:
            real_images = self.real_images[user]
            fake_images = self.fake_images[user]

            # Positive pairs: (real/user, real/user)
            if len(real_images) > 1:
                pairs.extend([(random.choice(real_images), random.choice(real_images), 1) for _ in range(10)])

            # Negative pairs: (real/user, fake/user)
            if real_images and fake_images:
                pairs.extend([(random.choice(real_images), random.choice(fake_images), 0) for _ in range(10)])

            # Negative pairs: (real/daniel, real/ricky)
            for other_user in self.users:
                if user != other_user:
                    other_real_images = self.real_images[other_user]
                    if real_images and other_real_images:
                        pairs.extend([(random.choice(real_images), random.choice(other_real_images), 0) for _ in range(10)])

        return pairs


    def __len__(self):
        # return min(len(self.real_images), len(self.fake_images)) * 2  # Equal number of pos/neg pairs
        return len(self.pairs)

    def __getitem__(self, index):
        img1_path, img2_path, label = self.pairs[index]

        img1 = np.load(img1_path)
        img2 = np.load(img2_path)

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def train_siamese_network(base_dir, num_epochs=10, batch_size=8, lr=0.001):
    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    # Create Dataset and DataLoader
    dataset = SignatureDataset(base_dir, transform=transform)
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
        dataset.generate_pairs()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Save the Model
    model_path = "siamese_signature_model.pth"
    torch.save(model.features.state_dict(), model_path)
    print(f"Model saved successfully at {model_path}!")

def get_average_embedding(base_dir, proto_dir, model):
    """ Computes the average feature embedding for all `.npy` files in a directory. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    users = os.listdir(os.path.join(base_dir, "real"))
    for user in users:
        real_dir = os.path.join(base_dir, "real", user)
        files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.npy')]
    
        if not files:
            print("No .npy files found!")
            return None

        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for file in files:
                img = np.load(file)
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)  # (C, H, W)
                embedding = model.extract_features(img_tensor)
                embeddings.append(embedding.cpu().numpy())

        avg_embedding = np.mean(embeddings, axis=0)
        np.save(f"{os.path.join(proto_dir, user)}.npy", avg_embedding)

# ------------------ 5️⃣ Run Training ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument("-b", "--base_dir", type=str, default="signatures/train", help="source directory")
    parser.add_argument("-p", "--proto_dir", type=str, default="signatures/prototypes", help="prototype directory")
    parser.add_argument("--ckpt", type=str, default="siamese_signature_model.pth", help="checkpoint")
    args = parser.parse_args()
    
    train_siamese_network(args.base_dir, num_epochs=30, batch_size=8, lr=0.001)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SiameseNetwork().to(device)
    # get_average_embedding(args.base_dir, args.proto_dir, model)

