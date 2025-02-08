import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdb

class Encoder(nn.Module):
    def __init__(self, ckpt_path=None):
        super(Encoder, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        self.features = model.features
        if ckpt_path is not None:
            print(f"Loading custom weights at {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.features.load_state_dict(state_dict)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
    
class SignatureAuth():
    def __init__(self, ckpt_path=None):
        self.encoder = Encoder(ckpt_path)
        self.encoder.eval()

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to MobileNetV2 input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
        ])

    def extract_features(self, img):
        img = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(img)
        return features.numpy()

    def compare_images(self, img1_path, img2_path):
        img1 = transforms.ToTensor()(Image.open(img1_path).convert('RGB'))
        img2 = transforms.ToTensor()(Image.open(img2_path).convert('RGB'))
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        similarity = cosine_similarity(features1, features2)[0][0]
        # print(f"Cosine Similarity: {similarity:.4f}")
        return similarity
    def compare_npy(self, npy1_path, npy2_path):
        img1 = np.load(npy1_path)
        img2 = np.load(npy2_path)
        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        similarity = cosine_similarity(features1, features2)[0][0]
        # print(f"Cosine Similarity: {similarity:.4f}")
        return similarity