import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdb

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = mobilenet_v2(pretrained=True)
        self.features = model.features

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
    
class SignatureAuth():
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.eval()

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to MobileNetV2 input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
        ])

    def extract_features(self, img):
        img = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(img)
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