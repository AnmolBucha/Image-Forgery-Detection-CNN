import os
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.model_loader import ModelLoader
from config import Config


class ImageProcessor:
    def __init__(self):
        self.patch_size = Config.PATCH_SIZE
        self.feature_dim = Config.FEATURE_DIM
        self.num_patches = Config.NUM_PATCHES
        self.transform = transforms.Compose([transforms.ToTensor()])

    def extract_patches(self, image, num_patches=100, patch_size=64):
        h, w = image.shape[:2]
        patches = []

        if h < patch_size or w < patch_size:
            image = cv2.resize(image, (max(patch_size, w), max(patch_size, h)))
            h, w = image.shape[:2]

        stride_h = max(1, (h - patch_size) // int(np.sqrt(num_patches)))
        stride_w = max(1, (w - patch_size) // int(np.sqrt(num_patches)))

        for i in range(0, h - patch_size + 1, stride_h):
            for j in range(0, w - patch_size + 1, stride_w):
                patch = image[i : i + patch_size, j : j + patch_size]
                patches.append(patch)
                if len(patches) >= num_patches:
                    break
            if len(patches) >= num_patches:
                break

        while len(patches) < num_patches:
            i = np.random.randint(0, h - patch_size)
            j = np.random.randint(0, w - patch_size)
            patch = image[i : i + patch_size, j : j + patch_size]
            patches.append(patch)

        return patches[:num_patches]

    def get_patch_features(self, model, patch):
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(patch_rgb)
        tensor = self.transform(pil_image).double().unsqueeze(0)

        with torch.no_grad():
            features = model(tensor)
        return features.numpy().flatten()

    def feature_fusion(self, features, operation="max"):
        features_array = np.array(features)
        if operation == "max":
            return features_array.max(axis=0)
        elif operation == "mean":
            return features_array.mean(axis=0)
        else:
            return features_array.max(axis=0)

    def extract_features(self, image_path):
        cnn_model = ModelLoader.load_cnn_model()
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        patches = self.extract_patches(image, self.num_patches, self.patch_size)
        features = []

        for patch in patches:
            feat = self.get_patch_features(cnn_model, patch)
            features.append(feat)

        fused_features = self.feature_fusion(features)
        return fused_features, len(patches)

    def predict(self, image_path):
        features, num_patches = self.extract_features(image_path)
        svm_model = ModelLoader.load_svm_model()

        if svm_model is None:
            return None, None, None

        features_reshaped = features.reshape(1, -1)
        prediction = svm_model.predict(features_reshaped)[0]

        try:
            probability = svm_model.predict_proba(features_reshaped)[0]
            confidence = probability[prediction] * 100
        except AttributeError:
            decision = svm_model.decision_function(features_reshaped)[0]
            confidence = 50 + abs(decision) / (abs(decision) + 1) * 50

        return prediction, confidence, num_patches
