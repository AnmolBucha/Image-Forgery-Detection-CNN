import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
from config import Config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class ModelLoader:
    _cnn_instance = None
    _svm_instance = None

    @classmethod
    def load_cnn_model(cls):
        if cls._cnn_instance is None:
            from src.cnn.cnn import CNN

            cls._cnn_instance = CNN()
            if os.path.exists(Config.CNN_MODEL_PATH):
                cls._cnn_instance.load_state_dict(
                    torch.load(Config.CNN_MODEL_PATH, map_location=torch.device("cpu"))
                )
            cls._cnn_instance.eval()
            cls._cnn_instance.double()
        return cls._cnn_instance

    @classmethod
    def load_svm_model(cls):
        if cls._svm_instance is None:
            from joblib import load

            if os.path.exists(Config.SVM_MODEL_PATH):
                cls._svm_instance = load(Config.SVM_MODEL_PATH)
        return cls._svm_instance

    @classmethod
    def is_models_available(cls):
        return os.path.exists(Config.CNN_MODEL_PATH) and os.path.exists(
            Config.SVM_MODEL_PATH
        )
