import os


class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.pt")
    SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.joblib")
    PATCH_SIZE = 128
    FEATURE_DIM = 400
    NUM_PATCHES = 100
