# Image Forgery Detection Web Application

A Flask-based web interface for detecting image forgeries using CNN and SVM.

## Features

- **Single Image Analysis**: Upload and analyze individual images
- **Batch Processing**: Analyze multiple images at once
- **Real-time Results**: Get instant feedback with confidence scores
- **Modern UI**: Clean, responsive design with dark theme
- **Drag & Drop**: Easy file upload interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the models (if not already trained):
```bash
# From project root
cd ..
python src/extract_patches.py
python src/train_net.py
python src/feature_extraction.py
python src/svm_classification.py
```

3. Copy trained models to the models folder:
```bash
cp ../data/output/pre_trained_cnn/*.pt web_app/models/cnn_model.pt
cp ../data/output/pre_trained_svm/*.pt web_app/models/svm_model.joblib
```

Note: You may need to rename the files to `cnn_model.pt` and `svm_model.joblib`

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to:
```
http://localhost:5000
```

## Project Structure

```
web_app/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── models/             # Trained CNN and SVM models
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   └── about.html
├── static/
│   ├── css/
│   │   └── style.css   # Styling
│   ├── js/
│   │   └── main.js     # Frontend JavaScript
│   ├── uploads/        # Uploaded images
│   └── results/       # Result images
└── utils/
    └── image_processor.py  # Image processing utilities
```

## How It Works

1. **Upload**: User uploads an image through the web interface
2. **Processing**: Image is divided into patches
3. **Feature Extraction**: CNN extracts 400-D features from each patch
4. **Fusion**: Features are fused using max pooling
5. **Classification**: SVM determines if image is forged or authentic
6. **Results**: Display prediction with confidence score

## Requirements

- Python 3.8+
- Flask 2.3+
- PyTorch 1.8+
- OpenCV 4.5+
- scikit-learn 1.3+
- Trained CNN and SVM models

## API Endpoints

- `POST /upload` - Upload single image for analysis
- `POST /batch-upload` - Upload multiple images
- `GET /api/check-models` - Check if models are loaded
- `GET /api/train-models` - Get training instructions

## Notes

- Images are temporarily stored in `static/uploads/`
- Maximum file size: 16MB
- Supported formats: JPG, PNG, JPEG, TIF, BMP, WEBP
- Models must be placed in `models/` folder before running
