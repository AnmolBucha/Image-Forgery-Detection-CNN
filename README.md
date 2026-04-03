# Image Forgery Detection with CNNs

![ForgeryShield](https://img.shields.io/badge/ForgeryShield-AI%20Powered-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-4B8BBE?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)

A deep learning-based web application for detecting image forgeries using CNN and SVM. Upload images to instantly analyze if they have been tampered with or are authentic.

## Features

- **AI-Powered Detection**: Uses CNN + SVM architecture for accurate forgery detection
- **Modern Web Interface**: Beautiful, responsive Flask web application
- **Real-time Analysis**: Instant feedback with confidence scores
- **Batch Processing**: Analyze multiple images at once
- **Analytics Dashboard**: Track statistics and trends
- **Analysis History**: View and manage all past analyses
- **Drag & Drop Upload**: Easy image upload interface
- **Mobile Responsive**: Works on all devices

## Live Demo

The web application provides:
- Single image upload and analysis
- Batch image processing
- Interactive analytics dashboard with charts
- Paginated analysis history
- Filter and search capabilities

## System Overview

The pipeline of the system is:
1. Train the CNN with image patches close to the distribution of the images
2. Extract features from images by breaking them into patches and applying feature fusion
3. Use an SVM classifier on the 400 extracted features for final classification

### Architecture

The CNN architecture uses SRM (High-Pass) filters for noise residual extraction, followed by:
- 2 convolutions → Max Pooling → 4 convolutions → Max Pooling → 3 convolutions
- Feature fusion (max/mean pooling) for image-level representation
- SVM classifier for binary classification (Forged/Authentic)

## Results

| Dataset | Accuracy |
|---------|----------|
| CASIA2  | 96.82% ± 1.19% |
| NC2016  | 84.89% ± 6.06% |

## Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Image-Forgery-Detection-CNN
```

### 2. Install Dependencies

**For Core ML Training:**
```bash
pip install -r requirements.txt
```

**For Web Application:**
```bash
cd web_app
pip install -r requirements.txt
```

### 3. Train Models (Optional - pre-trained models included)

```bash
cd ..
python src/extract_patches.py
python src/train_net.py
python src/feature_extraction.py
python src/svm_classification.py
```

### 4. Copy Models to Web App

```bash
cp data/output/pre_trained_cnn/CASIA2_WithRot_LR001_b128_nodrop.pt web_app/models/cnn_model.pt
cp data/output/pre_trained_svm/CASIA2_WithRot_LR001_b128_nodrop.pt web_app/models/svm_model.joblib
```

## Usage

### Start the Web Application

```bash
cd web_app
python app.py
```

Open your browser and go to: **http://localhost:5001**

### Pages

- **Home** (`/`) - Upload and analyze images
- **Dashboard** (`/dashboard`) - View analytics and statistics
- **History** (`/history`) - View all past analyses
- **About** (`/about`) - Project information

## Project Structure

```
Image-Forgery-Detection-CNN/
├── src/                    # Source code
│   ├── cnn/               # CNN implementation
│   │   ├── cnn.py        # CNN architecture
│   │   ├── train_cnn.py   # Training script
│   │   └── SRM_filters.py # SRM high-pass filters
│   ├── classification/     # SVM classifier
│   │   └── SVM.py         # SVM implementation
│   ├── feature_fusion/    # Feature fusion
│   ├── patch_extraction/  # Patch extraction
│   └── plots/             # Visualization
├── web_app/               # Web application
│   ├── app.py            # Flask application
│   ├── config.py         # Configuration
│   ├── database.py       # SQLite database
│   ├── models/           # Trained models
│   ├── templates/         # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   ├── history.html
│   │   └── about.html
│   ├── static/
│   │   ├── css/style.css # Styling
│   │   ├── js/main.js    # JavaScript
│   │   ├── uploads/      # Uploaded images
│   │   └── results/      # Result images
│   └── utils/
│       └── image_processor.py
├── data/                   # Dataset files
│   └── output/            # Training outputs
├── reports/               # Project reports
└── requirements.txt       # Dependencies
```

## Technologies Used

- **PyTorch** - Deep learning framework
- **scikit-learn** - SVM classification
- **Flask** - Web framework
- **SQLite** - Database for history
- **OpenCV** - Image processing
- **JavaScript** - Frontend interactions
- **CSS3** - Modern styling with animations

## Datasets

- [CASIA2](https://www.kaggle.com/sophatvathana/casia-dataset) - Image splicing and copy-move forgeries
- [NC2016](https://www.nist.gov/itl/iad/mig/media-forensics-challenge) - NIST forensics challenge

## References

This project is inspired by:
- Y. Rao et al., "A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images" ([IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911))

## License

MIT License

## Author

Final Year Major Project - Image Forgery Detection using Deep Learning
