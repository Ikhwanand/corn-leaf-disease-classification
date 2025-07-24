# 🌽 Corn Leaf Disease Classification

A deep learning-based web application for detecting and classifying corn leaf diseases using computer vision. The system can identify three different types of corn diseases with high accuracy.

## 🎯 Features

- Real-time disease detection and classification
- Support for multiple image formats (JPG, JPEG, PNG)
- Interactive web interface built with Streamlit
- High accuracy disease classification (98.60%)
- Detailed disease information and treatment recommendations
- GPU acceleration support (when available)

## 🦠 Supported Diseases

1. **Grey Leaf Spot**
   - Caused by: Cercospora zeae-maydis fungus
   - Symptoms: Rectangular gray spots on leaves
   - Treatment: Strobilurin or triazole-based fungicides

2. **Corn Rust**
   - Caused by: Puccinia sorghi fungus
   - Symptoms: Reddish-brown rust pustules on leaves
   - Treatment: Triazole or strobilurin-based fungicides

3. **Leaf Blight**
   - Caused by: Exserohilum turcicum fungus
   - Symptoms: Elongated elliptical brown spots on leaves
   - Treatment: Mancozeb or chlorothalonil-based fungicides

## 🏗️ Project Structure

```
corn-leaf-disease-classification/
├── data/                      # Raw dataset directory
├── data_rebalanced/          # Processed and balanced dataset
│   ├── train/                # Training data
│   ├── test/                 # Testing data
│   └── valid/                # Validation data
├── models/                   # Trained model files
│   ├── corn_disease_model.onnx    # ONNX model
│   └── custom_corn_cnn_weights.h5 # CNN model weights
├── streamlit-app/           # Web application
│   ├── main.py             # Streamlit application code
│   └── requirements.txt    # App dependencies
├── nb-1.ipynb              # Model training notebook
└── pyproject.toml          # Project configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/corn-leaf-disease-classification.git
cd corn-leaf-disease-classification
```

2. Install the required packages:
```bash
cd streamlit-app
pip install -r requirements.txt
```

### Running the Application

1. Navigate to the streamlit-app directory:
```bash
cd streamlit-app
```

2. Start the Streamlit server:
```bash
streamlit run main.py
```

3. Open your browser and go to `http://localhost:8501`

## 🔧 Technical Details

- **Model Architecture**: Custom CNN (Convolutional Neural Network)
- **Framework**: ONNX Runtime with CPU/GPU support
- **Input Size**: 224x224 pixels
- **Output**: 3 disease classes with confidence scores
- **Performance**: 98.60% accuracy on test set

## 📊 Dataset

The dataset contains corn leaf images with three different disease classes. The data is organized into train, test, and validation sets, with XML annotations for each image. The dataset has been rebalanced to ensure equal representation of all classes.

Dataset statistics:
- Training set: 6,586 images
- Validation and test sets included
- Multiple angles and lighting conditions
- High-quality annotations

## 🛠️ Development

The project uses several key technologies:
- **Streamlit**: For the web interface
- **ONNX Runtime**: For efficient model inference
- **OpenCV**: For image processing
- **PIL**: For image handling
- **NumPy**: For numerical operations

## 📄 License

This project is licensed under CC BY 4.0. The dataset is provided by Roboflow users.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
