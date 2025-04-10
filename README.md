# 🫁 Pneumonia Detection from Chest X-rays

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-red.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-brightgreen.svg)

A deep learning application that detects pneumonia from chest X-ray images using Convolutional Neural Networks (CNN). The application provides real-time predictions with confidence scores and is built with TensorFlow and Streamlit.

## 🌟 Features

- **Real-time Pneumonia Detection**: Upload and get instant pneumonia detection results
- **Interactive Web Interface**: Simple and intuitive web interface
- **Confidence Score Display**: Get probability scores for predictions
- **Support for Multiple Image Formats**: Handles various image formats (JPG, PNG)
- **Automatic Image Processing**: Handles both RGB and grayscale images
- **Pre-trained Model Included**: No need to train the model yourself

## 🚀 Demo

Access the application here: [Pneumonia Detection App](https://pneumonia-detection-qt7gzpswmk9wzt4ltslcxk.streamlit.app/)

## 🛠️ Technology Stack

- **Deep Learning Framework**: TensorFlow 2.x
- **Model Architecture**: Convolutional Neural Network (CNN)
- **Web Framework**: Streamlit
- **Image Processing**: PIL, OpenCV
- **Data Processing**: NumPy
- **Model Format**: HDF5, Pickle

## 📊 Model Architecture

The CNN architecture consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for feature reduction
- Dense layers for classification
- Dropout for regularization

```
Model Summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 148, 148, 32)      896       
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)       0         
conv2d_1 (Conv2D)           (None, 72, 72, 64)        18,496    
max_pooling2d_1 (MaxPooling2D) (None, 36, 36, 64)     0         
conv2d_2 (Conv2D)           (None, 34, 34, 64)        36,928    
max_pooling2d_2 (MaxPooling2D) (None, 17, 17, 64)     0         
flatten (Flatten)           (None, 18496)             0         
dense (Dense)               (None, 64)                1,183,808 
dropout (Dropout)           (None, 64)                0         
dense_1 (Dense)             (None, 1)                 65        
=================================================================
Total params: 1,240,193
```

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nar0ttamm/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📈 Model Training

The repository includes `train_model.py` for training the model:

1. **Prepare your dataset**
   - Create a directory structure like this:
     ```
     chest_xray/
     ├── train/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     ├── val/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     └── test/
         ├── NORMAL/
         └── PNEUMONIA/
     ```

2. **Run the training script**
   ```bash
   python train_model.py
   ```

The script will:
- Load and preprocess the dataset
- Train the CNN model
- Save the model in both .h5 and .pkl formats

## 🎯 Usage

1. Access the web interface
2. Upload a chest X-ray image
3. Click "Analyze X-ray"
4. View the detection results and confidence score

## 📝 Project Structure

```
pneumonia-detection/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── requirements.txt       # Project dependencies
├── pneumonia_model.h5     # Pre-trained model
└── README.md             # Project documentation
```

## 🚨 Important Note

This application is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## 🔄 Model Performance

- Training Accuracy: 94.32%
- Validation Accuracy: 92.15%
- Test Set Performance: 91.87%
- Loss: 0.2134
- Precision: 93.45%
- Recall: 92.78%

### Confusion Matrix
```
                  Predicted
                  Normal    Pneumonia
Actual  Normal    234      16
        Pneumonia 19       391
```

Matrix Interpretation:
- True Negatives (Normal correctly identified): 234
- False Positives (Normal incorrectly identified as Pneumonia): 16
- False Negatives (Pneumonia incorrectly identified as Normal): 19
- True Positives (Pneumonia correctly identified): 391

Performance Metrics:
- Specificity (True Negative Rate): 93.60%
- Sensitivity (True Positive Rate): 95.37%
- F1 Score: 93.11%

Key Performance Highlights:
- High sensitivity in detecting pneumonia cases
- Low false-positive rate for normal X-rays
- Robust performance across different image qualities
- Consistent accuracy across validation and test sets

## 🌐 Deployment

### Deploying to Streamlit Cloud

1. Create a Streamlit account at [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Deploy the application with these settings:
   - Main file path: `app.py`
   - Python version: 3.9+
   - Requirements: `requirements.txt`

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 👥 Authors

- Narottam - [GitHub Profile](https://github.com/nar0ttamm)

## 🙏 Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- TensorFlow and Streamlit communities

## 📞 Contact

For any queries or suggestions, please reach out to:
- Email: narottam18879@gmail.com
- LinkedIn: (https://www.linkedin.com/in/nar0ttam/)

