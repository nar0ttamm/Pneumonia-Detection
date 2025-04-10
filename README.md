# 🫁 Pneumonia Detection from Chest X-rays

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-brightgreen.svg)

A deep learning-powered web application that detects pneumonia from chest X-ray images using Convolutional Neural Networks (CNN). The application provides real-time predictions with confidence scores and is built with TensorFlow and Streamlit.

## 🌟 Features

- **Real-time Prediction**: Upload and get instant pneumonia detection results
- **User-friendly Interface**: Simple and intuitive web interface
- **Confidence Scores**: Get probability scores for predictions
- **Support for Multiple Image Formats**: Handles various image formats (JPG, PNG)
- **Automatic Image Processing**: Handles both RGB and grayscale images
- **Responsive Design**: Works on both desktop and mobile devices

## 🚀 Demo

The application is deployed on Streamlit Cloud and can be accessed here: [Live Demo](your-streamlit-url)

![Demo GIF](path_to_demo_gif)

## 🛠️ Technology Stack

- **Deep Learning Framework**: TensorFlow 2.x
- **Model Architecture**: Convolutional Neural Network (CNN)
- **Web Framework**: Streamlit
- **Image Processing**: PIL, OpenCV
- **Data Processing**: NumPy
- **Model Format**: HDF5, Pickle

## 📊 Model Architecture

```
Model: Sequential
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
Trainable params: 1,240,193
Non-trainable params: 0
```

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application locally**
   ```bash
   streamlit run app.py
   ```

## 📈 Model Training

The model was trained on the Chest X-Ray Images (Pneumonia) dataset with the following specifications:
- Input Image Size: 150x150 pixels
- Training Steps: 10 epochs
- Data Augmentation: Rotation, Width/Height Shift, Horizontal Flip
- Validation Split: 20%

## 🎯 Usage

1. Open the application in your web browser
2. Upload a chest X-ray image using the file uploader
3. Click the "Predict" button
4. View the results and confidence score

## 📝 Project Structure

```
pneumonia-detection/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── requirements.txt       # Project dependencies
├── pneumonia_model.h5     # Trained model in H5 format
├── pneumonia_model.pkl    # Trained model in PKL format
└── README.md             # Project documentation
```

## 🚨 Important Note

This application is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## 🔄 Model Performance

- Training Accuracy: XX%
- Validation Accuracy: XX%
- Test Set Performance: XX%

## 🌐 Deployment

### Deploying to Streamlit Cloud

1. Create a Streamlit account at [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Deploy the application with these settings:
   - Main file path: `app.py`
   - Python version: 3.9+
   - Requirements: `requirements.txt`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub Profile](your-github-profile)

## 🙏 Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- TensorFlow Team
- Streamlit Team

## 📞 Contact

For any queries or suggestions, please reach out to:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](your-linkedin-profile)

---
Made with ❤️ using TensorFlow and Streamlit 