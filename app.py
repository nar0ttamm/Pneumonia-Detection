import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pickle

# Application configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁"
)

# Model loading function with caching
@st.cache_resource
def load_prediction_model():
    try:
        # Primary model loading attempt from pickle
        with open('pneumonia_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            return model_data['model']
    except:
        # Fallback model loading from h5
        return tf.keras.models.load_model('pneumonia_model.h5')

def preprocess_image(img):
    # Image resizing to model dimensions
    img = img.resize((150, 150))
    
    # RGB format conversion
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Array conversion and normalization
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def main():
    # Application header
    st.title("Pneumonia Detection from Chest X-rays")
    st.write("""
    Upload a chest X-ray image for pneumonia detection analysis.
    """)

    # Image upload interface
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Image display
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption='Uploaded X-ray image', use_column_width=True)

            # Analysis button
            if st.button('Analyze X-ray'):
                with st.spinner('Processing image...'):
                    # Model loading
                    model = load_prediction_model()
                    
                    # Image preprocessing
                    processed_img = preprocess_image(img)
                    
                    # Prediction generation
                    prediction = model.predict(processed_img)
                    probability = prediction[0][0]

                    # Results display
                    st.subheader('Analysis Results:')
                    if probability > 0.5:
                        st.error(f'Pneumonia detected with {probability:.2%} confidence')
                    else:
                        st.success(f'Normal chest X-ray with {1-probability:.2%} confidence')

                    # Medical disclaimer
                    st.write("""
                    **Important Note:** This tool is for educational purposes only. 
                    Consult healthcare professionals for medical diagnosis and treatment.
                    """)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please ensure a valid X-ray image is uploaded.")

if __name__ == "__main__":
    main() 