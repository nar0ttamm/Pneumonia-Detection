import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import pickle

# Image parameters - 150x150 balances detail and processing efficiency
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

def create_model():
    # CNN architecture for pneumonia detection
    model = Sequential([
        # Initial convolution layer for basic feature detection
        Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        
        # Second layer for intermediate pattern detection
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third layer for complex feature detection
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten layer for classification preparation
        Flatten(),
        Dense(64, activation='relu'),
        # Dropout layer for regularization
        Dropout(0.5),
        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ])
    
    # Model compilation settings
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    # Data augmentation configuration
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        rotation_range=20,  # Random rotation
        width_shift_range=0.2,  # Horizontal shift
        height_shift_range=0.2,  # Vertical shift
        horizontal_flip=True,  # Horizontal flip for symmetrical features
        fill_mode='nearest'
    )

    # Validation data preprocessing
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Training data preparation
    print("Initializing training data...")
    train_generator = train_datagen.flow_from_directory(
        '../chest_xray/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Validation data preparation
    print("Initializing validation data...")
    validation_generator = validation_datagen.flow_from_directory(
        '../chest_xray/val',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Model creation and training
    print("Initiating model training...")
    model = create_model()
    
    history = model.fit(
        train_generator,
        epochs=10,  # Optimal epoch count for this application
        validation_data=validation_generator
    )

    # Model saving
    print("Saving trained model...")
    model.save('pneumonia_model.h5')
    
    # Additional model data saving
    with open('pneumonia_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'history': history.history,
            'img_height': IMG_HEIGHT,
            'img_width': IMG_WIDTH
        }, f)
    
    print("Model training and saving completed.")

if __name__ == "__main__":
    train_model() 