import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load your trained model (ensure you have uploaded your model file)
model = tf.keras.models.load_model('/content/dinosaur_classification_model_latestupdated.h5')

# Function to predict image
def predict_image(image):
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize the uploaded image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
st.title('Dinosaur Species Classifying App')

# Allow all image types by not restricting the file type
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Use PIL to open the image from the BytesIO object
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    # Call the predict function with the image object
    label = predict_image(image)
    
    st.write(f'Prediction: {label}')
