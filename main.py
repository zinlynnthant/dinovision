import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
# Define constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
# Load your trained model (ensure you have uploaded your model file)
model = tf.keras.models.load_model('dinosaur_classification_model.h5')
# List of actual class names (dinosaur species)
class_names = [
    'Allosaurus', 'Dilophosaurus', 'Mamenchisaurus', 'Pachyrhinosaurus', 'Stygimoloch',
    'Ankylosaurus', 'Dimetrodon', 'Microceratus', 'Parasaurolophus', 'Suchomimus',
    'Apatosaurus', 'Dimorphodon', 'Monolophosaurus', 'Pteranodon', 'Tarbosaurus',
    'Baryonyx', 'Dreadnoughtus', 'Mosasaurus', 'Pyroraptor', 'Therizinosaurus',
    'Brachiosaurus', 'Gallimimus', 'Nasutoceratops', 'Quetzalcoatlus', 'Triceratops',
    'Carnotaurus', 'Giganotosaurus', 'Nothosaurus', 'Sinoceratops', 'Tyrannosaurus rex',
    'Ceratosaurus', 'Iguanodon', 'Ouranosaurus', 'Smilodon', 'Velociraptor',
    'Compsognathus', 'Kentrosaurus', 'Oviraptor', 'Spinosaurus',
    'Corythosaurus', 'Lystrosaurus', 'Pachycephalosaurus', 'Stegosaurus'
]
# Function to predict image
def predict_image(image):
    # Convert the image to RGB if it has an alpha channel (transparency)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize the uploaded image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image to [0, 1]
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Get the index of the class with the highest prediction
    predicted_class = class_names[predicted_class_index]  # Get the class name
    
    return predicted_class
# Streamlit app UI design
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    .title {
        font-size: 48px;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        color: #2980b9;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
# Title and subtitle
st.markdown('<div class="title">🦕 DinoVision</div>', unsafe_allow_html=True)


# File uploader with enhanced UI
uploaded_file = st.file_uploader("Upload a Dinosaur Image (PNG, JPG, JPEG, BMP, GIF, TIFF)", type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"])

# Side-by-side layout for better spacing
col1, col2 = st.columns(2)
if uploaded_file is not None:
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image', use_column_width=True)
        
    with col2:
        st.write("🦖 Classifying... Please wait.")
        with st.spinner("Analyzing the image..."):
            predicted_class = predict_image(image)
        
        # Display the result with some styling
        st.success(f'Prediction: {predicted_class}')
        
        # Add a progress bar for a more interactive experience
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
else:
    st.write("Upload an image to start the classification.")
