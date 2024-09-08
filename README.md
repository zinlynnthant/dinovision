## Dino Vision Project

### Contributors:
- Zin Lynn Thant
- Hein Htet Aung (David Chang)

### Supervisor:
- Tr. Cynthia

### Institution:
- Simbolo

---

### Project Overview

Dino Vision is a computer vision-based classifier designed to identify 43 different species of dinosaurs. This project uses TensorFlow and Keras to build a Convolutional Neural Network (CNN) that processes augmented image data for each dinosaur species, ensuring that all classes are balanced with 600 images each.

---

### Dataset and Augmentation

The dataset consists of various dinosaur images categorized by species. We applied image augmentation techniques to ensure each class contains exactly 600 images, using rotation, brightness adjustment, zoom, and horizontal flipping to prevent overfitting and improve model performance.

---

### Image Augmentation Code

# Augment data to achieve 400 images per species
for species in os.listdir(dataset_dir):
    species_dir = os.path.join(dataset_dir, species)
    augmented_species_dir = os.path.join(augmented_dir, species)
    os.makedirs(augmented_species_dir, exist_ok=True)

    image_count = len(os.listdir(species_dir))
    target_image_count = 400
    augmentations_needed = (target_image_count - image_count) // image_count + 1

    for image_name in os.listdir(species_dir):
        image_path = os.path.join(species_dir, image_name)
        image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_species_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= augmentations_needed:
                break

---

### Dataset Splitting

The dataset is split into three parts:
- Training Set: 70% of the total dataset
- Validation Set: 20% of the total dataset
- Testing Set: 10% of the total dataset

This splitting helps ensure the model generalizes well to unseen data.

train_split = 0.7
valid_split = 0.2
test_split = 0.1

# Further code for splitting

---

### Model Architecture

The model is a Convolutional Neural Network (CNN) built using Keras Sequential API. The architecture includes 4 convolutional layers, max-pooling, and a dense layer with Dropout to reduce overfitting. The final layer uses a softmax activation function, allowing classification across 43 dinosaur species.

#### Model Layers:
1. Conv2D layer with 32 filters
2. MaxPooling2D layer
3. Conv2D layer with 64 filters
4. MaxPooling2D layer
5. Conv2D layer with 128 filters
6. MaxPooling2D layer
7. Flatten layer
8. Dropout (0.5) for regularization
9. Dense (512) fully connected layer
10. Dense (43) output layer

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(43, activation='softmax') # Output layer for 43 species
])

---

### Model Compilation

We used categorical crossentropy as the loss function since this is a multi-class classification problem. Adam optimizer was chosen for its efficiency, and accuracy was used as the performance metric.

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

---

### Training the Model

We trained the model using the ImageDataGenerator to rescale image data, with 15 epochs for convergence. The training generator processes images from the training set, while the validation generator evaluates the model’s performance during training.

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

---

### Testing the Model with a New Image

After loading the saved model and training it successfully, the model's performance can be evaluated by uploading and testing new images. This ensures that the model correctly classifies new dinosaur images.

### Load the Saved Model

```python
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('/content/gdrive/MyDrive/dinosaur_classification_model.h5')
```

### Upload and Test with a New Image

```python
from google.colab import files
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# List of species corresponding to class indices
species_names = [
    "Brachiosaurus", "Parasaurolophus", "Dreadnoughtus", "Ceratosaurus", "Compsognathus",
    "Triceratops", "Mamenchisaurus", "Spinosaurus", "Allosaurus", "Microceratus",
    "Nothosaurus", "Giganotosaurus", "Ankylosaurus", "Therizinosaurus", "Kentrosaurus",
    "Velociraptor", "Oviraptor", "Monolophosaurus", "Corythosaurus", "Dimetrodon",
    "Apatosaurus", "Pteranodon", "Pyroraptor", "Pachycephalosaurus", "Baryonyx",
    "Carnotaurus", "Tyrannosaurus rex", "Gallimimus", "Ouranosaurus", "Mosasaurus",
    "Dilophosaurus", "Tarbosaurus", "Suchomimus", "Pachyrhinosaurus", "Dimorphodon",
    "Quetzalcoatlus", "Smilodon", "Sinoceratops", "Lystrosaurus", "Iguanodon",
    "Nasutoceratops", "Stygimoloch", "Stegosaurus"
]

# Upload the image file
uploaded = files.upload()

# Load and preprocess the image
for filename in uploaded.keys():
    img_path = filename
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index

    # Get the species name
    predicted_species = species_names[predicted_class]
    print(f"Predicted species for {filename}: {predicted_species}")
```

### Final Test Accuracy and Classification Metrics

The model was evaluated on a test dataset with 2,580 images belonging to 43 dinosaur species. The following performance metrics were achieved:

- **Test Loss:** 2.1214
- **Test Accuracy:** 0.6178

The model achieved a balanced accuracy across species, though some species performed better than others based on precision, recall, and F1-score. Below are the summary statistics:

- **Accuracy:** 61.78%
- **Precision:** 62.70%
- **Recall:** 61.78%
- **F1-Score:** 61.83%

These metrics indicate that the model successfully generalizes to unseen data and provides an acceptable performance level for dinosaur species classification.

### Conclusion

The Dino Vision project uses a pre-trained CNN model to classify 43 species of dinosaurs. By leveraging saved models and new image testing, this implementation achieves reasonable performance. With an overall test accuracy of 61.78%, the model is capable of recognizing new dinosaur images. Future improvements could include further data augmentation and model fine-tuning to enhance accuracy and reduce classification errors.
