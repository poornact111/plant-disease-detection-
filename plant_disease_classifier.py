import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class PlantDiseaseClassifier:
    def __init__(self):
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        self.model = None
        self.class_names = None
    
    def prepare_data(self, dataset_path):
        """
        Prepare data generators for training and validation
        dataset_path: Path to the main dataset directory containing train and valid folders
        """
        # Setup data directories
        train_dir = os.path.join(dataset_path, 'train')
        valid_dir = os.path.join(dataset_path, 'valid')
        
        # Get class names from directory
        self.class_names = sorted(os.listdir(train_dir))
        print(f"Found {len(self.class_names)} classes")
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        self.valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        return self.train_generator, self.valid_generator
    
    def build_model(self):
        """Build the CNN model"""
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, epochs=10):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=epochs,
            validation_data=self.valid_generator,
            validation_steps=self.valid_generator.samples // self.batch_size
        )
        
        return history
    
    def save_model(self, model_path):
        """Save the model and class names"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save class names
        class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.npy')
        np.save(class_names_path, self.class_names)
    
    def load_model(self, model_path):
        """Load the model and class names"""
        self.model = load_model(model_path)
        
        # Load class names
        class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.npy')
        self.class_names = np.load(class_names_path, allow_pickle=True)
    
    def predict(self, image_path):
        """Predict disease for a given image"""
        img = image.load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100
        
        # Parse plant type and disease from class name
        class_name = self.class_names[predicted_class_index]
        plant_type = class_name.split('___')[0]
        disease = class_name.split('___')[1] if '___' in class_name else 'healthy'
        
        result = {
            'plant_type': plant_type,
            'disease': disease,
            'confidence': f"{confidence:.2f}%",
            'is_healthy': disease.lower() == 'healthy'
        }
        
        return result

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()