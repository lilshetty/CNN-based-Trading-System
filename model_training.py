# model_training.py 

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import pandas as pd
import os

class CNNTrading:
    def __init__(self, dataset_path='trading_dataset'):
        self.dataset_path = dataset_path
        self.image_size = (12, 15)
        self.batch_size = 32
        
    def load_dataset(self):
        """Load and prepare dataset for training"""
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{self.dataset_path}/train",
            labels='inferred',
            label_mode='categorical',
            class_names=['hold', 'buy', 'sell'],
            image_size=self.image_size,
            batch_size=self.batch_size,
            color_mode='grayscale'
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{self.dataset_path}/validation",
            labels='inferred',
            label_mode='categorical',
            class_names=['hold', 'buy', 'sell'],
            image_size=self.image_size,
            batch_size=self.batch_size,
            color_mode='grayscale'
        )
        
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{self.dataset_path}/test",
            labels='inferred',
            label_mode='categorical',
            class_names=['hold', 'buy', 'sell'],
            image_size=self.image_size,
            batch_size=self.batch_size,
            color_mode='grayscale'
        )
        
        return train_ds, val_ds, test_ds
    
    def build_model(self):
        """Create CNN model"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.image_size, 1)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=150):
        """Train the model"""
        # Load dataset
        train_ds, val_ds, test_ds = self.load_dataset()
        
        # Build model
        model = self.build_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Evaluate on test set
        test_results = model.evaluate(test_ds)
        print(f"Test accuracy: {test_results[1]:.4f}")
        
        return model, history

if __name__ == "__main__":
    # Train model
    trader = CNNTrading()
    model, history = trader.train_model()
    
    # Save model
    model.save('trading_model.h5')
