import os
import pickle
import tensorflow as tf
import numpy as np

model_path = "new_model/ai_human_voice_classifier.h5"

print(f"Loading model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print(f"Error loading model: {e}")
