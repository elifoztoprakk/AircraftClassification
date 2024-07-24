import tensorflow as tf
import numpy as np
import logging
import warnings

class FlyingVehiclePredictor:
    def __init__(self, model_path, threshold=0.6):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
    
    def predict_image(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        max_prob = np.max(predictions)
        predicted_class_idx = np.argmax(predictions)
       
        if max_prob < self.threshold:
            predicted_class = 'UNKNOWN'
            probability = max_prob
        else:
            if predicted_class_idx == 0:
                predicted_class = 'DRONE'
            elif predicted_class_idx == 1:
                predicted_class = 'FIGHTER JET'
            elif predicted_class_idx == 2:
                predicted_class = 'HELICOPTER'
            else:
                predicted_class = 'PASSENGER PLANE'
            probability = max_prob
     
        return predicted_class, probability

# Initialize the predictor with the saved model
predictor = FlyingVehiclePredictor('flying_vehicles_model.h5', threshold=0.6)

# Prompt the user to input the image path
test_image_path = input("Please enter the path to the image: ")

# Make a prediction
predicted_class, probability = predictor.predict_image(test_image_path)
if predicted_class == 'UNKNOWN':
    print(f"This image is not an aircraft.")
else:
    print(f"Predicted class for the image: {predicted_class} with a probability of {probability*100:.2f}%")
