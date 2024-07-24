import tensorflow as tf
import numpy as np
import logging
import warnings

# Suppress the specific warning about compiled metrics
warnings.filterwarnings('ignore', category=UserWarning, message='Compiled the loaded model, but the compiled metrics have yet to be built.*')

# Suppress TensorFlow logs (optional)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class FlyingVehiclePredictor:
    def __init__(self, model_path):
         self.model = tf.keras.models.load_model(model_path)
    
    def predict_image(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)
       
            
        return predicted_class

# Initialize the predictor with the saved model
predictor = FlyingVehiclePredictor('flying_vehicles_model.h5')

# Prompt the user to input the image path
test_image_path = input("Please enter the path to the image: ")

# Make a prediction
predicted_class = predictor.predict_image(test_image_path)
print("Predicted class for the image:", predicted_class)
