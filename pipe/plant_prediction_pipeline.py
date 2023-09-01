import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def plant_recognition(url):
    cnn = tf.keras.models.load_model('models/Plant_Recognition.h5')
    # image_path = 'https://i.ibb.co/d2KDGFc/blob.jpg'

    # image = tf.keras.preprocessing.image.load_img(image_path,target_size=[64,64])
    # input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # input_arr = np.array([input_arr]) #Converting single image to batch
    image_url = url

    # Download the image from the URL
    response = requests.get(image_url)
    with open('temp_image.jpg', 'wb') as f:
        f.write(response.content)

    # Load the downloaded image using PIL
    image = Image.open('temp_image.jpg')
    image = image.resize((64, 64))  # Resize the image

    # Convert the image to a numpy array
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Adding batch dimension

    predictions = cnn.predict(input_arr)

    return predictions

