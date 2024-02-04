import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

# Path to your TensorFlow Lite model
model_path = "model.tflite"

decode_index2numletter = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z'];

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = Image.open(image_path).convert('L')  # Open in grayscale mode
    img = img.resize(input_size, Image.ANTIALIAS)
    img = np.array(img)
    if len(img.shape) == 3 and img.shape[2] == 3:  # Check for 3 color channels
        img = tf.image.rgb_to_grayscale(img)  # Convert to grayscale if needed
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = img[..., tf.newaxis]  # Add a new axis for the channel dimension
    return img
    
def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(input_tensor_index)()[0] = image
  
def get_output_tensor(interpreter, index):
    """Return the output tensor at the given index."""
    output = interpreter.get_tensor(interpreter.get_output_details()[index]['index'])
    return np.squeeze(output)
  
def recognize_text(image_path, interpreter, window_size=(28, 28), step_size=10):
    """Run text recognition on the input image."""
    image = cv2.imread(image_path)  # Use OpenCV for image loading
    predicted_text = ""
    image_height, image_width = image.shape[:2]
    for y in range(0, image_height - window_size[0], step_size):
        for x in range(0, image_width - window_size[1], step_size):
            window = image[y:y+window_size[0], x:x+window_size[1]]
            character = recognize_character(window, interpreter)
            predicted_text += character

    return predicted_text

# Input Fields
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    path = os.path.join("/tmp", uploaded_file.name)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run text recognition and get the result
    predicted_text = recognize_text(path, interpreter)

    # Display the recognition result
    st.write("Recognized Text:", predicted_text)

    # Display the recognition result
    st.write("Predicted Characters:", predicted_character)

