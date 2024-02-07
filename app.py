import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

# Path to your TensorFlow Lite model
model_path = "2.tflite"

decode_index2numletter = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z']

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
    input_tensor_shape = interpreter.get_input_details()[0]['shape']

    if input_tensor_shape[0] == 1:
        # For models with batch size 1, directly set the input tensor
        interpreter.set_tensor(input_tensor_index, tf.convert_to_tensor(image, dtype=tf.float32))
    else:
        # For models with batch size greater than 1, reshape and set the input tensor
        interpreter.tensor(input_tensor_index)()[0, :, :, :] = tf.convert_to_tensor(image, dtype=tf.float32)


def get_output_tensor(interpreter, index):
    """Return the output tensor at the given index."""
    output = interpreter.get_tensor(interpreter.get_output_details()[index]['index'])
    return np.squeeze(output)

def recognize_character(window, interpreter):
    """Recognize a character using the TFLite model."""
    # Preprocess the window
    window = cv2.resize(window, (28, 28))
    window = np.expand_dims(window, axis=-1)
    window = np.expand_dims(window, axis=0)
    window = window / 255.0  # Normalize to [0, 1]

    # Set input tensor
    set_input_tensor(interpreter, window)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = get_output_tensor(interpreter, 0)

    # Convert the output to the predicted character
    predicted_index = np.argmax(output)
    predicted_character = decode_index2numletter[predicted_index]

    return predicted_character

def recognize_text(image_path, interpreter, window_size=(28, 28), step_size=10):
    """Run text recognition on the input image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use OpenCV for image loading
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
    st.write("Predicted Characters:", predicted_text)
