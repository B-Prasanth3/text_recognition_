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

def run_decoder_on_output(output_text_indices):
    # Placeholder implementation assuming output_text_indices is a list of character indices
    decoded_text = ''.join([decode_index2numletter[idx] for idx in output_text_indices])
    return decoded_text

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = Image.open(image_path).convert('L')  # Open in grayscale mode
    img = img.resize(input_size, Image.ANTIALIAS)
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = img[..., tf.newaxis]  # Add a new axis for the channel dimension
    return img

def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    input_tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor_shape = interpreter.get_input_details()[0]['shape']

    if len(input_tensor_shape) == 4:
        # For models with 4D input (batch, height, width, channels)
        interpreter.tensor(input_tensor_index)()[0] = image
    elif len(input_tensor_shape) == 3:
        # For models with 3D input (height, width, channels)
        interpreter.tensor(input_tensor_index)()[0, :, :, 0] = image  # Assuming single-channel image
    else:
        # Handle other cases based on your model's input requirements
        raise ValueError("Unsupported input tensor shape")

def recognize_text(image_path, interpreter, window_size=(31, 200), step_size=10):
    """Run text recognition on the input image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    predicted_text = ""
    image_height, image_width = image.shape[:2]

    for y in range(0, image_height - window_size[0], step_size):
        for x in range(0, image_width - window_size[1], step_size):
            window = image[y:y + window_size[0], x:x + window_size[1]]
            window = cv2.resize(window, (128, 64))  # Assume KerasOCR expects this size
            window = window / 255.0  # Normalize to [0, 1]
            set_input_tensor(interpreter, window)
            interpreter.invoke()

            # Assuming you have a decoder output, adjust the following line
            output_indices = [np.argmax(interpreter.tensor(i)()[0]) for i in interpreter.get_output_details()]
            character = run_decoder_on_output(output_indices)
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
