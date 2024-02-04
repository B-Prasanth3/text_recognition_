import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image

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
  
def recognize_text(image_path, interpreter):
    """Run text recognition on the input image."""
    input_size = (28, 28)  # Replace with the input size expected by your model
    image = preprocess_image(image_path, input_size)

    # Set the input tensor
    set_input_tensor(interpreter, image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_tensor = get_output_tensor(interpreter, 0)

    # Find the index of the most probable class
    index = np.argmax(output_tensor)

    # Map the index to the corresponding character
    predicted_character = decode_index2numletter[index]

    return predicted_character


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
    predicted_character = recognize_text(path, interpreter)

    # Display the recognition result
    st.write("Predicted Characters:", predicted_character)

