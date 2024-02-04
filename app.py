import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image

# Path to your TensorFlow Lite model
model_path = "model.tflite"

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size, Image.ANTIALIAS)
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = tf.expand_dims(img, axis=0)
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
    input_size = (your_width, your_height)  # Replace with the input size expected by your model
    image = preprocess_image(image_path, input_size)

    # Set the input tensor
    set_input_tensor(interpreter, image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_tensor = get_output_tensor(interpreter, 0)

    # Perform post-processing on the output tensor if needed
    # ...

    return output_tensor

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
    recognition_result = recognize_text(path, interpreter)

    # Display the recognition result
    st.write("Recognition Result:", recognition_result)
