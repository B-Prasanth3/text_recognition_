import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow Lite model
model_path = "2.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def main():
    st.title("Text Recognition App")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform text recognition on the image using the loaded model
        result = recognize_text(image)

        st.success(f"Text Recognition Result: {result}")

def recognize_text(image):
    # Get input details
    input_details = interpreter.get_input_details()

    # Print expected input shape
    input_shape = input_details[0]['shape']
    print("Expected Input Shape:", input_shape)

    # Preprocess the input image
    input_image = preprocess_image(image)

    # Print actual input shape
    print("Actual Input Shape:", input_image.shape)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Get output details
    output_details = interpreter.get_output_details()

    # Get the expected output shape
    output_shape = output_details[0]['shape']
    print("Expected Output Shape:", output_shape)

    # Get the output
    output_text = interpreter.get_tensor(output_details[0]['index'])

    return output_text

def preprocess_image(image):
    # Get expected dimensions
    input_shape = input_details[0]['shape'][1:3]
    width, height = input_shape
    
    # Resize the image while preserving aspect ratio
    image = image.resize((width, height), Image.ANTIALIAS)
    
    # Convert to NumPy array and normalize
    image_array = np.array(image) / 255.0
    
    # Expand dimension for batch size
    input_image = np.expand_dims(image_array, axis=0).astype(np.float32)
    return input_image





if __name__ == "__main__":
    main()
