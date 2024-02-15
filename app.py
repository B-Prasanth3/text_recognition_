import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow Lite model
model_path = "model.tflite"
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
    # Preprocess the input image
    input_image = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Get the output
    output_text = interpreter.get_tensor(output_details[0]['index'])

    return output_text

def preprocess_image(image):
    # Replace this with your actual image preprocessing logic
    # Convert image to a format compatible with the model input
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    input_image = np.expand_dims(image_array, axis=0).astype(np.float32)

    return input_image

if __name__ == "__main__":
    main()
