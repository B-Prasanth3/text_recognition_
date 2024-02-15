import streamlit as st
from PIL import Image
import easyocr

def main():
    st.title("Text Recognition App")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Recognizing text...")

        # Perform text recognition on the image using easyocr
        result = recognize_text(image)

        st.success(f"Text Recognition Result: {result}")

def recognize_text(image):
    # Use easyocr reader
    reader = easyocr.Reader(['en'])
    # Extract text from the image
    result = reader.readtext(np.array(image))

    return result

if __name__ == "__main__":
    main()
