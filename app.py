import streamlit as st
from PIL import Image
import easyocr

def main():
    st.title("Text Recognition App")

    # Add language selection (use st.selectbox/st.radio etc.)
    recognition_language = st.selectbox("Select Recognition Language", ["English", "French", "Spanish"])

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Downscale for larger images
            if image.width > 1000:
                width = 1000
                height = int(image.height * width / image.width)
                image = image.resize((width, height))

            st.write("Recognizing text...")

            # Preprocess if needed (e.g., grayscale conversion)
            # ...

            reader = easyocr.Reader([recognition_language])
            result = reader.readtext(np.array(image))

            download_button = st.button("Download Text")
            if download_button:
                with open("recognized_text.txt", "w") as f:
                    f.write(result)
                st.success("Text downloaded as recognized_text.txt")

            st.success(f"Text Recognition Result: {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
