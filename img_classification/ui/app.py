import streamlit as st
import requests

st.title("Image Classification")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Create a button to confirm the upload
    confirm_button = st.button("Confirm Upload")

    if confirm_button:
        # Access the uploaded image data
        image_data = uploaded_file.getvalue()
        response = requests.post(
            'http://localhost:5000/predict',
            files={'file': ('image.png', image_data, 'multipart/form-data')}
        )

        if response.status_code == 200:
            # Display the classification results
            predictions = response.json()
            for result in predictions:
                st.write(f"{result['class']}: {result['confidence']:.4f}")
        else:
            # Handle errors
            st.error("Failed to get a response from the API")
else:
    st.warning("Please upload an image")
