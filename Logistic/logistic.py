import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image, ImageTk


# Define labels for the different fruit and vegetable categories
labels = ["apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic", "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple", "pomegranate", "potato", "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip", "watermelon"]

with open('../logistic/model.pkl', 'rb') as f:
    model = pickle.load(f)



# Define CSS style for label




def predict(image):
    # Preprocess image
    img = cv2.resize(image, (32, 32))
    img_vector = img.flatten()
    # Use the trained model to predict the label of the input image
    label_number = model.predict([img_vector])[0]
    label = labels[label_number]
    a = f"Đây là {label} và nó thuộc fruit. 🍍 "
    b = f"Đây là {label} và nó thuộc rau củ. 🍅"
    # Determine whether the input image belongs to fruits or vegetables
    if label in ['apple', 'banana', 'kiwi', 'lemon', 'mango', 'orange', 'pear', 'pineapple', 'pomegranate', 'watermelon']:
        return a
    else:
        return b


st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">Nhận Dạng Và Phân Loại Rau Củ🍅 và Quả 🍍</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Chọn ảnh của bạn</h2>', unsafe_allow_html=True)
# Create a file uploader in Streamlit
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Check if file is uploaded
if uploaded_file is not None:
    # Load image using PIL
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption="Uploaded", use_column_width=True)
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Use the predict function to get the label of the uploaded image
    result = predict(img)
    # Display the label
    button = st.button("Dự Đoán")
    if button:
        st.write(f"<h3 style='text-align: center; color: white ;background-color: red;'>{result}</h3>", unsafe_allow_html=True)
    else:
        st.write("Please upload an image")
    
        


import streamlit as st

