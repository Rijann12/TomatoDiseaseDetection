import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai

class_names = [ 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
# class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
#                     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
#                     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
#                     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                     'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
#                     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
#                     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
#                     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
#                     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
#                     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                       'Tomato___healthy']

#Database connection
import db_utils

db_utils.init_db() 

#Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("tomato_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # convert to batch
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    return predicted_class
#Precaution / Prevention tips
genai.configure(api_key="AIzaSyBhZHzQJ2a5Ll35ICau_LRFhSkpbN9r-B0EY")
model = genai.GenerativeModel("gemini-1.5-flash")
import requests
def get_precaution_from_ai(disease_name):
    try:
        res = requests.post(
            "http://localhost:5000/precaution",
            json={"disease": disease_name},
            timeout=10
        )
        if res.status_code == 200:
            return res.json().get("Precaution", " Could not parse suggestions.")
        else:
            return " Could not fetch suggestions right now. Please try again later."
    except Exception as e:  
        return f" Gemini service unreachable: {str(e)}"
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","View History"])

#Main Page

if(app_mode=="Home"):
    st.header("DeepLeaf")
    st.subheader("Smart Plant Disease Detection System")
    image_path = "home_page1.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
     ## Welcome to **DeepLeaf** – Your AI-Powered Plant Health Companion!

    Say goodbye to guesswork and hello to smart farming.  
    **DeepLeaf** uses deep learning to detect plant diseases from images — fast, simple, and accessible to everyone.

    ###  What is DeepLeaf?
    DeepLeaf is an intelligent plant disease detection platform that helps users identify crop issues by simply uploading a leaf image. Whether you're growing vegetables at home or managing a farm,
     DeepLeaf gives you quick insights and actionable suggestions to protect your plants.
    ###  What Does This System Do?
                
    This platform allows you to:
    -  Upload images of affected plant leaves
    -  Detect potential plant diseases using a trained CNN model
    -  Get instant prevention and treatment suggestions powered by AI

    ###  How It Works:
    1. **Capture or Select a Leaf Image** — Preferably with visible symptoms.
    2. **Upload the Image** in the **Disease Recognition** section.
    3. **Receive Diagnosis** — The model predicts the disease, and you'll get tailored advice.

    ###  Key Features:
    -  **Accurate Predictions** using a Convolutional Neural Network (CNN)
    -  **Real-time Analysis** for faster decision-making
    -  **Integrated AI Suggestions** for remedies and preventive actions
    -  **Simple, Clean Interface** with no login or complex setup required

    ###  Need Help?
    Visit the **About** page to learn more about the project, its development team, and the technology stack used.

    ---
     Ready to identify plant diseases?  
    Head over to the **Disease Recognition** tab in the sidebar to get started!
    """)
# elif app_mode == "View History":
#     st.header(" Prediction & Precaution History")

#     # --- Prediction History ---
#     st.subheader(" Model Prediction History")
#     prediction_data = db_utils.get_prediction_history()
#     if prediction_data:
#         st.table(
#             [{"Disease": row[0], "Timestamp": row[1]} for row in prediction_data]
#         )
#     else:
#         st.info("No prediction history found.")

#     # --- Precaution History ---
#     st.subheader(" Gemini Precaution History")
#     precaution_data = db_utils.get_precaution_history()
#     if precaution_data:
#         st.table(
#             [{"Disease": row[0], "Precaution": row[1], "Timestamp": row[2]} for row in precaution_data]
#         )
#     else:
#         st.info("No precaution history found.")
#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image:")

    predicted_disease= None
    precaution= None


    if test_image:
        file_type = test_image.type
        valid_types = ['image/jpeg', 'image/png']

        if file_type not in valid_types:
            st.error(" Invalid file type. Please upload a JPEG or PNG image.")
        else:
            if st.button("Show Image"):
                st.image(test_image, use_column_width=True)

            if st.button("Predict"):
                st.subheader("Our Prediction")
                predicted_disease = model_prediction(test_image)
                st.success(f"Model Prediction: **{predicted_disease}**")

                with st.spinner("Generating precaution tips using Gemini AI..."):
                    if predicted_disease.lower() == "tomato___healthy":
                       precaution = "This leaf is healthy. No remedies are required."
                    else:
                     prompt = (
                        f"You are a plant pathologist. Give 2–3 short and technical remedy suggestions "
                        f"for the tomato leaf disease: {predicted_disease}. Keep it clear, practical, "
                        f"and easy for farmers to follow. No long paragraphs."
                    )
                     
                     precaution = get_precaution_from_ai(prompt)
            # try:
            #     response = requests.post(
            #         "http://localhost:5000/precaution",
            #         json={"disease": prompt},
            #         timeout=10
            #     )
            #     if response.status_code == 200:
            #         precaution = response.json().get("Precaution", "Could not parse suggestions.")
            #     else:
            #         precaution = "Could not fetch suggestions right now. Please try again later."
            # except Exception as e:
            #     precaution = f"Gemini service unreachable: {str(e)}"

    st.markdown(f"**Precaution & Treatment Tips for {predicted_disease}:**\n\n{precaution}")               
# Save prediction and precaution to the database
    db_utils.save_prediction(predicted_disease)
    db_utils.save_precaution(predicted_disease, precaution)
        
elif app_mode == "View History":
    st.markdown("##  Prediction History")
    prediction_history = db_utils.get_prediction_history()
    if prediction_history:
        for disease, timestamp in prediction_history:
            st.markdown(f"-  **{disease}** – `{timestamp}`")
    else:
        st.info("No prediction history available.")

    st.markdown("---")
    st.markdown("##  Precaution History")
    precaution_history = db_utils.get_precaution_history()
    if precaution_history:
        for disease, precaution, timestamp in precaution_history:
            st.markdown(f"-  **{disease}** – `{timestamp}`  ↳ _{precaution}_")
    else:
        st.info("No precaution history available.")

