import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
import database as db
import plotly.express as px
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from validate_email_address import validate_email



data_categories = ["Name", "Email", "Address",
                   "Phone Number", "Alternative Phone Number"]
expenses = ["Gender", "Age", "Years Of Education", "Socioeconomic Status", "Mini Mental Stage Examination",
            "Clinical Dementia Rating", "Estimated Total Intracranial Volume", "Normalized Whole-Brain Volume","Auto Scaling Factor"]
edata = ["Email"]
prediction = ["Result"]
page_title = "Brain Twist Odyssey"
page_icon = "./Logo.png"
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- NAVIGATION MENU ---

selected = option_menu(
    menu_title=None,
    options=["Informations","Data Entry", "Prediction",
             "Data Visualization", "Result Visualization"],
    # https://icons.getbootstrap.com/
    icons=["info-circle-fill","pencil-fill", "search-heart-fill", "bar-chart-fill","card-checklist"],
    orientation="horizontal",
)

if selected == "Informations":
    st.header("How To Use")
    with st.expander("About"):
        st.write("I am sorry, but the given text cannot be rewritten in a formal style as it is not a sentence or a phrase that can be expanded or modified to convey a formal tone. Please provide a complete sentence or a paragraph that needs to be rewritten in a formal style.")
        
    video_url = "./video.mp4"
    st.video(video_url)
    

# Define regular expressions for email, name, address, and phone number validation
email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$"
name_pattern = r"^[A-Za-z\s]*$"
address_pattern = r"^[A-Za-z0-9\s,.'-]*$"
phone_pattern = r"^\d{10}$"

if selected == "Data Entry":
    with st.form("entry_form", clear_on_submit=True):
        with st.expander("User Information"):
            
            # Name validation (required)
            user_input = st.text_input("Name:", key="name")
            if not re.match(name_pattern, user_input) or not user_input:
                st.warning("Name is required, and it must contain only letters and spaces.")
                
            # Email validation
            user_input = st.text_input("Email:", key="email")
            if not re.match(email_pattern, user_input):
                st.warning("Please enter a valid email address.")


            # Phone number validation
            user_input = st.text_input("Phone Number:", key="phone")
            if not re.match(phone_pattern, user_input):
                st.warning("Please enter a valid 10-digit phone number.")

            # Alternate Phone Number validation
            user_input = st.text_input("Alternate Phone Number:", key="alternate_phone")
            if not re.match(phone_pattern, user_input):
                st.warning("Please enter a valid 10-digit alternate phone number.")

            # Ensure Phone Number and Alternate Phone Number don't match

            # Address validation (required)
            user_input = st.text_input("Address:", key="address")
            if not re.match(address_pattern, user_input) or not user_input:
                st.warning("Address is required, and it must contain only letters, numbers, and common characters.")
            if st.session_state["phone"] == st.session_state["alternate_phone"]:
                st.warning("Phone Number and Alternate Phone Number must be different.")

        with st.expander("User Data"):
            for expense in expenses:
                if expense in ['Clinical Dementia Rating', 'Normalized Whole-Brain Volume','Auto Scaling Factor']:
                    user_input = st.number_input(f"{expense}:", min_value=0.0, step=0.01, key=expense)
                else:
                    user_input = st.number_input(f"{expense}:", min_value=0, format="%i", step=1, key=expense)
                if user_input is None:
                    st.warning(f"{expense} is required. Please enter a valid value.")
        submitted = st.form_submit_button("Save Data")

    if submitted:
        validation_failed = False
        data_values = {}
        expenses_values = {}

        # User Information Validation
        for field in ["email", "name", "address", "phone", "alternate_phone"]:
            user_input = st.session_state[field]
            if field in ["name", "address"] and (not re.match(name_pattern, user_input) or not user_input):
                st.warning(f"{field.capitalize()} is required and must be in the correct format.")
                validation_failed = True
            elif field == "email" and (not user_input or not re.match(email_pattern, user_input)):
                st.warning("Email is required and must be in the correct format.")
                validation_failed = True
            elif field in ["phone", "alternate_phone"] and (not re.match(phone_pattern, user_input) or st.session_state["phone"] == st.session_state["alternate_phone"]):
                st.warning("Please enter a valid 10-digit phone number and ensure it's different from the alternate phone number.")
                validation_failed = True
            else:
                data_values[field] = user_input

        # User Data Validation
        for expense in expenses:
            user_input = st.session_state[expense]
            if user_input is None:
                st.warning(f"{expense} is required. Please enter a valid value.")
                validation_failed = True
            else:
                expenses_values[expense] = user_input

        if not validation_failed:
            db.insert_period(data_values, expenses_values)
            st.success("Data saved!")





if selected == "Data Visualization":
    st.header("Data Visualization")
    # Fetch data from the database
    df = db.fetch_all_periods()
    if df is not None:
        st.subheader("Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        fig_gender = px.bar(x=gender_counts.index, y=gender_counts.values, labels={
                            'x': 'Gender', 'y': 'Count'})
        st.plotly_chart(fig_gender)

        # Example: Age distribution
        st.subheader("Age Distribution")
        fig_age = px.histogram(df, x='Age', nbins=20)
        st.plotly_chart(fig_age)

        st.subheader("Clinical Dementia Rating")
        fig_age = px.histogram(df, x='Clinical Dementia Rating', nbins=20)
        st.plotly_chart(fig_age)

        # Create more visualizations for other features if needed
        st.subheader(
            "Clinical Dementia Rating vs Socioeconomic Status (Scatter Plot)")
        scatter_plot = px.scatter(df, x='Clinical Dementia Rating', y='Socioeconomic Status',
                                  title="Clinical Dementia Rating vs Socioeconomic Status")
        scatter_plot.update_layout(title_x=0.25)
        st.plotly_chart(scatter_plot)

        st.subheader(
            "Clinical Dementia Rating vs Years Of Education (Scatter Plot)")
        scatter_plot = px.scatter(df, x='Clinical Dementia Rating', y='Years Of Education',
                                  title="Clinical Dementia Rating vs Years Of Education")
        scatter_plot.update_layout(title_x=0.25)
        st.plotly_chart(scatter_plot)

        st.subheader("Clinical Dementia Rating vs Gender (Scatter Plot)")
        scatter_plot = px.scatter(df, x='Clinical Dementia Rating',
                                  y='Gender',  title="Clinical Dementia Rating vs Gender")
        scatter_plot.update_layout(title_x=0.28)
        st.plotly_chart(scatter_plot)

        st.subheader("Clinical Dementia Rating vs Age (Scatter Plot)")
        bar_plot = px.bar(df, x='Clinical Dementia Rating',
                          y='Age',  title="Clinical Dementia Rating vs Age")
        bar_plot.update_layout(title_x=0.28)
        st.plotly_chart(bar_plot)

        st.subheader(
            "Clinical Dementia Rating vs Mini Mental Stage Examination (Bar Plot)")
        bar_plot = px.bar(df, x='Clinical Dementia Rating', y='Mini Mental Stage Examination',
                          title="Clinical Dementia Rating vs Mini Mental Stage Examination")
        bar_plot.update_layout(title_x=0.28)
        st.plotly_chart(bar_plot)

        st.subheader(
            "Clinical Dementia Rating vs Estimated Total Intracranial Volume (Bar Plot)")
        bar_plot = px.bar(df, x='Clinical Dementia Rating', y='Estimated Total Intracranial Volume',
                          title="Clinical Dementia Rating vs Estimated Total Intracranial Volume")
        bar_plot.update_layout(title_x=0.28)
        st.plotly_chart(bar_plot)

        st.subheader(
            "Clinical Dementia Rating vs Normalized Whole-Brain Volume (Bar Plot)")
        bar_plot = px.bar(df, x='Clinical Dementia Rating', y='Normalized Whole-Brain Volume',
                          title="Clinical Dementia Rating vs Normalized Whole-Brain Volume")
        bar_plot.update_layout(title_x=0.28)
        st.plotly_chart(bar_plot)

    else:
        st.warning("No data available for visualization. Please add data.")


if selected == "Prediction":
    st.header("Prediction")
    data = pd.read_csv("./Dataset/Dementia_Detection_clead_data.csv")
    y = data['Group']
    x = data.drop('Group', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    x_train_sc = sc.fit_transform(x_train)
    x_test_sc = sc.transform(x_test)

    model = Sequential()
    model.add(Dense(units=6, kernel_initializer='he_uniform',
                    activation='relu', input_dim=9))
    model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
    model.add(
        Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Load the trained model weights
    model.load_weights("DementiaDetection_DL_Model.h5")

    # Define class labels
    class_labels = ['Non-Demented', 'Demented']

    # Function to make predictions

    def predict_dementia(features):
        # Preprocess the features
        processed_features = sc.transform([features])

        # Make predictions
        prediction = model.predict(processed_features)[0]

        # Get the predicted class label
        predicted_label = class_labels[int(np.round(prediction))]

        return predicted_label

    # Streamlit app code

    def main():
        
        def is_valid_email(email):
            if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return True
            return False
        
        with st.expander("User Email"):
            for e in edata:
                email = st.text_input(f"{e}:", key=e)
                if email and not is_valid_email(email):
                    st.error("Please enter a valid email.")
                elif not email:
                    st.warning("Email is required.")
        
        # Feature inputs
        gender = st.radio("Gender", [0, 1])
        age = st.number_input("Age", min_value=0)
        educ = st.number_input("EDUC", min_value=0)
        ses = st.number_input("SES", min_value=0)
        mmse = st.number_input("MMSE", min_value=0)
        cdr = st.number_input("CDR", min_value=0.0, max_value=1.0, step=0.1)
        etiv = st.number_input("eTIV", min_value=0)
        nwbv = st.number_input("nWBV", min_value=0.0,
                               max_value=1.0, step=0.001)
        ASF = st.number_input("ASF", min_value=0.0,
                               max_value=2.0, step=0.001)
        
        # Make predictions if all features are provided
        # if st.button("Predict"):
        #     edata_value = {edata: st.session_state[edata]
        #                    for edata in edata}
        #     features = [gender, age, educ, ses, mmse, cdr, etiv, nwbv]
        #     predicted_label = predict_dementia(features)
        #     db.insert_perdiction(edata_value, predicted_label, features)
        #     st.write("Predicted Label:", predicted_label)
        
        if st.button("Predict") and email and is_valid_email(email):
            edata_value = {edata: st.session_state[edata] for edata in edata}
            features = [gender, age, educ, ses, mmse, cdr, etiv, nwbv,ASF]
            predicted_label = predict_dementia(features)
            db.insert_perdiction(edata_value, predicted_label, features)
            st.write("Predicted Label:", predicted_label)

        # d0msktwripg_p6T7mVAf28nxSB1HnEEcrrXUpknPAaES
    # Run the app
    if __name__ == '__main__':
        main()


if selected == "Result Visualization":
    st.header("Result Visualization")
    prediction_df = db.fetch_all_predictions()
    if prediction_df is not None:
        # st.write(prediction_df)

        # Visualize predictions, e.g., using a bar chart
        st.subheader("Predicted Labels Distribution")
        prediction_counts = prediction_df['Prediction'].value_counts()
        st.bar_chart(prediction_counts)
        
        
        
            