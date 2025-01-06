import streamlit as st 
import pandas as pd
import requests
import plotly.express as px
from io import BytesIO

# Define the API endpoint
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

# Single text input for prediction
st.header("Single Text Prediction")

input_text = st.text_input("Enter text for sentiment prediction:")

if st.button("Predict Single Text"):
    if input_text:
        try:
            response = requests.post(prediction_endpoint, json={"text": input_text})
            if response.status_code == 200:
                prediction = response.json()['prediction']
                st.write(f"Prediction: {prediction}")
            else:
                st.write("Error: Could not retrieve prediction.")
        except requests.exceptions.RequestException:
            st.write("Error: Failed to connect to prediction service.")
    else:
        st.write("Please enter some text for prediction.")

# Bulk prediction from CSV upload
st.header("Bulk Prediction from CSV")
uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction",
    type="csv",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if st.button("Predict and Visualize"):
        predictions = []
        
        for index, row in df.iterrows():
            text = row['text']  # Assuming your CSV has a column named 'text'
            try:
                response = requests.post(prediction_endpoint, json={"text": text})
                if response.status_code == 200:
                    prediction = response.json()['prediction']
                    predictions.append(prediction)
                else:
                    predictions.append("Error")
            except requests.exceptions.RequestException:
                predictions.append("Error")
        
        df['Prediction'] = predictions
        
        # Count the occurrences of each sentiment
        sentiment_counts = df['Prediction'].value_counts()
        
        # Create a pie chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution'
        )
        st.plotly_chart(fig)
        
        # Create a bar chart
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title='Sentiment Distribution',
            labels={'x': 'Sentiment', 'y': 'Count'}
        )
        st.plotly_chart(fig_bar)
        
        # Display the dataframe
        st.write(df)
        
        # Provide download option for the results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="sentiment_predictions.csv",
            mime="text/csv",
        )

else:
    st.write("Please upload a CSV file to begin.")
