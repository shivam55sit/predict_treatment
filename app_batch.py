import streamlit as st
import joblib
import numpy as np
import pandas as pd
from io import BytesIO

# Load the pre-trained pipeline
pipe = joblib.load(open('pipe_logit.joblib', 'rb'))

# Load the dataframe (for select box choices)
df = joblib.load(open('final_dataframe.joblib', 'rb'))

# Display the logo
st.logo('lvpei.jfif', size="large")  

st.title("ROP - 'Treatment Prediction'")

# File upload section for batch processing
uploaded_file = st.file_uploader("Upload an Excel file for batch prediction", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file into a DataFrame
    input_df = pd.read_excel(uploaded_file)

    # Ensure the necessary columns are present in the uploaded file
    required_columns = ['Birth_Weight', 'Resp_Distress', 'Septicemia', 'Blood_Transfusion', 'NVI', 
                        'GA_Days', 'PMA_Days', 'NICU_rating', 'Plus', 'Zone', 'Stage']
    
    if all(col in input_df.columns for col in required_columns):
        # Preprocess the inputs from the file
        input_df['Resp_Distress'] = input_df['Resp_Distress'].apply(lambda x: 1 if x == 'Yes' else 0)
        input_df['Septicemia'] = input_df['Septicemia'].apply(lambda x: 1 if x == 'Yes' else 0)
        input_df['Blood_Transfusion'] = input_df['Blood_Transfusion'].apply(lambda x: 1 if x == 'Yes' else 0)
        input_df['NVI'] = input_df['NVI'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Handle NaN values for zone, stage, and plus
        input_df['NICU_rating'].fillna(1, inplace=True)
        input_df['Zone'].fillna(3, inplace=True)  # Default zone is set to 3
        input_df['Stage'].fillna(1, inplace=True)  # Default stage is set to 1
        input_df['Plus'].fillna('no plus', inplace=True)

        # Make predictions for the entire batch
        y_prob_batch = pipe.predict_proba(input_df)

        # Get predicted classes and confidence scores
        predicted_classes = np.argmax(y_prob_batch, axis=1)
        confidence_scores = np.max(y_prob_batch, axis=1)

        # Add predictions and confidence to the DataFrame
        input_df['Predicted_Treatment'] = np.where(predicted_classes == 1, 'Treatment Required', 'No Treatment')
        input_df['Confidence'] = confidence_scores * 100

        # Display the results with indices as a table
        st.write("Batch Predictions")
        st.dataframe(input_df[['Predicted_Treatment', 'Confidence']])  # Display table with indices

        # Option to download the results as an Excel file
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            processed_data = output.getvalue()
            return processed_data

        result_excel = convert_df_to_excel(input_df)

        st.download_button(label="Download Predictions",
                           data=result_excel,
                           file_name="batch_predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error(f"The uploaded file does not contain the required columns: {', '.join(required_columns)}")
else:
    st.write("Or manually input values for a single prediction.")

    # Input fields for manual prediction (single instance)
    birth_weight = st.number_input('Birth Weight (Grams)', value=1000)
    Resp_Distress = st.selectbox('Respiratory Distress', ['No', 'Yes'])
    Septicemia = st.selectbox('Septicemia', ['No', 'Yes'])
    Blood_Transfusion = st.selectbox('Blood Transfusion', ['No', 'Yes'])
    nvi = st.selectbox('NVI', ['No', 'Yes'])
    GA_Days = st.number_input('Gestation Period (GA) in Days', value=240)
    PMA_Days = st.number_input('Postmenstrual Age (PMA) in Days', value=260)
    nicu = st.selectbox('NICU Rating', df['NICU_rating'].unique())
    plus = st.selectbox('Plus', df['Plus'].unique())
    zone = st.selectbox('Zone', df['Zone'].unique())
    stage = st.selectbox('Stage', df['Stage'].unique())

    # Preprocessing the input when 'Predict' button is pressed
    if st.button('Predict'):
        # Convert categorical inputs to numeric
        Resp_Distress = 1 if Resp_Distress == 'Yes' else 0
        Septicemia = 1 if Septicemia == 'Yes' else 0
        Blood_Transfusion = 1 if Blood_Transfusion == 'Yes' else 0
        nvi = 1 if nvi == 'Yes' else 0

        # Handle NaN values for zone, stage, and plus
        nicu = 1 if pd.isna(nicu) else nicu 
        zone = 3 if pd.isna(zone) else zone  # Default zone is set to 3
        stage = 1 if pd.isna(stage) else stage  # Default stage is set to 1
        plus = 'no plus' if pd.isna(plus) else plus  # Default plus is set to 'no plus'

        # Create a DataFrame from the input values
        input_data = pd.DataFrame({
            'NICU_rating': [nicu],
            'Birth_Weight': [birth_weight],
            'Resp_Distress': [Resp_Distress],
            'Septicemia': [Septicemia],
            'Blood_Transfusion': [Blood_Transfusion],
            'GA_Days': [GA_Days],
            'PMA_Days': [PMA_Days],
            'Stage': [stage],
            'Plus': [plus],
            'Zone': [zone],
            'NVI': [nvi]
        })

        # Make the prediction (probabilities) using predict_proba
        y_prob = pipe.predict_proba(input_data)

        # Predicted class (0: No treatment, 1: Treatment required)
        predicted_class = np.argmax(y_prob, axis=1)[0]

        # Confidence score (max probability in each row)
        confidence_score = np.max(y_prob, axis=1)[0]

        # Display the result with confidence
        if predicted_class == 0:
            st.title(f"The treatment is not required.")
            st.write(f"Confidence: {confidence_score*100:.2f}% ")
            st.image("thumbs-up.webp")
        else:
            st.title(f"The treatment is required. Please consult Dr. Tapas !")
            st.write(f"Confidence: {confidence_score*100:.2f}% ")
            st.image("thumb-down.jpg")
