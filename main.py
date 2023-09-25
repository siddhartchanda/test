import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create a Streamlit app
st.title("Travel Insurance Prediction")

# Input fields for user interaction
Agency = st.text_input("Agency")
Agency_Type = st.text_input("Agency Type")
Distribution_Channel = st.text_input("Distribution Channel")
Product_Name = st.text_input("Product Name")
Duration = st.number_input("Duration", min_value=0)
Destination = st.text_input("Destination")
Gender = st.selectbox("Gender", ["M", "F", "None"])
Age = st.number_input("Age", min_value=0)

# Button to trigger prediction
if st.button("Predict"):
    # Load the dataset
    data = pd.read_csv('TRAVEL.csv')

    # Data Preprocessing
    # Handle missing values (if any)
    data.dropna(inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Gender', 'Destination']

    for col in categorical_columns:
        # Use handle_unknown='use_encoded_value' to handle unseen labels
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])

    # Split the data into features (X) and target variable (y)
    if 'Claim' in data.columns:
        data['Claim'] = label_encoder.fit_transform(data['Claim'])
        y = data['Claim']
        X = data.drop(columns=['Net Sales', 'Commision (in value)', 'Claim'])
    else:
        raise ValueError("The 'Claim' column is missing in the dataset.")

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    # Create a Sequential model
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model (you should replace this with your model training)
    # model.fit(X_train, y, epochs=10, batch_size=32)

    # Sample input data (you can modify this to match your requirements)
    sample_data = {
        'Agency': [Agency],
        'Agency Type': [Agency_Type],
        'Distribution Channel': [Distribution_Channel],
        'Product Name': [Product_Name],
        'Duration': [Duration],  # Example duration in days
        'Destination': [Destination],  # Example destination
        'Gender': [Gender],  # You can specify gender if it's known, or use None
        'Age': [Age]  # Example age
    }

    # Handle previously unseen labels in other categorical columns in the sample input data
    for col in categorical_columns:
        valid_classes = set(label_encoder.classes_)
        sample_data[col] = [x if x in valid_classes else -1 for x in sample_data[col]]  # Use -1 for unseen labels

    # Define numerical columns
    numerical_columns = X.columns.tolist()

    # Create a DataFrame from the sample data
    sample_df = pd.DataFrame(sample_data)

    # Standardize numerical features in the sample data using the same scaler
    sample_df[numerical_columns] = scaler.transform(sample_df[numerical_columns])

    # Make a prediction using the trained model (you should replace this with your model prediction)
    # prediction = model.predict(sample_df)
    predicted_claim = 'Yes'  # Replace with your actual prediction logic

    # Display the prediction result
    st.write(f"Predicted Claim: {predicted_claim}")
