import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import streamlit as st
def main():
    app=Flask(__name__,template_folder='.')
    data=pd.read_csv('TRAVEL.csv')
    #model=pickle.load(open("Sid.pkl",'rb'))
    #model = keras.models.load_model('travel_insurance_model.h5')
    @app.route('/')
    def index():
        agency_list = sorted(data['Agency'].unique())
        agency_type_list = sorted(data['Agency Type'].unique())
        distribution_channels = sorted(data['Distribution Channel'].unique())
        product_names = sorted(data['Product Name'].unique())
        d_names = sorted(data['Destination'].unique())
        return render_template('index.html', agencies=agency_list, agency_type=agency_type_list, ch=distribution_channels, pn=product_names,gender=['M','F',None],d=d_names)
    
    def res(l):
        # Load the dataset
        df = pd.read_csv("TRAVEL.csv")
    
        # Data Preprocessing
        # Handle missing values (if any)
        df.dropna(inplace=True)
    
        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Gender', 'Destination']
    
        for col in categorical_columns:
            # Use handle_unknown='use_encoded_value' to handle unseen labels
            label_encoder.fit(df[col])
            df[col] = label_encoder.transform(df[col])
    
        # Split the data into features (X) and target variable (y)
        if 'Claim' in df.columns:
            df['Claim'] = label_encoder.fit_transform(df['Claim'])
            y = df['Claim']
            X = df.drop(columns=['Net Sales', 'Commision (in value)', 'Claim'])
        else:
            raise ValueError("The 'Claim' column is missing in the dataset.")
    
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Standardize numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
        # Create a Sequential model
        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
        ])
    
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
        # Sample input data (you can modify this to match your requirements)
        sample_data = {
            'Agency': [l[0]],
            'Agency Type': [l[1]],
            'Distribution Channel': [l[2]],
            'Product Name': [l[3]],
            'Duration': [int(l[4])],  # Example duration in days
            'Destination': [l[5]],  # Example destination
            'Gender': [l[6]],  # You can specify gender if it's known, or use None
            'Age': [int(l[7])]  # Example age
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
        y_pred_prob = model.predict(X_test)
        y_pred = y_pred_prob.argmax(axis=1)
        # Make a prediction using the trained model
        prediction = model.predict(sample_df)
        #print(prediction[0][0])
        # Convert prediction to a binary claim decision (0 or 1)
        predicted_claim = 'Yes' if prediction[0][0] >= 0.00005 else 'No'
        #print(classification_report(y_test, y_pred,zero_division=1))
        # Print the prediction
        return "Predicted Claim: " + predicted_claim
    
    @app.route('/predict', methods=['POST'])
    def predict():
        l=[]
        Agency = request.form.get('agency')
        Agency_Type=request.form.get('t')
        dis=request.form.get('s')
        pn=request.form.get('product')
        d=request.form.get('duration')
        des=request.form.get('destination')
        g=request.form.get('gender')
        a=request.form.get('age')
        #print(location, bhk, bath,sqft)
        input = pd.DataFrame([[Agency,Agency_Type,dis,pn,d,des,g,a]],columns=['Agency','Agency Type','Distribution Channel','Product Name','Duration','Destination','Gender','Age'])
        l.extend([Agency,Agency_Type,dis,pn,d,des,g,a])
        return res(l)
    if __name__=="__main__":
        app.run(debug=True,port=5000)
    # Your Streamlit app code goes here

if __name__ == "__main__":
    main()

    # Stop the Streamlit app when the user hits Ctrl+C
    st.stop()


