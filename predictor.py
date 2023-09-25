import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

def make_prediction(input_data):
    # Load the dataset
    df = pd.read_csv("TRAVEL.csv")

    # Data Preprocessing
    # ... (rest of the preprocessing code)

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

    # ... (rest of the prediction code)

    return "Yes" if prediction[0][0] >= 0.00005 else "No"
