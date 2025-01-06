# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import save_model

# Load the data
building_data = pd.read_excel("building_count.xlsx")
temperature_data = pd.read_excel("temperature_data.xlsx")

# Ensure the data aligns by AOI
assert all(building_data['aoi'] == temperature_data['aoi']), "AOI mismatch between datasets"

aois = building_data['aoi']

# Drop AOI column to work with numerical data
building_data = building_data.drop(columns=['aoi'])
temperature_data = temperature_data.drop(columns=['aoi'])

# Convert data to numpy arrays
building_values = building_data.values
temperature_values = temperature_data.values

# Normalize the data
scaler_building = MinMaxScaler()
building_values_scaled = scaler_building.fit_transform(building_values)

scaler_temperature = MinMaxScaler()
temperature_values_scaled = scaler_temperature.fit_transform(temperature_values)

# Combine building counts and temperature data for the model
combined_data = np.concatenate([building_values_scaled, temperature_values_scaled], axis=1)

# Prepare sequences for LSTM
sequence_length = 5  # Use 5 years of data to predict the next year
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, :])
    return np.array(X), np.array(y)

X, y = create_sequences(combined_data, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, combined_data.shape[1])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(y.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
epochs = 50
batch_size = 16
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

# Save the model
model.save("lstm_model.h5")

# Make predictions
predictions = model.predict(X_test)

# Reverse scaling for interpretation
y_test_unscaled = scaler_building.inverse_transform(y_test[:, :building_values.shape[1]])
predictions_unscaled = scaler_building.inverse_transform(predictions[:, :building_values.shape[1]])

# Identify AOIs with increased building counts and decreased temperature
result = []
for i in range(len(aois)):
    if np.all(predictions_unscaled[:, i] > y_test_unscaled[:, i]):  # Building counts increased
        temp_diff = predictions[:, building_values.shape[1]:] - y_test[:, building_values.shape[1]:]
        if np.all(temp_diff < 0):  # Temperature decreased
            result.append(aois[i])

# Save results to an Excel file
result_df = pd.DataFrame({"AOI": result})
result_df.to_excel("aoi_results.xlsx", index=False)

# Generate confusion matrix (assuming binary classification for demo purposes)
# Here we classify based on whether the building count increased and temperature decreased
true_labels = []
pred_labels = []

for i in range(len(X_test)):
    true_labels.append(1 if np.all(y_test[i, :building_values.shape[1]] > X_test[i, -1, :building_values.shape[1]]) and
                           np.all(y_test[i, building_values.shape[1]:] < X_test[i, -1, building_values.shape[1]:]) else 0)
    pred_labels.append(1 if np.all(predictions[i, :building_values.shape[1]] > X_test[i, -1, :building_values.shape[1]]) and
                           np.all(predictions[i, building_values.shape[1]:] < X_test[i, -1, building_values.shape[1]:]) else 0)

conf_matrix = confusion_matrix(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels)

# Save confusion matrix and classification report
conf_matrix_df = pd.DataFrame(conf_matrix, index=["True Negative", "True Positive"], columns=["Pred Negative", "Pred Positive"])
conf_matrix_df.to_excel("confusion_matrix.xlsx")

with open("classification_report.txt", "w") as f:
    f.write(report)

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")

# Print results
print("AOIs with increased building counts and decreased temperature:")
print(result)


