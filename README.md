# Machine-learning-

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# --- SDG & Problem Statement ---
# SDG: SDG 13: Climate Action
# Problem: Forecasting urban carbon emissions to inform policy decisions.
# ML Approach: Supervised Learning (Neural Network).

# --- 1. Synthetic Data Generation ---
# This section creates a synthetic dataset that mimics real-world urban data.
# The goal is to make the code fully runnable without requiring external files.

np.random.seed(42) # for reproducibility
n_samples = 1000

# Generate features
population = np.random.randint(50000, 2000000, n_samples)
gdp_per_capita = np.random.normal(50000, 15000, n_samples)
industrial_output = np.random.uniform(10, 500, n_samples)
avg_temp = np.random.normal(15, 8, n_samples)
vehicles = np.random.uniform(5000, 500000, n_samples)
public_transport_usage = np.random.uniform(0, 0.6, n_samples)
energy_consumption = np.random.uniform(5000, 50000, n_samples)
month = np.random.randint(1, 13, n_samples)

# Create a target variable (carbon emissions) with a complex relationship
# This formula simulates how different factors influence emissions.
# It includes non-linear terms to make it a suitable problem for a Neural Network.
carbon_emissions = (
    1.2 * (energy_consumption**1.1) +
    0.8 * vehicles +
    0.5 * industrial_output +
    0.001 * population +
    200 * avg_temp +
    (5000 * np.cos(np.pi * month / 6)) - # Seasonal effect
    50000 * public_transport_usage +
    np.random.normal(0, 10000, n_samples) # Add some random noise
)

# Ensure emissions are non-negative
carbon_emissions[carbon_emissions < 0] = 0

# Create a pandas DataFrame
data = pd.DataFrame({
    'population': population,
    'gdp_per_capita': gdp_per_capita,
    'industrial_output': industrial_output,
    'avg_temp': avg_temp,
    'vehicles': vehicles,
    'public_transport_usage': public_transport_usage,
    'energy_consumption': energy_consumption,
    'month': month,
    'carbon_emissions': carbon_emissions
})

print("--- Synthetic Dataset Generated ---")
print(data.head())
print("-" * 35)

# --- 2. Data Preprocessing ---
# This is a critical step to prepare the data for the neural network.

# Define features (X) and target (y)
X = data.drop('carbon_emissions', axis=1)
y = data['carbon_emissions']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
# Scaling is essential for neural networks to ensure all features contribute equally.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data split into training and testing sets.")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")
print("-" * 35)

# --- 3. Build the Neural Network Model ---
# This section defines the architecture of our Multi-layer Perceptron (MLP).
# We'll use Keras, a high-level API for TensorFlow.

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) # Output layer for regression (1 neuron)

# Compile the model
# We use 'adam' optimizer and 'mean_squared_error' as the loss function.
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("--- Neural Network Model Architecture ---")
model.summary()
print("-" * 35)

# --- 4. Train the Model ---
# The model learns the relationship between the features and emissions.

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50, # Number of passes through the entire dataset
    batch_size=32,
    validation_split=0.2, # Use 20% of training data for validation
    verbose=1
)

print("-" * 35)
print("Model training complete.")
print("-" * 35)

# --- 5. Evaluate the Model ---
# We assess the model's performance on the unseen test data.

# Make predictions on the test set
y_pred = model.predict(X_test_scaled).flatten()

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation on Test Data ---")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")
print("-" * 35)

# --- 6. Visualize the Results ---
# Visualizations help us understand the model's performance intuitively.

# Plot the training & validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Carbon Emissions')
plt.xlabel('Actual Emissions')
plt.ylabel('Predicted Emissions')
plt.grid(True)
plt.show()

