import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Load and preprocess the dataset
df = pd.read_csv('C:/Users/Hp/Desktop/code_projects/power-system-anomally-detection/datasets/binary/powersys_data_supervised_binary.csv')  # Replace with your dataset path

y_test = df['marker'] if 'marker' in df.columns else None # Labels 
train = df.drop(columns=['marker'], errors='ignore')  # Features

scaler = StandardScaler()
X = scaler.fit_transform(train)


# Define the autoencoder
input_dim = X.shape[1]

param_grid = {
    'encoding_dim': [8, 16, 32],
    'dropout_rate': [0.1, 0.2, 0.3],
    'learning_rate': [1e-3, 1e-4]
}

best_model = None
best_score = float('inf')

# Grid search loop
for params in ParameterGrid(param_grid):
    # Define autoencoder
    autoencoder = Sequential([
        Dense(params['encoding_dim'], activation='relu', input_dim=input_dim),
        Dropout(params['dropout_rate']),
        Dense(8, activation='relu'),
        Dense(params['encoding_dim'], activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])

    # Compile with learning rate
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train the model
    history = autoencoder.fit(
        X, X,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

# Calculate reconstruction error
reconstructed = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - reconstructed), axis=1)

# Set threshold and classify anomalies
threshold = np.percentile(reconstruction_error, 95)
y_pred = (reconstruction_error > threshold).astype(int)

# Example ground truth (if available)

print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)  # Anomalies are now labeled as 1
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.4f}") 
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")