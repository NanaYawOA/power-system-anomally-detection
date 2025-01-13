⚡ Power System Anomaly Detection App




🌟 Overview
The Power System Anomaly Detection App is a machine learning-powered application designed to detect and visualize anomalies in power systems in real time. This tool provides operators with critical insights to ensure system reliability and prevent failures.

🚀 Features
Real-Time Anomaly Detection:
Detects anomalies using a Random Forest binary classification model.
Supports unsupervised learning with autoencoders for anomaly detection.
Simulated Real-Time Data Stream:
Generates data dynamically based on historical system data for testing and demonstrations.
Dynamic Visualization:
Live graphs showing critical features such as bus voltage, current magnitude, and system impedance.
Event Log:
Logs detected anomalies with timestamps and affected feature values.
Interactive Dashboard:
Built with Flask and Chart.js for an intuitive user experience.
🛠️ Technologies
This app is built using:

Backend: Flask
Machine Learning: Scikit-learn, Autoencoders
Data Visualization: Matplotlib, Chart.js
Data Handling: Pandas, NumPy
APIs:
Custom APIs for real-time streaming and anomaly detection
Deployment: Localhost or cloud platforms like Heroku and AWS
📂 Repository Structure
bash
Copy code
.
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── templates/
│   ├── dashboard.html        # Dashboard for visualization
├── static/
│   ├── styles.css            # Custom CSS for styling
├── models/
│   ├── random_forest_model.pkl # Trained Random Forest model
│   ├── autoencoder_model.pkl  # Trained Autoencoder model
├── data/
│   ├── powersys_data.csv      # Historical power system data
├── .env                      # Environment variables (not included in GitHub)
🚀 Getting Started
Prerequisites
Python 3.8 or later
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Running the App
Start the Flask server:
bash
Copy code
python app.py
Access the app: Open your browser and navigate to http://127.0.0.1:5000.
⚙️ Features Demo
📊 Real-Time Visualization
The app dynamically visualizes key power system metrics, including:

Bus voltage phase angle
Current phase magnitude
Impedance values
Frequency and frequency delta
🔔 Anomaly Detection
The app uses:

Supervised Learning: Random Forest model for binary anomaly detection.
Unsupervised Learning: Autoencoders to detect deviations from normal system behavior.
📋 Event Logging
Automatically logs all detected anomalies with:
Timestamp
Anomaly description
Affected feature values
🧑‍💻 Development Workflow
Training the Models
Data Preprocessing:

Load the dataset from data/powersys_data.csv.
Separate features and target (marker column).
Standardize the data using StandardScaler.
Train the Models:

Random Forest Binary Classifier:
python
Copy code
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
Autoencoder (Unsupervised):
python
Copy code
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

autoencoder = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(X_train.shape[1], activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32)
Save the Models:

python
Copy code
import joblib
joblib.dump(rf_model, "models/random_forest_model.pkl")
autoencoder.save("models/autoencoder_model.pkl")
🌟 Future Enhancements
Support multi-class anomaly detection.
Extend visualization for additional metrics (e.g., power factor, harmonic distortion).
Integrate IoT sensors for real-time monitoring.
Add predictive maintenance capabilities.
🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository.
Create a new branch (feature-name).
Commit your changes.
Push to your fork and submit a pull request.
📄 License
This project is licensed under the MIT License.

💡 Acknowledgments
Scikit-learn
TensorFlow
Matplotlib
