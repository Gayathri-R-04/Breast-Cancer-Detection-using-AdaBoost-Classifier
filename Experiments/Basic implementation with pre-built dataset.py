# Import necessary libraries
import pandas as pd
import numpy as np

#You can use any other ML algorithms instead of adaboost

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the breast cancer dataset
data = load_breast_cancer()

# Extract features (X) and labels (y)
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Check the shape of data
print(f"Feature Matrix Shape: {X.shape}")
print(f"Label Vector Shape: {y.shape}")

# Map target labels for clarity: 0 -> malignant, 1 -> benign
label_map = {0: "Malignant", 1: "Benign"}

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the AdaBoost classifier
model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display the confusion matrix with label mapping
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, 
                   index=["Actual Malignant", "Actual Benign"], 
                   columns=["Predicted Malignant", "Predicted Benign"]))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# Predict for new samples (Optional)
sample = X_test[0].reshape(1, -1)  # Select a random sample from test data
prediction = model.predict(sample)
print(f"\nPrediction for sample: {label_map[prediction[0]]}")
