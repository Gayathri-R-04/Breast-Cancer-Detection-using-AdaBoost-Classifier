# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset
data = load_breast_cancer()

# Extract features (X) and labels (y)
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Map target labels for clarity: 0 -> malignant, 1 -> benign
label_map = {0: "Malignant", 1: "Benign"}

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models for comparison
models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
}

# Dictionary to store model accuracies
accuracies = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    
    # Display model performance
    print(f"\n{model_name} Model Accuracy: {accuracy:.2f}")
    
    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, 
                       index=["Actual Malignant", "Actual Benign"], 
                       columns=["Predicted Malignant", "Predicted Benign"]))
    
    # Display classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# Visualization: Model Accuracies
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'orange', 'red'])
plt.title("Comparison of Model Accuracies")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Visualization: Confusion Matrices for Each Model
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Malignant", "Predicted Benign"],
                yticklabels=["Actual Malignant", "Actual Benign"],
                ax=axes[i])
    axes[i].set_title(f"{model_name} Confusion Matrix")
plt.tight_layout()
plt.show()
