# Breast-Cancer-Detection-using-AdaBoost-Classifier


This project implements a **Breast Cancer Classification** system using machine learning techniques. It uses the **Breast Cancer Wisconsin (Diagnostic) dataset** to classify tumors as either malignant or benign. The project has two versions, with **Version 2** integrating a web-based interface using **Streamlit**.

## **Overview**

Breast cancer is one of the most common types of cancer affecting women worldwide. Early detection and accurate classification of breast tumors as benign or malignant are crucial for effective treatment. This project demonstrates how machine learning models can assist in such classification tasks. It uses **AdaBoost** as the classifier and allows real-time predictions via a web interface (in Version 2).

## **Features**

- Uses **AdaBoostClassifier** to classify breast cancer as malignant or benign
- Displays the confusion matrix and classification report for performance evaluation
- Predicts tumor type for new input data samples
- Preprocesses the dataset with **standardization** (mean = 0, variance = 1)
- **Version 2** includes a web-based interface using **Streamlit**
- Displays feature importance and correlation heatmap for better insights

## **Dependencies**

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `streamlit` (required for Version 2)

You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib streamlit
```

## **Version 1**
In Version 1, the model is a simple command-line Python script that trains the AdaBoostClassifier on the Breast Cancer Wisconsin (Diagnostic) dataset. The results include accuracy, a confusion matrix, and a classification report.

# **How to Run (Version 1)**

- Clone this repository or download the breast_cancer_classification_v1.py file to your local machine.
- Install the required dependencies using pip install -r requirements.txt.
- Run the Python script:
```bash
python breast_cancer_classification_v1.py
```
- The script will load the dataset, preprocess the features, train the AdaBoost model, and display the evaluation metrics such as accuracy, confusion matrix, and classification report.

## **Version 2**
Version 2 enhances the project by integrating Streamlit to provide an interactive web interface. Users can adjust feature values through the sidebar and receive real-time predictions, including prediction probabilities. The model also displays important features, a correlation heatmap, and actual vs predicted values with colored visuals.

# **How to Run (Version 2)**
- Clone this repository or download the breast_cancer_classification_v2.py file to your local machine.
- Install the required dependencies using pip install -r requirements.txt.
- Run the Streamlit app:
```bash
streamlit run breast_cancer_classification_v2.py
```
- A web browser will open the Streamlit app. You can adjust input features via the sidebar, and the model will display real-time predictions, feature importance, and correlation heatmaps.

## **Features in Version 2**
- **Interactive Sidebar:** Users can modify feature values via sliders to simulate new input data.
- **Real-time Predictions:** The app predicts if the tumor is malignant or benign and shows prediction probabilities.
- **Model Performance:** Displays accuracy, confusion matrix, and classification report.
- **Visualization:** Shows feature importance and a correlation heatmap for a better understanding of the model.
- **Actual vs Predicted Table:** Displays actual vs predicted values, with color-coding for correct and incorrect predictions.
