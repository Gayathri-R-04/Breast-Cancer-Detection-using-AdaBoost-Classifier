### **README for Breast Cancer Classification Project**

---

# **AI Breast Cancer Diagnostic Assistant**

This project implements an **AI-powered Breast Cancer Diagnostic Assistant** leveraging machine learning techniques. It uses the **Breast Cancer Wisconsin (Diagnostic) dataset** to classify tumors as malignant or benign. The project is enhanced with a user-friendly **Streamlit-based web interface**, providing interactive insights and predictions.

---

## **Overview**

Breast cancer remains one of the leading causes of cancer-related deaths worldwide. Early and accurate detection is vital for effective treatment. This project uses the **AdaBoost** algorithm combined with **SMOTE (Synthetic Minority Oversampling Technique)** to enhance classification. The tool integrates features like feature importance visualization, real-time predictions, and advanced performance analysis.

---

## **Features**

### **Key Functionalities:**
- **Interactive Sidebar**: Modify patient feature inputs dynamically.
- **Real-Time Predictions**:
  - Predicts whether the tumor is **Malignant** or **Benign**.
  - Provides prediction **confidence levels**.
- **Visual Performance Metrics**:
  - ROC Curve
  - Precision-Recall Curve
  - Confusion Matrix
  - Detailed Classification Report
- **Feature Insights**:
  - Feature Importance
  - Correlation Heatmap
- **Interactive Design**: Customizable interface with options for background themes.

---

## **Streamlit Application**

The **Streamlit web app** enhances user interaction with features like dynamic sliders for feature input, visualizations, and real-time updates.

### **Steps to Run**:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Breast-Cancer-Diagnostic
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run Streamlit_App.py
   ```
5. Open the app in your browser and explore the diagnostic tool!

---

## **Dependencies**

Install the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `plotly`
- `joblib`
- `streamlit`

Install them via:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly joblib streamlit
```

---

## **Model and Techniques**

### **Machine Learning**:
- **Algorithm**: AdaBoost Classifier
- **Data Preprocessing**:
  - Standardization (mean=0, variance=1)
  - Oversampling with **SMOTE** to handle class imbalance.
- **Evaluation**:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - Precision, Recall, F1-Score

### **Visualization**:
- **Correlation Heatmap**: Understand feature relationships.
- **Gauge Chart**: Display confidence levels.
- **Pie Chart**: Represent probability distribution.

---

## **Performance Highlights**

- **Accuracy**: Achieved with advanced AdaBoost techniques.
- **AUC-ROC**: Indicates the model’s capability to distinguish between classes.
- **Precision-Recall**: Provides trade-offs for imbalanced datasets.

---

## **Interactive Features**

- **Dynamic Input**: Adjust feature sliders to simulate patient data.
- **Real-Time Predictions**: Immediate results with confidence visualization.
- **Background Customization**: Choose from multiple themes for a personalized experience.

---

## **Medical Disclaimer**

**Important Notice**:
- This tool is a screening aid, not a substitute for professional medical diagnosis.
- Always consult a certified medical professional for treatment decisions.

---

## **Developer Team**

| **Name**                   | **Contact**                   | **GitHub**                              | **LinkedIn**                                     |
|----------------------------|-------------------------------|-----------------------------------------|-------------------------------------------------|
| Gayathri R                 | gayathri.22ad@kct.ac.in      | [GitHub](https://github.com/Gayathri-R-04/) | [LinkedIn](https://www.linkedin.com/in/gayathri-r-14rg) |
| Naveen S                  | snaveen8105@gmail.com        | [GitHub](https://github.com/naveeen0308) | [LinkedIn](https://www.linkedin.com/in/naveen-s-a70854268/) |
| B.Krishna Raja Sree        | 22b01a4609@svecw.edu.in      | [GitHub](https://github.com/krishnasree76/) | [LinkedIn](https://www.linkedin.com/in/krishna-raja-sree-bonam-7b6079257/) |
| Joseph Boban               | joseph.dm254031@greatlakes.edu.in | [GitHub](https://github.com/josephboban2000) | [LinkedIn](https://www.linkedin.com/in/josephboban/) |
| Shaik Ayesha Parveen       | ayeshparveen25@gmail.com     | [GitHub](https://github.com/ShaikAyeshaparveen25/) | |


---

## **Future Scope**

- Expand datasets for broader demographic inclusion.
- Incorporate advanced deep learning models for improved accuracy.
- Add multilingual support for global accessibility.

---

**Transforming diagnosis with AI for better healthcare.**
