import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
import seaborn as sns
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    classification_report, precision_recall_curve
)

class BreastCancerDetectionApp:
    def __init__(self):
        st.set_page_config(
            page_title="AI Breast Cancer Diagnostic Assistant", 
            page_icon="🩺", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
        self.setup_page()
        self.load_resources()

    def setup_page(self):
        # Option to change background color
        bg_color = st.sidebar.selectbox("Select Background Color", ["Grey", "White", "Light Blue", "Light Green", "Light Yellow", "Light Grey", "Light Pink"])
        if bg_color == "Grey":
            bg_color_code = "#F4F6F7"
        elif bg_color == "White":
            bg_color_code = "#FFFFFF"
        elif bg_color == "Light Blue":
            bg_color_code = "#E3F2FD"
        elif bg_color == "Light Green":
            bg_color_code = "#E8F5E9"
        elif bg_color == "Light Yellow":
            bg_color_code = "#FFFDE7"
        elif bg_color == "Light Grey":
            bg_color_code = "#F5F5F5"
        elif bg_color == "Light Pink":
            bg_color_code = "#FCE4EC"

        st.markdown(f"""
        <style>
        .main-header {{
            background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .stApp {{
            background-color: {bg_color_code};
        }}
        .metric-container {{
            background: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            border: 1px solid #E0E0E0;
        }}
        .metric-container:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .prediction-card {{
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .developer-section {{
            background-color: #F8F9F9;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 24px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding: 20px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }}
        </style>
        """, unsafe_allow_html=True)

    def load_resources(self):
        @st.cache_resource
        def load_model_data(_self):
            model_url = "adaboost_model_with_smote_on_original_data.pkl"
            scaler_url = "scaler.pkl"
            data_url = "data_metadata_with_smote_on_original_data.pkl"
            feature_names_url = "feature_names.pkl"
    
            # Helper function to load a pickle file from a URL
            def load_pickle_from_url(url):
                response = requests.get(url)
                response.raise_for_status()  # Ensure the request was successful
                return joblib.load(BytesIO(response.content))  # Load from in-memory bytes
    
            # Load all files
            model = load_pickle_from_url(model_url)
            scaler = load_pickle_from_url(scaler_url)
            data = load_pickle_from_url(data_url)
            feature_names = load_pickle_from_url(feature_names_url)
    
            return model, scaler, data, feature_names

        self.model, self.scaler, self.data, self.feature_names = load_model_data(self)
        self.df = pd.DataFrame(
            self.data["X_data"], 
            columns=self.feature_names
        )
        self.label_map = self.data["label_map"]

    def create_gauge_chart(self, value, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#2ECC71" if value > 0.5 else "#E74C3C"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(231, 76, 60, 0.2)'},
                    {'range': [50, 100], 'color': 'rgba(46, 204, 113, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value * 100
                }
            }
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def advanced_prediction_visualization(self, prediction, prediction_proba):
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence = prediction_proba[0].max()
            pred_class = "Malignant" if prediction[0] == 1 else "Benign"
            
            st.markdown(f"### Prediction: {pred_class}")
            gauge_fig = self.create_gauge_chart(
                confidence,
                f"Confidence Level for {pred_class}"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            if confidence > 0.8:
                st.success("🎯 High Confidence Prediction")
                st.info("💡 Recommended Action: Schedule follow-up with specialist")
            elif confidence > 0.6:
                st.warning("⚠️ Moderate Confidence Prediction")
                st.info("💡 Recommended Action: Additional testing suggested")
            else:
                st.error("⚠️ Low Confidence Prediction")
                st.info("💡 Recommended Action: Comprehensive medical assessment required")

        with col2:
            st.markdown("### Probability Distribution")
            
            prob_data = {
                'Class': ['Malignant', 'Benign'],
                'Probability': [prediction_proba[0][1], prediction_proba[0][0]]
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=prob_data['Class'],
                values=prob_data['Probability'],
                hole=.7,
                marker_colors=['#E74C3C', '#2ECC71'],
                textinfo='label+percent',
                textfont_size=14,
                hovertemplate="<b>%{label}</b><br>" +
                            "Probability: %{percent}<br>" +
                            "<extra></extra>"
            )])
            
            fig.update_layout(
                showlegend=False,
                annotations=[dict(
                    text='Prediction<br>Confidence',
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )],
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    def performance_metrics(self):
        st.header("🔍 Model Performance Analysis")
        
        tabs = st.tabs([
            "📈 ROC Curve",
            "📊 Precision-Recall",
            "🎯 Confusion Matrix",
            "📑 Classification Report"
        ])
        
        with tabs[0]:
            self.roc_curve_visualization()
            st.markdown("""
            **Understanding ROC Curve:**
            - Plots True Positive Rate vs False Positive Rate
            - AUC closer to 1.0 indicates better model performance
            - Diagonal line represents random prediction
            """)
        
        with tabs[1]:
            self.precision_recall_curve()
            st.markdown("""
            **Understanding Precision-Recall Curve:**
            - Shows trade-off between precision and recall
            - Higher curve indicates better model performance
            - Useful for imbalanced classification problems
            """)
        
        with tabs[2]:
            self.confusion_matrix_heatmap()
            st.markdown("""
            **Understanding Confusion Matrix:**
            - True Positives: Correctly identified positive cases
            - True Negatives: Correctly identified negative cases
            - False Positives: Incorrectly identified positive cases
            - False Negatives: Incorrectly identified negative cases
            """)
        
        with tabs[3]:
            st.markdown("### Detailed Classification Metrics")
            st.code(self.data["classification_report"], language='text')
            st.markdown("""
            **Key Metrics Explained:**
            - Precision: Ratio of correct positive predictions
            - Recall: Ratio of actual positives correctly identified
            - F1-Score: Harmonic mean of precision and recall
            - Support: Number of samples for each class
            """)

    def roc_curve_visualization(self):
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.3f})',
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgb(52, 152, 219)', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            line=dict(color='rgb(189, 195, 199)', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='x unified',
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def precision_recall_curve(self):
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name='Precision-Recall curve',
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(color='rgb(46, 204, 113)', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            hovermode='x unified',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def confusion_matrix_heatmap(self):
        conf_matrix = self.data["conf_matrix"]
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Benign', 'Predicted Malignant'],
            y=['Actual Benign', 'Actual Malignant'],
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def interactive_feature_selection(self):
        features = {}
        for feature in self.feature_names:
            min_val = self.df[feature].min()
            max_val = self.df[feature].max()
            mean_val = self.df[feature].mean()
            features[feature] = st.sidebar.slider(
                feature, float(min_val), float(max_val), float(mean_val),
                help=f"""
                Range: {min_val:.2f} - {max_val:.2f}
                Mean: {mean_val:.2f}
                """
            )
        return pd.DataFrame(features, index=[0])

    def run(self):
        st.markdown("<div class='main-header'><h1>🩺 AI Breast Cancer Diagnostic Assistant</h1></div>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        metrics = [
            ("🎯 Model Accuracy", f"{self.data['accuracy']:.2%}", "Overall predictive performance of the model"),
            ("📊 Total Samples", len(self.data["y_test"]), "Size of the testing dataset"),
            ("🤖 Model Algorithm", "AdaBoost", "Advanced ensemble learning technique")
        ]
        
        for col, (title, value, help_text) in zip(cols, metrics):
            with col:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(title, str(value), help=help_text)
                st.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.markdown("## 🔬 Patient Feature Input")
        input_df = self.interactive_feature_selection()
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)
        prediction_proba = self.model.predict_proba(input_scaled)

        st.header("🔬 Diagnostic Prediction")
        self.advanced_prediction_visualization(prediction, prediction_proba)

        self.performance_metrics()

        st.markdown("---")
        st.markdown("### ⚕️ Medical Disclaimer")
        st.warning("""
        **Important Notice:**
        - This AI tool is designed for screening purposes only
        - Not a replacement for professional medical diagnosis
        - All results should be interpreted by qualified healthcare professionals
        - Always consult with medical experts for proper diagnosis and treatment
        """)

        st.markdown("---")
        st.markdown("### 👨‍💻 Developer Information")

        # Create two columns for developer information
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("🧑🏻‍💻 Naveen S")
            st.markdown("📞 Contact: +91 7448863062")
            st.markdown("📧 Email: snaveen8105@gmail.com")
            st.markdown("Github: [Click here!](https://github.com/naveeen0308/Breast-cancer-detection)")
            st.markdown("Linkedin: [Click here!](https://www.linkedin.com/in/naveen-s-a70854268/)")

        with col2:
            st.markdown("🧑🏻‍💻 B.Krishna Raja Sree")
            st.markdown("📧 Email: 22b01a4609@svecw.edu.in")
            st.markdown("Github: [Click here!](https://github.com/krishnasree76/)")
            st.markdown("Linkedin: [Click here!](https://www.linkedin.com/in/krishna-raja-sree-bonam-7b6079257/)")
            
        with col3:
            st.markdown("🧑🏻‍💻 Joseph Boban")
            st.markdown("📧 Email: joseph.dm254031@greatlakes.edu.in")
            st.markdown("Github: [Click here!](https://github.com/josephboban2000)")
            st.markdown("Linkedin: [Click here!](https://www.linkedin.com/in/josephboban/)")
         
        with col4:
            st.markdown("🧑🏻‍💻 Shaik Ayesha Parveen")
            st.markdown("📧 Email: ayeshparveen25@gmail.com")
            st.markdown("Github: [Click here!](https://github.com/ShaikAyeshaparveen25/)")
        with col5:
            st.markdown("🧑🏻‍💻 Gayathri R")
            st.markdown("📧 Email: gayathri.22ad@kct.ac.in")
            st.markdown("Github: [Click here!](https://github.com/Gayathri-R-04/)")    
                        
                       

def main():
    app = BreastCancerDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
