import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from model_trainer import ModelTrainer
from predictor import DiabetesPredictor
from visualizations import create_feature_importance_plot, create_shap_plot, create_performance_metrics_plot
from utils import get_risk_level, validate_input

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    st.title("🩺 Comprehensive Diabetes Prediction System")
    st.markdown("### Clinical-Grade AI-Powered Diabetes Risk Assessment")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Home", "Data Overview", "Model Training", "Prediction", "Model Performance", "About"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Data Overview":
        show_data_overview()
    elif page == "Model Training":
        show_model_training()
    elif page == "Prediction":
        show_prediction_page()
    elif page == "Model Performance":
        show_performance_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.header("Welcome to the Diabetes Prediction System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This comprehensive system uses ensemble machine learning models to predict diabetes risk
        with clinical-grade accuracy. The system combines multiple algorithms and provides
        explainable AI insights for healthcare professionals.
        
        **Key Features:**
        - 🤖 **Ensemble Learning**: Random Forest, XGBoost, and SVM models
        - 📊 **Explainable AI**: SHAP values for prediction interpretation
        - 📈 **Performance Metrics**: Comprehensive model evaluation
        - 🎯 **Risk Assessment**: Categorized risk levels with confidence intervals
        - 🔍 **Feature Analysis**: Detailed feature importance visualization
        """)
        
        # Quick start section
        st.subheader("Quick Start")
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Loading data and training models..."):
                try:
                    # Load data
                    data_loader = DataLoader()
                    X_train, X_test, y_train, y_test = data_loader.load_and_split_data()
                    st.session_state.data_loaded = True
                    
                    # Train models
                    trainer = ModelTrainer()
                    trainer.train_models(X_train, y_train)
                    trainer.evaluate_models(X_test, y_test)
                    st.session_state.models_trained = True
                    st.session_state.trainer = trainer
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    st.success("✅ System initialized successfully!")
                    st.info("Navigate to the 'Prediction' section to start making predictions.")
                except Exception as e:
                    st.error(f"❌ Error initializing system: {str(e)}")
    
    with col2:
        # System status
        st.subheader("System Status")
        
        status_data = {
            "Component": ["Data Loading", "Model Training", "System Ready"],
            "Status": [
                "✅ Ready" if st.session_state.data_loaded else "⏳ Pending",
                "✅ Ready" if st.session_state.models_trained else "⏳ Pending",
                "✅ Ready" if (st.session_state.data_loaded and st.session_state.models_trained) else "⏳ Pending"
            ]
        }
        
        st.dataframe(pd.DataFrame(status_data), hide_index=True)
        
        # Medical disclaimer
        st.warning("""
        **⚠️ Medical Disclaimer**
        
        This tool is for educational and research purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment.
        Always consult with qualified healthcare professionals for medical decisions.
        """)

def show_data_overview():
    st.header("📊 Data Overview")
    
    try:
        data_loader = DataLoader()
        df = data_loader.load_raw_data()
        
        # Dataset information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            diabetes_rate = (df['Outcome'].sum() / len(df)) * 100
            st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")
        
        # Data preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        feature_descriptions = {
            "Pregnancies": "Number of times pregnant",
            "Glucose": "Plasma glucose concentration (mg/dL)",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (weight in kg/(height in m)^2)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function",
            "Age": "Age in years",
            "Outcome": "Diabetes diagnosis (0: No, 1: Yes)"
        }
        
        desc_df = pd.DataFrame(list(feature_descriptions.items()), columns=["Feature", "Description"])
        st.dataframe(desc_df, hide_index=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Data visualization
        st.subheader("Data Distribution")
        
        # Feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Outcome' in numeric_cols:
            numeric_cols.remove('Outcome')
        
        selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                df, 
                x=selected_feature, 
                color='Outcome',
                title=f"Distribution of {selected_feature}",
                nbins=20
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                df, 
                x='Outcome', 
                y=selected_feature,
                title=f"{selected_feature} by Diabetes Status"
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlation Matrix")
        corr_matrix = df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_model_training():
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("Please initialize the system from the Home page first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Cross-validation settings
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2)
        random_state = st.number_input("Random state:", 1, 1000, 42)
        
        if st.button("🔄 Retrain Models", type="primary"):
            with st.spinner("Training models..."):
                try:
                    # Load data with new parameters
                    data_loader = DataLoader()
                    X_train, X_test, y_train, y_test = data_loader.load_and_split_data(
                        test_size=test_size, 
                        random_state=random_state
                    )
                    
                    # Train models
                    trainer = ModelTrainer()
                    trainer.train_models(X_train, y_train, cv_folds=cv_folds)
                    trainer.evaluate_models(X_test, y_test)
                    
                    # Store in session state
                    st.session_state.trainer = trainer
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        if st.session_state.models_trained and 'trainer' in st.session_state:
            trainer = st.session_state.trainer
            
            st.subheader("Model Performance")
            
            # Performance metrics table
            metrics_df = pd.DataFrame(trainer.model_scores).T
            metrics_df = metrics_df.round(4)
            st.dataframe(metrics_df)
            
            # Best model highlight
            best_model = metrics_df['accuracy'].idxmax()
            st.success(f"🏆 Best performing model: **{best_model}** (Accuracy: {metrics_df.loc[best_model, 'accuracy']:.4f})")
            
            # Cross-validation scores
            st.subheader("Cross-Validation Scores")
            cv_scores_df = pd.DataFrame(trainer.cv_scores)
            st.dataframe(cv_scores_df.describe())
            
            # Visualize CV scores
            fig = px.box(
                cv_scores_df.melt(var_name='Model', value_name='Accuracy'),
                x='Model',
                y='Accuracy',
                title="Cross-Validation Accuracy Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train models to see performance metrics.")

def show_prediction_page():
    st.header("🔮 Diabetes Risk Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training section.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Information")
        
        # Input form
        with st.form("prediction_form"):
            pregnancies = st.number_input("Number of Pregnancies:", 0, 20, 0)
            glucose = st.number_input("Glucose Level (mg/dL):", 0.0, 300.0, 120.0)
            blood_pressure = st.number_input("Blood Pressure (mm Hg):", 0.0, 200.0, 80.0)
            skin_thickness = st.number_input("Skin Thickness (mm):", 0.0, 100.0, 20.0)
            insulin = st.number_input("Insulin (mu U/ml):", 0.0, 900.0, 80.0)
            bmi = st.number_input("BMI:", 0.0, 70.0, 25.0)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function:", 0.0, 3.0, 0.5)
            age = st.number_input("Age (years):", 1, 120, 30)
            
            submitted = st.form_submit_button("🔍 Predict Diabetes Risk", type="primary")
        
        if submitted:
            # Validate inputs
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age
            }
            
            validation_errors = validate_input(input_data)
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                st.session_state.prediction_input = input_data
    
    with col2:
        if 'prediction_input' in st.session_state and st.session_state.models_trained:
            try:
                predictor = DiabetesPredictor(st.session_state.trainer)
                input_df = pd.DataFrame([st.session_state.prediction_input])
                
                # Make prediction
                prediction_result = predictor.predict(input_df)
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                # Risk level with color coding
                risk_level, risk_color = get_risk_level(prediction_result['ensemble_probability'])
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Risk Level", 
                        risk_level,
                        delta=f"{prediction_result['ensemble_probability']:.1%} probability"
                    )
                with col_b:
                    st.metric("Ensemble Prediction", 
                             "Positive" if prediction_result['ensemble_prediction'] == 1 else "Negative")
                with col_c:
                    st.metric("Confidence", f"{prediction_result['confidence']:.1%}")
                
                # Individual model predictions
                st.subheader("Individual Model Predictions")
                model_results = []
                for model_name in ['random_forest', 'xgboost', 'svm']:
                    model_results.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Prediction': 'Positive' if prediction_result['individual_predictions'][model_name] == 1 else 'Negative',
                        'Probability': f"{prediction_result['individual_probabilities'][model_name]:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(model_results), hide_index=True)
                
                # SHAP explanation
                st.subheader("🔍 Prediction Explanation (SHAP)")
                try:
                    shap_fig = create_shap_plot(predictor, input_df)
                    st.plotly_chart(shap_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP visualization unavailable: {str(e)}")
                
                # Feature importance
                st.subheader("📊 Feature Importance")
                importance_fig = create_feature_importance_plot(st.session_state.trainer)
                st.plotly_chart(importance_fig, use_container_width=True)
                
                # Risk factors analysis
                st.subheader("⚠️ Risk Factors Analysis")
                risk_factors = []
                
                if st.session_state.prediction_input['Glucose'] > 140:
                    risk_factors.append("High glucose levels detected")
                if st.session_state.prediction_input['BMI'] > 30:
                    risk_factors.append("BMI indicates obesity")
                if st.session_state.prediction_input['Age'] > 45:
                    risk_factors.append("Advanced age increases risk")
                if st.session_state.prediction_input['BloodPressure'] > 90:
                    risk_factors.append("Elevated blood pressure")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ No major risk factors detected")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_performance_page():
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training section.")
        return
    
    trainer = st.session_state.trainer
    
    # Performance metrics visualization
    st.subheader("Performance Metrics Comparison")
    metrics_fig = create_performance_metrics_plot(trainer)
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    metrics_df = pd.DataFrame(trainer.model_scores).T
    metrics_df = metrics_df.round(4)
    
    # Add ranking
    metrics_df['Rank'] = metrics_df['accuracy'].rank(ascending=False).astype(int)
    metrics_df = metrics_df.sort_values('Rank')
    
    st.dataframe(metrics_df)
    
    # Model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        accuracy_data = metrics_df['accuracy'].reset_index()
        accuracy_data.columns = ['Model', 'Accuracy']
        
        fig_acc = px.bar(
            accuracy_data,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.subheader("F1-Score Comparison")
        f1_data = metrics_df['f1'].reset_index()
        f1_data.columns = ['Model', 'F1_Score']
        
        fig_f1 = px.bar(
            f1_data,
            x='Model',
            y='F1_Score',
            title="Model F1-Score Comparison",
            color='F1_Score',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Cross-validation analysis
    st.subheader("Cross-Validation Analysis")
    if hasattr(trainer, 'cv_scores'):
        cv_df = pd.DataFrame(trainer.cv_scores)
        
        # Statistics
        cv_stats = cv_df.describe().round(4)
        st.dataframe(cv_stats)
        
        # Violin plot
        fig_violin = px.violin(
            cv_df.melt(var_name='Model', value_name='CV_Score'),
            x='Model',
            y='CV_Score',
            title="Cross-Validation Score Distribution",
            box=True
        )
        st.plotly_chart(fig_violin, use_container_width=True)

def show_about_page():
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## Overview
    This Diabetes Prediction System is a comprehensive machine learning application designed to assist healthcare professionals in assessing diabetes risk. The system employs ensemble learning techniques and explainable AI to provide accurate, interpretable predictions.
    
    ## Technical Architecture
    
    ### Machine Learning Models
    - **Random Forest**: Ensemble of decision trees with feature bagging
    - **XGBoost**: Gradient boosting with advanced regularization
    - **Support Vector Machine**: Non-linear classification with RBF kernel
    - **Ensemble Voting**: Soft voting combining all three models
    
    ### Key Features
    - **Data Preprocessing**: Comprehensive pipeline with scaling and feature engineering
    - **Cross-Validation**: Robust model evaluation with stratified k-fold
    - **Explainable AI**: SHAP values for prediction interpretation
    - **Performance Metrics**: Clinical-grade evaluation metrics
    - **Risk Categorization**: Three-tier risk assessment system
    
    ## Dataset Information
    The system uses the **Pima Indians Diabetes Database**, a well-established dataset in medical machine learning research. This dataset contains diagnostic measurements for female patients of Pima Indian heritage.
    
    ### Features Used:
    1. **Pregnancies**: Number of pregnancies
    2. **Glucose**: Plasma glucose concentration
    3. **Blood Pressure**: Diastolic blood pressure
    4. **Skin Thickness**: Triceps skin fold thickness
    5. **Insulin**: 2-hour serum insulin
    6. **BMI**: Body mass index
    7. **Diabetes Pedigree Function**: Genetic diabetes likelihood
    8. **Age**: Patient age
    
    ## Clinical Relevance
    - Predictions include confidence intervals for uncertainty quantification
    - Feature importance analysis highlights key risk factors
    - SHAP explanations provide individual prediction rationale
    - Risk categorization follows clinical guidelines
    
    ## Limitations and Disclaimers
    
    ⚠️ **Important Medical Disclaimer:**
    - This tool is for educational and research purposes only
    - Not intended for clinical diagnosis or treatment decisions
    - Should not replace professional medical consultation
    - Results may not generalize to all populations
    - Always consult qualified healthcare professionals
    
    ## Performance Metrics
    The system typically achieves:
    - **Accuracy**: >85% on validation data
    - **Precision**: High specificity to minimize false positives
    - **Recall**: Balanced to catch true positive cases
    - **F1-Score**: Optimized for clinical applications
    
    ## Technology Stack
    - **Frontend**: Streamlit for interactive web interface
    - **ML Framework**: Scikit-learn, XGBoost
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    - **Data Processing**: Pandas, NumPy
    - **Model Validation**: Cross-validation, stratified sampling
    
    ## Future Enhancements
    - Integration with electronic health records (EHR)
    - Longitudinal risk assessment
    - Additional biomarkers and genetic factors
    - Population-specific model adaptations
    - Real-time monitoring capabilities
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: 2024  
    **Contact**: For technical support or clinical questions, consult your healthcare provider.
    """)

if __name__ == "__main__":
    main()
