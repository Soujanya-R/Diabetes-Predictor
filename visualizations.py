import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def create_feature_importance_plot(trainer):
    """Create feature importance visualization"""
    if not trainer.trained:
        return None
    
    # Get feature importance from tree-based models
    importance_data = trainer.get_feature_importance()
    
    if not importance_data:
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=1, 
        cols=len(importance_data),
        subplot_titles=list(importance_data.keys()),
        shared_yaxes=True
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (model_name, importance) in enumerate(importance_data.items()):
        # Get feature names (assuming they're from the first model's training data)
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'BMI_Category', 'Age_Group', 'Glucose_Risk', 'Insulin_Efficiency',
            'Pressure_Age_Ratio', 'Metabolic_Risk'
        ]
        
        # Ensure we don't have more features than importance values
        num_features = min(len(feature_names), len(importance))
        display_features = feature_names[:num_features]
        display_importance = importance[:num_features]
        
        # Sort by importance
        sorted_indices = np.argsort(display_importance)
        sorted_features = [display_features[i] for i in sorted_indices]
        sorted_importance = display_importance[sorted_indices]
        
        fig.add_trace(
            go.Bar(
                y=sorted_features,
                x=sorted_importance,
                orientation='h',
                name=model_name,
                marker_color=colors[idx % len(colors)],
                showlegend=False
            ),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        title="Feature Importance Comparison Across Models",
        height=600,
        showlegend=False
    )
    
    return fig

def create_performance_metrics_plot(trainer):
    """Create performance metrics comparison plot"""
    if not trainer.model_scores:
        return None
    
    # Prepare data
    metrics_df = pd.DataFrame(trainer.model_scores).T
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create radar chart
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (model_name, scores) in enumerate(metrics_df.iterrows()):
        values = [scores[metric] for metric in metrics]
        values.append(values[0])  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],  # Close the chart
            fill='toself',
            name=model_name,
            line_color=colors[idx % len(colors)],
            fillcolor=colors[idx % len(colors)],
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Metrics Comparison",
        height=500
    )
    
    return fig

def create_shap_plot(predictor, X):
    """Create SHAP waterfall plot"""
    try:
        # Get SHAP values for Random Forest (most interpretable)
        shap_values = predictor.get_shap_explanation(X, 'random_forest')
        
        if shap_values is None:
            return create_feature_values_plot(X)
        
        # Get feature contributions
        contributions = predictor.get_feature_contributions(X, 'random_forest')
        
        if contributions is None:
            return create_feature_values_plot(X)
        
        # Create waterfall-style plot
        fig = go.Figure()
        
        # Prepare data for waterfall plot
        features = [contrib['feature'] for contrib in contributions[:10]]  # Top 10 features
        shap_vals = [contrib['shap_value'] for contrib in contributions[:10]]
        
        # Colors based on positive/negative contribution
        colors = ['red' if val > 0 else 'blue' for val in shap_vals]
        
        fig.add_trace(go.Bar(
            x=features,
            y=shap_vals,
            marker_color=colors,
            text=[f"{val:.3f}" for val in shap_vals],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="SHAP Feature Contributions to Prediction",
            xaxis_title="Features",
            yaxis_title="SHAP Value (Impact on Prediction)",
            height=400,
            xaxis_tickangle=-45
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
        
    except Exception as e:
        print(f"Error creating SHAP plot: {e}")
        return create_feature_values_plot(X)

def create_feature_values_plot(X):
    """Create feature values plot as fallback"""
    if isinstance(X, pd.DataFrame):
        feature_data = X.iloc[0]
    else:
        feature_names = [f'Feature_{i}' for i in range(len(X[0]))]
        feature_data = pd.Series(X[0], index=feature_names)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=feature_data.index,
        y=feature_data.values,
        marker_color='lightblue',
        text=[f"{val:.2f}" for val in feature_data.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Patient Feature Values",
        xaxis_title="Features",
        yaxis_title="Feature Values",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_confusion_matrix_plot(trainer, X_test, y_test):
    """Create confusion matrix visualization"""
    if not trainer.trained:
        return None
    
    confusion_matrices = trainer.get_confusion_matrices(X_test, y_test)
    
    # Create subplots for each model
    fig = make_subplots(
        rows=1, 
        cols=len(confusion_matrices),
        subplot_titles=list(confusion_matrices.keys()),
        shared_yaxes=True
    )
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        # Create heatmap
        heatmap = go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            showscale=idx == 0,  # Only show scale for first plot
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        )
        
        fig.add_trace(heatmap, row=1, col=idx+1)
    
    fig.update_layout(
        title="Confusion Matrices for All Models",
        height=400
    )
    
    return fig

def create_roc_curve_plot(trainer, X_test, y_test):
    """Create ROC curve comparison plot"""
    if not trainer.trained:
        return None
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (model_name, model) in enumerate(trainer.models.items()):
        # Get probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True
    )
    
    return fig

def create_prediction_confidence_plot(predictions_data):
    """Create prediction confidence visualization"""
    if not predictions_data:
        return None
    
    # Extract confidence scores
    confidences = [pred['confidence'] for pred in predictions_data]
    predictions = [pred['ensemble_prediction'] for pred in predictions_data]
    
    # Create scatter plot
    fig = go.Figure()
    
    colors = ['blue' if pred == 0 else 'red' for pred in predictions]
    
    fig.add_trace(go.Scatter(
        x=range(len(confidences)),
        y=confidences,
        mode='markers',
        marker=dict(
            color=colors,
            size=8,
            opacity=0.7
        ),
        text=[f"Pred: {pred}, Conf: {conf:.3f}" for pred, conf in zip(predictions, confidences)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prediction Confidence Distribution',
        xaxis_title='Prediction Index',
        yaxis_title='Confidence Score',
        height=400
    )
    
    # Add confidence threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="High Confidence Threshold")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Confidence Threshold")
    
    return fig

def create_risk_distribution_plot(risk_scores):
    """Create risk score distribution plot"""
    if not risk_scores:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=risk_scores,
        nbinsx=20,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Risk Score Distribution',
        xaxis_title='Risk Score',
        yaxis_title='Frequency',
        height=400
    )
    
    # Add risk level boundaries
    fig.add_vline(x=25, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk")
    fig.add_vline(x=50, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk")
    fig.add_vline(x=75, line_dash="dash", line_color="red", 
                  annotation_text="High Risk")
    
    return fig

def create_ensemble_agreement_plot(individual_predictions):
    """Create ensemble model agreement visualization"""
    if not individual_predictions:
        return None
    
    # Calculate agreement statistics
    models = list(individual_predictions[0]['individual_predictions'].keys())
    agreement_data = []
    
    for pred_result in individual_predictions:
        predictions = list(pred_result['individual_predictions'].values())
        agreement = sum(predictions) / len(predictions)  # Proportion of models predicting positive
        agreement_data.append(agreement)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=agreement_data,
        nbinsx=10,
        marker_color='purple',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Model Agreement Distribution',
        xaxis_title='Proportion of Models Predicting Positive',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig
