import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, but handle gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - prediction explanations will use alternative methods")

class DiabetesPredictor:
    def __init__(self, model_trainer):
        self.trainer = model_trainer
        self.shap_explainers = {}
        self._initialize_shap_explainers()
    
    def _initialize_shap_explainers(self):
        """Initialize SHAP explainers for each model"""
        if not self.trainer.trained or not SHAP_AVAILABLE:
            return
        
        try:
            # Initialize explainers for tree-based models
            if 'random_forest' in self.trainer.models:
                self.shap_explainers['random_forest'] = shap.TreeExplainer(
                    self.trainer.models['random_forest']
                )
            
            if 'xgboost' in self.trainer.models:
                self.shap_explainers['xgboost'] = shap.TreeExplainer(
                    self.trainer.models['xgboost']
                )
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainers: {e}")
    
    def predict(self, X):
        """Make comprehensive prediction with explanations"""
        if not self.trainer.trained:
            raise ValueError("Models must be trained before prediction")
        
        # Get ensemble prediction
        ensemble_pred, ensemble_prob, individual_results = self.trainer.get_ensemble_prediction(X)
        
        # Calculate confidence metrics
        confidence = self._calculate_confidence(individual_results['probabilities'])
        
        # Prepare individual model results
        individual_predictions = {}
        individual_probabilities = {}
        
        for model_name in self.trainer.models.keys():
            individual_predictions[model_name] = individual_results['predictions'][model_name][0]
            individual_probabilities[model_name] = individual_results['probabilities'][model_name][0]
        
        prediction_result = {
            'ensemble_prediction': ensemble_pred[0],
            'ensemble_probability': ensemble_prob[0],
            'confidence': confidence[0],
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'risk_factors': self._analyze_risk_factors(X),
            'recommendation': self._generate_recommendation(ensemble_pred[0], ensemble_prob[0])
        }
        
        return prediction_result
    
    def _calculate_confidence(self, probabilities_dict):
        """Calculate prediction confidence based on model agreement"""
        probs = list(probabilities_dict.values())
        prob_array = np.array(probs)
        
        # Confidence based on standard deviation (lower std = higher confidence)
        prob_std = np.std(prob_array, axis=0)
        max_std = 0.5  # Maximum possible standard deviation for binary classification
        confidence = 1 - (prob_std / max_std)
        
        return confidence
    
    def _analyze_risk_factors(self, X):
        """Analyze individual risk factors based on input values"""
        risk_factors = []
        
        # Convert to dictionary if DataFrame
        if isinstance(X, pd.DataFrame):
            patient_data = X.iloc[0].to_dict()
        else:
            patient_data = X
        
        # Define risk thresholds based on medical guidelines
        risk_thresholds = {
            'Glucose': {'high': 140, 'very_high': 200},
            'BMI': {'high': 30, 'very_high': 35},
            'BloodPressure': {'high': 90, 'very_high': 100},
            'Age': {'high': 45, 'very_high': 65},
            'DiabetesPedigreeFunction': {'high': 0.5, 'very_high': 1.0},
            'Insulin': {'low': 16, 'high': 166}
        }
        
        # Analyze each risk factor
        for feature, thresholds in risk_thresholds.items():
            if feature in patient_data:
                value = patient_data[feature]
                
                if feature == 'Insulin':
                    if value < thresholds['low']:
                        risk_factors.append({
                            'factor': feature,
                            'value': value,
                            'risk_level': 'High',
                            'description': 'Very low insulin levels'
                        })
                    elif value > thresholds['high']:
                        risk_factors.append({
                            'factor': feature,
                            'value': value,
                            'risk_level': 'Medium',
                            'description': 'Elevated insulin levels'
                        })
                else:
                    if value >= thresholds.get('very_high', float('inf')):
                        risk_factors.append({
                            'factor': feature,
                            'value': value,
                            'risk_level': 'Very High',
                            'description': f'Very elevated {feature.lower()}'
                        })
                    elif value >= thresholds['high']:
                        risk_factors.append({
                            'factor': feature,
                            'value': value,
                            'risk_level': 'High',
                            'description': f'Elevated {feature.lower()}'
                        })
        
        return risk_factors
    
    def _generate_recommendation(self, prediction, probability):
        """Generate clinical recommendations based on prediction"""
        recommendations = []
        
        if prediction == 1:  # Positive prediction
            recommendations.append("⚠️ High diabetes risk detected - recommend immediate medical consultation")
            recommendations.append("🩺 Suggest comprehensive diabetes screening including HbA1c test")
            recommendations.append("🥗 Consider lifestyle modifications: diet and exercise counseling")
            recommendations.append("📊 Monitor blood glucose levels regularly")
        else:  # Negative prediction
            if probability > 0.3:  # High probability even if negative
                recommendations.append("⚡ Moderate risk detected - recommend preventive measures")
                recommendations.append("🏃‍♂️ Encourage regular physical activity and healthy diet")
                recommendations.append("📅 Schedule regular health check-ups")
            else:
                recommendations.append("✅ Low risk detected - maintain current healthy lifestyle")
                recommendations.append("📅 Continue regular health monitoring")
        
        return recommendations
    
    def get_shap_explanation(self, X, model_name='random_forest'):
        """Get SHAP explanation for a specific model"""
        if not SHAP_AVAILABLE or model_name not in self.shap_explainers:
            return None
        
        try:
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(X)
            
            # For binary classification, get positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            return shap_values
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None
    
    def get_feature_contributions(self, X, model_name='random_forest'):
        """Get feature contributions for prediction explanation"""
        shap_values = self.get_shap_explanation(X, model_name)
        
        if shap_values is None:
            return None
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            feature_values = X.iloc[0].tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            feature_values = X[0].tolist()
        
        # Create contribution analysis
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_values[0])):
            contributions.append({
                'feature': name,
                'value': value,
                'shap_value': shap_val,
                'contribution': 'Increases Risk' if shap_val > 0 else 'Decreases Risk',
                'magnitude': abs(shap_val)
            })
        
        # Sort by magnitude
        contributions.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return contributions
    
    def predict_batch(self, X_batch):
        """Make predictions for multiple samples"""
        if not self.trainer.trained:
            raise ValueError("Models must be trained before prediction")
        
        results = []
        
        for i in range(len(X_batch)):
            X_sample = X_batch.iloc[i:i+1] if isinstance(X_batch, pd.DataFrame) else X_batch[i:i+1]
            result = self.predict(X_sample)
            results.append(result)
        
        return results
    
    def get_prediction_summary(self, X):
        """Get a comprehensive prediction summary"""
        prediction_result = self.predict(X)
        
        # Create summary
        summary = {
            'prediction_summary': {
                'risk_level': self._get_risk_category(prediction_result['ensemble_probability']),
                'probability': prediction_result['ensemble_probability'],
                'confidence': prediction_result['confidence'],
                'primary_prediction': 'Positive' if prediction_result['ensemble_prediction'] == 1 else 'Negative'
            },
            'model_agreement': {
                'models_predicting_positive': sum(prediction_result['individual_predictions'].values()),
                'total_models': len(prediction_result['individual_predictions']),
                'agreement_percentage': (sum(prediction_result['individual_predictions'].values()) / 
                                       len(prediction_result['individual_predictions'])) * 100
            },
            'risk_factors': prediction_result['risk_factors'],
            'recommendations': prediction_result['recommendation']
        }
        
        return summary
    
    def _get_risk_category(self, probability):
        """Categorize risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def calculate_risk_score(self, X):
        """Calculate a comprehensive risk score (0-100)"""
        prediction_result = self.predict(X)
        
        # Base score from ensemble probability
        base_score = prediction_result['ensemble_probability'] * 100
        
        # Adjust based on confidence
        confidence_adjustment = (prediction_result['confidence'] - 0.5) * 20
        
        # Adjust based on number of risk factors
        risk_factor_adjustment = len(prediction_result['risk_factors']) * 5
        
        # Calculate final score
        final_score = base_score + confidence_adjustment + risk_factor_adjustment
        final_score = max(0, min(100, final_score))  # Clamp between 0 and 100
        
        return {
            'risk_score': final_score,
            'base_score': base_score,
            'confidence_adjustment': confidence_adjustment,
            'risk_factor_adjustment': risk_factor_adjustment,
            'interpretation': self._interpret_risk_score(final_score)
        }
    
    def _interpret_risk_score(self, score):
        """Interpret the risk score"""
        if score < 25:
            return "Very Low Risk - Excellent metabolic health indicators"
        elif score < 50:
            return "Low Risk - Good metabolic health with minor concerns"
        elif score < 75:
            return "Moderate Risk - Several concerning factors present"
        else:
            return "High Risk - Multiple significant risk factors detected"
