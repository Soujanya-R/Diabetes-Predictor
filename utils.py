import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any

def validate_input(input_data: Dict[str, float]) -> List[str]:
    """Validate user input for medical reasonableness"""
    errors = []
    
    # Define realistic ranges for each feature
    valid_ranges = {
        'Pregnancies': (0, 20, "Number of pregnancies should be between 0 and 20"),
        'Glucose': (50, 300, "Glucose level should be between 50 and 300 mg/dL"),
        'BloodPressure': (40, 200, "Blood pressure should be between 40 and 200 mm Hg"),
        'SkinThickness': (0, 100, "Skin thickness should be between 0 and 100 mm"),
        'Insulin': (0, 1000, "Insulin level should be between 0 and 1000 mu U/ml"),
        'BMI': (10, 70, "BMI should be between 10 and 70"),
        'DiabetesPedigreeFunction': (0, 3, "Diabetes Pedigree Function should be between 0 and 3"),
        'Age': (1, 120, "Age should be between 1 and 120 years")
    }
    
    # Check each input
    for feature, value in input_data.items():
        if feature in valid_ranges:
            min_val, max_val, error_msg = valid_ranges[feature]
            if not (min_val <= value <= max_val):
                errors.append(f"❌ {error_msg}. Current value: {value}")
    
    # Additional medical logic validations
    if input_data.get('BMI', 0) < 15 and input_data.get('Age', 0) > 18:
        errors.append("⚠️ BMI appears very low for an adult - please verify")
    
    if input_data.get('Glucose', 0) > 200 and input_data.get('Insulin', 0) < 10:
        errors.append("⚠️ Very high glucose with very low insulin - please verify values")
    
    if input_data.get('BloodPressure', 0) < 60 and input_data.get('Age', 0) > 20:
        errors.append("⚠️ Blood pressure appears very low - please verify")
    
    return errors

def get_risk_level(probability: float) -> Tuple[str, str]:
    """Categorize risk level and return appropriate color"""
    if probability < 0.3:
        return "Low Risk", "green"
    elif probability < 0.7:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

def format_medical_value(value: float, feature: str) -> str:
    """Format medical values with appropriate units and precision"""
    units = {
        'Glucose': 'mg/dL',
        'BloodPressure': 'mm Hg',
        'SkinThickness': 'mm',
        'Insulin': 'mu U/ml',
        'BMI': 'kg/m²',
        'Age': 'years'
    }
    
    if feature in units:
        if feature in ['BMI', 'DiabetesPedigreeFunction']:
            return f"{value:.2f} {units.get(feature, '')}"
        else:
            return f"{value:.0f} {units.get(feature, '')}"
    else:
        return f"{value:.2f}"

def calculate_diabetes_risk_factors(input_data: Dict[str, float]) -> Dict[str, Any]:
    """Calculate various diabetes risk factors based on input"""
    risk_analysis = {
        'metabolic_syndrome_criteria': 0,
        'major_risk_factors': [],
        'protective_factors': [],
        'risk_score': 0
    }
    
    # Metabolic syndrome criteria (simplified)
    if input_data.get('BMI', 0) >= 30:
        risk_analysis['metabolic_syndrome_criteria'] += 1
        risk_analysis['major_risk_factors'].append("Obesity (BMI ≥ 30)")
    
    if input_data.get('Glucose', 0) >= 100:
        risk_analysis['metabolic_syndrome_criteria'] += 1
        risk_analysis['major_risk_factors'].append("Elevated fasting glucose")
    
    if input_data.get('BloodPressure', 0) >= 85:
        risk_analysis['metabolic_syndrome_criteria'] += 1
        risk_analysis['major_risk_factors'].append("Elevated blood pressure")
    
    # Age-related risk
    age = input_data.get('Age', 0)
    if age >= 45:
        risk_analysis['major_risk_factors'].append("Age ≥ 45 years")
    elif age < 30:
        risk_analysis['protective_factors'].append("Young age")
    
    # BMI categories
    bmi = input_data.get('BMI', 0)
    if bmi < 25:
        risk_analysis['protective_factors'].append("Normal BMI")
    elif bmi >= 35:
        risk_analysis['major_risk_factors'].append("Severe obesity")
    
    # Family history (diabetes pedigree function)
    dpf = input_data.get('DiabetesPedigreeFunction', 0)
    if dpf > 0.5:
        risk_analysis['major_risk_factors'].append("Strong family history")
    elif dpf < 0.2:
        risk_analysis['protective_factors'].append("Limited family history")
    
    # Calculate overall risk score (0-100)
    risk_score = 0
    risk_score += min(input_data.get('Age', 0) / 80 * 20, 20)  # Age component (max 20 points)
    risk_score += min(max(input_data.get('BMI', 0) - 18.5, 0) / 20 * 25, 25)  # BMI component (max 25 points)
    risk_score += min(max(input_data.get('Glucose', 0) - 70, 0) / 130 * 30, 30)  # Glucose component (max 30 points)
    risk_score += min(input_data.get('DiabetesPedigreeFunction', 0) / 2 * 15, 15)  # Family history (max 15 points)
    risk_score += min(max(input_data.get('BloodPressure', 0) - 60, 0) / 40 * 10, 10)  # BP component (max 10 points)
    
    risk_analysis['risk_score'] = min(risk_score, 100)
    
    return risk_analysis

def generate_lifestyle_recommendations(input_data: Dict[str, float], risk_level: str) -> List[str]:
    """Generate personalized lifestyle recommendations"""
    recommendations = []
    
    # BMI-based recommendations
    bmi = input_data.get('BMI', 0)
    if bmi >= 30:
        recommendations.append("🏃‍♂️ Weight management: Aim for 5-10% weight loss through caloric restriction and increased physical activity")
        recommendations.append("🥗 Nutrition: Consider consultation with a registered dietitian for personalized meal planning")
    elif bmi >= 25:
        recommendations.append("⚖️ Weight maintenance: Focus on preventing further weight gain through balanced diet and regular exercise")
    
    # Glucose-based recommendations
    glucose = input_data.get('Glucose', 0)
    if glucose >= 140:
        recommendations.append("🍽️ Dietary changes: Limit refined carbohydrates and sugary foods")
        recommendations.append("⏰ Meal timing: Consider smaller, frequent meals to help manage blood glucose")
    elif glucose >= 100:
        recommendations.append("🌾 Carbohydrate awareness: Choose complex carbohydrates and monitor portion sizes")
    
    # Blood pressure recommendations
    bp = input_data.get('BloodPressure', 0)
    if bp >= 90:
        recommendations.append("🧂 Sodium reduction: Limit sodium intake to less than 2,300mg per day")
        recommendations.append("🧘‍♀️ Stress management: Practice relaxation techniques and regular stress-reduction activities")
    
    # Age-based recommendations
    age = input_data.get('Age', 0)
    if age >= 45:
        recommendations.append("📅 Regular screening: Annual diabetes screening and comprehensive metabolic panels")
        recommendations.append("💊 Medication review: Regular review of medications that may affect glucose levels")
    
    # General recommendations based on risk level
    if risk_level == "High Risk":
        recommendations.extend([
            "🩺 Medical follow-up: Schedule appointment with healthcare provider within 2-4 weeks",
            "📊 Glucose monitoring: Consider home glucose monitoring as recommended by physician",
            "💉 Medication consideration: Discuss preventive medications with healthcare provider"
        ])
    elif risk_level == "Medium Risk":
        recommendations.extend([
            "📋 Lifestyle modification: Implement structured diet and exercise program",
            "📅 Regular monitoring: Schedule follow-up in 3-6 months",
            "👥 Support system: Consider joining diabetes prevention programs"
        ])
    else:  # Low Risk
        recommendations.extend([
            "✅ Maintenance: Continue current healthy lifestyle practices",
            "📊 Annual screening: Maintain regular annual health check-ups"
        ])
    
    # Physical activity recommendations
    recommendations.append("🏋️‍♀️ Physical activity: Aim for at least 150 minutes of moderate-intensity aerobic activity per week")
    recommendations.append("💪 Strength training: Include muscle-strengthening activities at least 2 days per week")
    
    return recommendations

def create_patient_summary(input_data: Dict[str, float], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive patient summary"""
    risk_factors = calculate_diabetes_risk_factors(input_data)
    risk_level, _ = get_risk_level(prediction_result['ensemble_probability'])
    recommendations = generate_lifestyle_recommendations(input_data, risk_level)
    
    summary = {
        'patient_data': {
            'age': input_data.get('Age', 0),
            'bmi': input_data.get('BMI', 0),
            'glucose': input_data.get('Glucose', 0),
            'blood_pressure': input_data.get('BloodPressure', 0),
            'bmi_category': get_bmi_category(input_data.get('BMI', 0)),
            'glucose_category': get_glucose_category(input_data.get('Glucose', 0))
        },
        'prediction_summary': {
            'risk_level': risk_level,
            'probability': prediction_result['ensemble_probability'],
            'confidence': prediction_result['confidence'],
            'ensemble_prediction': prediction_result['ensemble_prediction']
        },
        'risk_analysis': risk_factors,
        'recommendations': recommendations,
        'follow_up': get_follow_up_schedule(risk_level),
        'red_flags': identify_red_flags(input_data)
    }
    
    return summary

def get_bmi_category(bmi: float) -> str:
    """Categorize BMI according to medical standards"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity Class I"
    elif bmi < 40:
        return "Obesity Class II"
    else:
        return "Obesity Class III (Severe)"

def get_glucose_category(glucose: float) -> str:
    """Categorize glucose levels according to medical standards"""
    if glucose < 70:
        return "Hypoglycemic"
    elif glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Prediabetic"
    else:
        return "Diabetic range"

def get_follow_up_schedule(risk_level: str) -> Dict[str, str]:
    """Recommend follow-up schedule based on risk level"""
    schedules = {
        "Low Risk": {
            "next_screening": "12 months",
            "lifestyle_review": "6 months",
            "medical_check": "Annual physical exam"
        },
        "Medium Risk": {
            "next_screening": "6 months",
            "lifestyle_review": "3 months",
            "medical_check": "6 months with healthcare provider"
        },
        "High Risk": {
            "next_screening": "3 months",
            "lifestyle_review": "Monthly",
            "medical_check": "Immediate consultation recommended"
        }
    }
    
    return schedules.get(risk_level, schedules["Medium Risk"])

def identify_red_flags(input_data: Dict[str, float]) -> List[str]:
    """Identify critical values that require immediate attention"""
    red_flags = []
    
    glucose = input_data.get('Glucose', 0)
    if glucose > 200:
        red_flags.append("🚨 Severely elevated glucose - immediate medical attention recommended")
    
    bp = input_data.get('BloodPressure', 0)
    if bp > 140:
        red_flags.append("🚨 Severely elevated blood pressure - immediate medical attention recommended")
    elif bp < 50:
        red_flags.append("🚨 Dangerously low blood pressure - immediate medical attention recommended")
    
    bmi = input_data.get('BMI', 0)
    if bmi > 40:
        red_flags.append("🚨 Severe obesity - comprehensive medical evaluation recommended")
    elif bmi < 15:
        red_flags.append("🚨 Severely underweight - medical evaluation recommended")
    
    age = input_data.get('Age', 0)
    if age > 75 and glucose > 140:
        red_flags.append("🚨 Elderly patient with elevated glucose - careful monitoring required")
    
    return red_flags

def export_patient_data(patient_summary: Dict[str, Any]) -> str:
    """Export patient data as formatted text for medical records"""
    export_text = f"""
DIABETES RISK ASSESSMENT REPORT
================================

PATIENT INFORMATION:
- Age: {patient_summary['patient_data']['age']} years
- BMI: {patient_summary['patient_data']['bmi']:.1f} ({patient_summary['patient_data']['bmi_category']})
- Glucose: {patient_summary['patient_data']['glucose']:.0f} mg/dL ({patient_summary['patient_data']['glucose_category']})
- Blood Pressure: {patient_summary['patient_data']['blood_pressure']:.0f} mm Hg

RISK ASSESSMENT:
- Risk Level: {patient_summary['prediction_summary']['risk_level']}
- Diabetes Probability: {patient_summary['prediction_summary']['probability']:.1%}
- Prediction Confidence: {patient_summary['prediction_summary']['confidence']:.1%}

RISK FACTORS IDENTIFIED:
"""
    
    for factor in patient_summary['risk_analysis']['major_risk_factors']:
        export_text += f"- {factor}\n"
    
    if patient_summary['red_flags']:
        export_text += "\nCRITICAL ALERTS:\n"
        for flag in patient_summary['red_flags']:
            export_text += f"- {flag}\n"
    
    export_text += "\nRECOMMENDATIONS:\n"
    for rec in patient_summary['recommendations'][:5]:  # Top 5 recommendations
        export_text += f"- {rec}\n"
    
    export_text += f"\nFOLLOW-UP SCHEDULE:\n"
    follow_up = patient_summary['follow_up']
    export_text += f"- Next screening: {follow_up['next_screening']}\n"
    export_text += f"- Lifestyle review: {follow_up['lifestyle_review']}\n"
    export_text += f"- Medical check: {follow_up['medical_check']}\n"
    
    export_text += "\n" + "="*50 + "\n"
    export_text += "Note: This assessment is for educational purposes only.\n"
    export_text += "Consult healthcare professionals for medical decisions.\n"
    
    return export_text

# Utility functions for data processing
def normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature names for consistency"""
    name_mapping = {
        'pregnancies': 'Pregnancies',
        'glucose': 'Glucose',
        'bloodpressure': 'BloodPressure',
        'skinthickness': 'SkinThickness',
        'insulin': 'Insulin',
        'bmi': 'BMI',
        'diabetespedigreefunction': 'DiabetesPedigreeFunction',
        'age': 'Age'
    }
    
    df_normalized = df.copy()
    df_normalized.columns = [name_mapping.get(col.lower(), col) for col in df.columns]
    
    return df_normalized

def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """Detect outliers in the dataset"""
    outliers = pd.DataFrame(index=df.index)
    
    for column in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
        elif method == 'z_score':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers[column] = z_scores > 3
    
    return outliers
