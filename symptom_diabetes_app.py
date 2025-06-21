import streamlit as st
import pandas as pd
import numpy as np
import math

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

class SymptomDiabetesPredictor:
    """Diabetes risk prediction based on symptoms and lifestyle factors"""
    
    def __init__(self):
        # Risk scoring weights based on medical literature
        self.symptom_weights = {
            'frequent_urination': 15,
            'excessive_thirst': 15,
            'unexplained_weight_loss': 12,
            'extreme_fatigue': 10,
            'blurred_vision': 8,
            'slow_healing_cuts': 8,
            'frequent_infections': 7,
            'tingling_hands_feet': 6,
            'dark_skin_patches': 5,
            'increased_hunger': 4
        }
        
        self.lifestyle_weights = {
            'family_history': 20,
            'age_risk': 15,
            'weight_status': 15,
            'physical_activity': 10,
            'diet_quality': 8,
            'stress_levels': 5,
            'sleep_quality': 4,
            'smoking_status': 3
        }
        
        self.demographic_weights = {
            'ethnicity_risk': 10,
            'previous_gestational': 8,
            'pcos_history': 6
        }
    
    def calculate_symptom_score(self, symptoms):
        """Calculate score based on reported symptoms"""
        score = 0
        for symptom, present in symptoms.items():
            if present and symptom in self.symptom_weights:
                score += self.symptom_weights[symptom]
        return min(score, 100)
    
    def calculate_lifestyle_score(self, lifestyle_factors):
        """Calculate score based on lifestyle risk factors"""
        score = 0
        
        # Family history
        if lifestyle_factors['family_history'] == 'Yes - Parent/Sibling':
            score += self.lifestyle_weights['family_history']
        elif lifestyle_factors['family_history'] == 'Yes - Grandparent/Other':
            score += self.lifestyle_weights['family_history'] * 0.5
        
        # Age risk
        age = lifestyle_factors['age']
        if age >= 65:
            score += self.lifestyle_weights['age_risk']
        elif age >= 45:
            score += self.lifestyle_weights['age_risk'] * 0.7
        elif age >= 35:
            score += self.lifestyle_weights['age_risk'] * 0.3
        
        # Weight status
        bmi_category = lifestyle_factors['weight_status']
        if bmi_category == 'Obese (BMI > 30)':
            score += self.lifestyle_weights['weight_status']
        elif bmi_category == 'Overweight (BMI 25-30)':
            score += self.lifestyle_weights['weight_status'] * 0.6
        
        # Physical activity
        if lifestyle_factors['exercise'] == 'Rarely/Never':
            score += self.lifestyle_weights['physical_activity']
        elif lifestyle_factors['exercise'] == '1-2 times per week':
            score += self.lifestyle_weights['physical_activity'] * 0.5
        
        # Diet quality
        if lifestyle_factors['diet'] == 'Poor (frequent fast food, sugary drinks)':
            score += self.lifestyle_weights['diet_quality']
        elif lifestyle_factors['diet'] == 'Fair (some processed foods)':
            score += self.lifestyle_weights['diet_quality'] * 0.5
        
        # Stress levels
        if lifestyle_factors['stress'] == 'High':
            score += self.lifestyle_weights['stress_levels']
        elif lifestyle_factors['stress'] == 'Moderate':
            score += self.lifestyle_weights['stress_levels'] * 0.5
        
        # Sleep quality
        if lifestyle_factors['sleep'] == 'Poor (< 6 hours or frequent interruptions)':
            score += self.lifestyle_weights['sleep_quality']
        
        # Smoking
        if lifestyle_factors['smoking'] == 'Current smoker':
            score += self.lifestyle_weights['smoking_status']
        
        return min(score, 100)
    
    def calculate_demographic_score(self, demographics):
        """Calculate score based on demographic risk factors"""
        score = 0
        
        # Ethnicity risk
        high_risk_ethnicities = ['African American', 'Hispanic/Latino', 'Native American', 'Asian', 'Pacific Islander']
        if demographics['ethnicity'] in high_risk_ethnicities:
            score += self.demographic_weights['ethnicity_risk']
        
        # Previous gestational diabetes
        if demographics['gestational_diabetes'] == 'Yes':
            score += self.demographic_weights['previous_gestational']
        
        # PCOS history
        if demographics['pcos'] == 'Yes':
            score += self.demographic_weights['pcos_history']
        
        return min(score, 100)
    
    def predict_risk(self, symptoms, lifestyle, demographics):
        """Calculate overall diabetes risk"""
        symptom_score = self.calculate_symptom_score(symptoms)
        lifestyle_score = self.calculate_lifestyle_score(lifestyle)
        demographic_score = self.calculate_demographic_score(demographics)
        
        # Weighted combination
        total_score = (symptom_score * 0.4) + (lifestyle_score * 0.45) + (demographic_score * 0.15)
        
        # Convert to probability (0-1)
        probability = min(total_score / 100, 0.95)
        
        return {
            'total_score': total_score,
            'probability': probability,
            'symptom_score': symptom_score,
            'lifestyle_score': lifestyle_score,
            'demographic_score': demographic_score
        }

def get_risk_category(probability):
    """Categorize risk level"""
    if probability < 0.25:
        return "Low Risk", "🟢", "Continue healthy lifestyle practices"
    elif probability < 0.50:
        return "Moderate Risk", "🟡", "Consider lifestyle changes and annual screening"
    elif probability < 0.75:
        return "High Risk", "🟠", "Recommend medical consultation and blood tests"
    else:
        return "Very High Risk", "🔴", "Urgent medical attention recommended"

def main():
    st.title("🏥 Diabetes Risk Assessment")
    st.markdown("### Symptom & Lifestyle-Based Screening Tool")
    
    st.info("""
    This innovative tool predicts diabetes risk based on symptoms you may be experiencing 
    and lifestyle factors - no blood tests required! Get insights about your risk level 
    and receive personalized recommendations.
    """)
    
    # Initialize predictor
    predictor = SymptomDiabetesPredictor()
    
    # Sidebar navigation
    st.sidebar.title("Assessment Sections")
    section = st.sidebar.radio(
        "Choose section:",
        ["Symptoms Check", "Lifestyle Factors", "Demographics", "Risk Assessment", "Learn More"]
    )
    
    # Initialize session state
    if 'symptoms' not in st.session_state:
        st.session_state.symptoms = {}
    if 'lifestyle' not in st.session_state:
        st.session_state.lifestyle = {}
    if 'demographics' not in st.session_state:
        st.session_state.demographics = {}
    
    if section == "Symptoms Check":
        show_symptoms_section()
    elif section == "Lifestyle Factors":
        show_lifestyle_section()
    elif section == "Demographics":
        show_demographics_section()
    elif section == "Risk Assessment":
        show_risk_assessment(predictor)
    elif section == "Learn More":
        show_learn_more()

def show_symptoms_section():
    st.header("🩺 Symptom Assessment")
    st.markdown("Check any symptoms you've been experiencing recently:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primary Symptoms")
        
        st.session_state.symptoms['frequent_urination'] = st.checkbox(
            "Frequent urination (especially at night)", 
            value=st.session_state.symptoms.get('frequent_urination', False)
        )
        
        st.session_state.symptoms['excessive_thirst'] = st.checkbox(
            "Excessive thirst that doesn't go away", 
            value=st.session_state.symptoms.get('excessive_thirst', False)
        )
        
        st.session_state.symptoms['unexplained_weight_loss'] = st.checkbox(
            "Unexplained weight loss", 
            value=st.session_state.symptoms.get('unexplained_weight_loss', False)
        )
        
        st.session_state.symptoms['extreme_fatigue'] = st.checkbox(
            "Extreme fatigue or tiredness", 
            value=st.session_state.symptoms.get('extreme_fatigue', False)
        )
        
        st.session_state.symptoms['increased_hunger'] = st.checkbox(
            "Increased hunger", 
            value=st.session_state.symptoms.get('increased_hunger', False)
        )
    
    with col2:
        st.subheader("Secondary Symptoms")
        
        st.session_state.symptoms['blurred_vision'] = st.checkbox(
            "Blurred vision", 
            value=st.session_state.symptoms.get('blurred_vision', False)
        )
        
        st.session_state.symptoms['slow_healing_cuts'] = st.checkbox(
            "Slow-healing cuts or bruises", 
            value=st.session_state.symptoms.get('slow_healing_cuts', False)
        )
        
        st.session_state.symptoms['frequent_infections'] = st.checkbox(
            "Frequent infections (skin, gum, bladder)", 
            value=st.session_state.symptoms.get('frequent_infections', False)
        )
        
        st.session_state.symptoms['tingling_hands_feet'] = st.checkbox(
            "Tingling in hands or feet", 
            value=st.session_state.symptoms.get('tingling_hands_feet', False)
        )
        
        st.session_state.symptoms['dark_skin_patches'] = st.checkbox(
            "Dark patches of skin (neck, armpits)", 
            value=st.session_state.symptoms.get('dark_skin_patches', False)
        )
    
    # Show current symptom count
    symptom_count = sum(st.session_state.symptoms.values())
    if symptom_count > 0:
        st.success(f"Tracking {symptom_count} symptom(s)")
        if symptom_count >= 3:
            st.warning("Multiple symptoms detected - consider medical consultation")
    
    st.markdown("---")
    st.info("💡 **Tip**: These symptoms can indicate diabetes, but may also be caused by other conditions. A medical evaluation is always recommended for persistent symptoms.")

def show_lifestyle_section():
    st.header("🏃‍♀️ Lifestyle & Health Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal & Family History")
        
        st.session_state.lifestyle['age'] = st.number_input(
            "Your age:", 
            min_value=18, max_value=100, 
            value=st.session_state.lifestyle.get('age', 30)
        )
        
        st.session_state.lifestyle['family_history'] = st.selectbox(
            "Family history of diabetes:",
            ["No family history", "Yes - Grandparent/Other", "Yes - Parent/Sibling"],
            index=["No family history", "Yes - Grandparent/Other", "Yes - Parent/Sibling"].index(
                st.session_state.lifestyle.get('family_history', "No family history")
            )
        )
        
        st.session_state.lifestyle['weight_status'] = st.selectbox(
            "Weight status (estimate):",
            ["Normal weight (BMI 18.5-25)", "Overweight (BMI 25-30)", "Obese (BMI > 30)", "Underweight (BMI < 18.5)"],
            index=["Normal weight (BMI 18.5-25)", "Overweight (BMI 25-30)", "Obese (BMI > 30)", "Underweight (BMI < 18.5)"].index(
                st.session_state.lifestyle.get('weight_status', "Normal weight (BMI 18.5-25)")
            )
        )
        
        st.session_state.lifestyle['exercise'] = st.selectbox(
            "Physical activity level:",
            ["Daily exercise", "3-4 times per week", "1-2 times per week", "Rarely/Never"],
            index=["Daily exercise", "3-4 times per week", "1-2 times per week", "Rarely/Never"].index(
                st.session_state.lifestyle.get('exercise', "1-2 times per week")
            )
        )
    
    with col2:
        st.subheader("Lifestyle Habits")
        
        st.session_state.lifestyle['diet'] = st.selectbox(
            "Diet quality:",
            ["Excellent (mostly whole foods)", "Good (balanced diet)", "Fair (some processed foods)", "Poor (frequent fast food, sugary drinks)"],
            index=["Excellent (mostly whole foods)", "Good (balanced diet)", "Fair (some processed foods)", "Poor (frequent fast food, sugary drinks)"].index(
                st.session_state.lifestyle.get('diet', "Good (balanced diet)")
            )
        )
        
        st.session_state.lifestyle['stress'] = st.selectbox(
            "Stress levels:",
            ["Low", "Moderate", "High"],
            index=["Low", "Moderate", "High"].index(
                st.session_state.lifestyle.get('stress', "Moderate")
            )
        )
        
        st.session_state.lifestyle['sleep'] = st.selectbox(
            "Sleep quality:",
            ["Excellent (7-9 hours, restful)", "Good (6-8 hours)", "Fair (5-7 hours)", "Poor (< 6 hours or frequent interruptions)"],
            index=["Excellent (7-9 hours, restful)", "Good (6-8 hours)", "Fair (5-7 hours)", "Poor (< 6 hours or frequent interruptions)"].index(
                st.session_state.lifestyle.get('sleep', "Good (6-8 hours)")
            )
        )
        
        st.session_state.lifestyle['smoking'] = st.selectbox(
            "Smoking status:",
            ["Never smoked", "Former smoker", "Current smoker"],
            index=["Never smoked", "Former smoker", "Current smoker"].index(
                st.session_state.lifestyle.get('smoking', "Never smoked")
            )
        )
    
    # Risk factor summary
    st.markdown("---")
    st.subheader("Your Risk Factor Summary:")
    
    risk_factors = []
    if st.session_state.lifestyle.get('age', 30) >= 45:
        risk_factors.append(f"Age {st.session_state.lifestyle['age']} (increased risk)")
    if "Parent/Sibling" in st.session_state.lifestyle.get('family_history', ''):
        risk_factors.append("Strong family history")
    if "Obese" in st.session_state.lifestyle.get('weight_status', ''):
        risk_factors.append("Weight management needed")
    if "Rarely/Never" in st.session_state.lifestyle.get('exercise', ''):
        risk_factors.append("Low physical activity")
    
    if risk_factors:
        for factor in risk_factors:
            st.warning(f"⚠️ {factor}")
    else:
        st.success("✅ Good lifestyle factors identified")

def show_demographics_section():
    st.header("👥 Demographics & Medical History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        
        st.session_state.demographics['ethnicity'] = st.selectbox(
            "Ethnicity (some groups have higher diabetes risk):",
            ["White/Caucasian", "African American", "Hispanic/Latino", "Asian", "Native American", "Pacific Islander", "Other/Mixed"],
            index=["White/Caucasian", "African American", "Hispanic/Latino", "Asian", "Native American", "Pacific Islander", "Other/Mixed"].index(
                st.session_state.demographics.get('ethnicity', "White/Caucasian")
            )
        )
        
        gender = st.selectbox(
            "Gender:",
            ["Male", "Female", "Prefer not to say"],
            index=["Male", "Female", "Prefer not to say"].index(
                st.session_state.demographics.get('gender', "Prefer not to say")
            )
        )
        st.session_state.demographics['gender'] = gender
    
    with col2:
        st.subheader("Medical History")
        
        # Show gender-specific questions
        if gender == "Female":
            st.session_state.demographics['gestational_diabetes'] = st.selectbox(
                "History of gestational diabetes:",
                ["No", "Yes", "Not applicable"],
                index=["No", "Yes", "Not applicable"].index(
                    st.session_state.demographics.get('gestational_diabetes', "Not applicable")
                )
            )
            
            st.session_state.demographics['pcos'] = st.selectbox(
                "History of PCOS (Polycystic Ovary Syndrome):",
                ["No", "Yes", "Not sure"],
                index=["No", "Yes", "Not sure"].index(
                    st.session_state.demographics.get('pcos', "No")
                )
            )
        else:
            st.session_state.demographics['gestational_diabetes'] = "Not applicable"
            st.session_state.demographics['pcos'] = "Not applicable"
        
        st.session_state.demographics['high_bp_history'] = st.selectbox(
            "History of high blood pressure:",
            ["No", "Yes", "Not sure"],
            index=["No", "Yes", "Not sure"].index(
                st.session_state.demographics.get('high_bp_history', "No")
            )
        )
        
        st.session_state.demographics['cholesterol_issues'] = st.selectbox(
            "History of high cholesterol:",
            ["No", "Yes", "Not sure"],
            index=["No", "Yes", "Not sure"].index(
                st.session_state.demographics.get('cholesterol_issues', "No")
            )
        )

def show_risk_assessment(predictor):
    st.header("📊 Your Diabetes Risk Assessment")
    
    # Check if user has completed enough information
    symptoms_completed = bool(st.session_state.symptoms)
    lifestyle_completed = bool(st.session_state.lifestyle)
    demographics_completed = bool(st.session_state.demographics)
    
    if not (symptoms_completed and lifestyle_completed and demographics_completed):
        st.warning("Please complete all sections (Symptoms, Lifestyle, Demographics) to get your risk assessment.")
        
        progress = sum([symptoms_completed, lifestyle_completed, demographics_completed]) / 3
        st.progress(progress)
        
        if not symptoms_completed:
            st.info("❌ Complete Symptoms Check section")
        if not lifestyle_completed:
            st.info("❌ Complete Lifestyle Factors section") 
        if not demographics_completed:
            st.info("❌ Complete Demographics section")
        
        return
    
    # Calculate risk
    risk_result = predictor.predict_risk(
        st.session_state.symptoms,
        st.session_state.lifestyle, 
        st.session_state.demographics
    )
    
    # Display results
    st.success("✅ Assessment Complete!")
    
    # Overall risk
    risk_category, risk_emoji, recommendation = get_risk_category(risk_result['probability'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Risk", f"{risk_emoji} {risk_category}")
    with col2:
        st.metric("Risk Score", f"{risk_result['total_score']:.0f}/100")
    with col3:
        st.metric("Probability", f"{risk_result['probability']:.1%}")
    
    # Detailed breakdown
    st.subheader("📈 Risk Factor Breakdown")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Symptom Score", f"{risk_result['symptom_score']:.0f}/100")
    with col2:
        st.metric("Lifestyle Score", f"{risk_result['lifestyle_score']:.0f}/100")
    with col3:
        st.metric("Demographic Score", f"{risk_result['demographic_score']:.0f}/100")
    
    # Visual representation
    st.subheader("🎯 Risk Composition")
    chart_data = pd.DataFrame({
        'Category': ['Symptoms', 'Lifestyle', 'Demographics'],
        'Score': [risk_result['symptom_score'], risk_result['lifestyle_score'], risk_result['demographic_score']]
    })
    st.bar_chart(chart_data.set_index('Category'))
    
    # Recommendations
    st.subheader("📋 Personalized Recommendations")
    st.info(recommendation)
    
    recommendations = generate_recommendations(risk_result, st.session_state.symptoms, st.session_state.lifestyle)
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Warning for high risk
    if risk_result['probability'] > 0.5:
        st.error("""
        ⚠️ **Important**: Your assessment indicates elevated diabetes risk. 
        This tool is for screening purposes only. Please consult a healthcare 
        professional for proper medical evaluation and blood tests.
        """)
    
    # Next steps
    st.subheader("🎯 Next Steps")
    
    if risk_result['probability'] < 0.25:
        st.success("""
        **Maintain Current Lifestyle**:
        - Continue healthy habits
        - Annual health check-ups
        - Stay aware of symptoms
        """)
    elif risk_result['probability'] < 0.5:
        st.warning("""
        **Preventive Action Recommended**:
        - Improve diet and exercise
        - Monitor symptoms closely
        - Consider annual diabetes screening
        - Discuss with healthcare provider
        """)
    else:
        st.error("""
        **Medical Consultation Recommended**:
        - Schedule appointment with doctor
        - Request diabetes blood tests (fasting glucose, HbA1c)
        - Discuss family history and symptoms
        - Consider diabetes prevention program
        """)

def generate_recommendations(risk_result, symptoms, lifestyle):
    """Generate personalized recommendations"""
    recommendations = []
    
    # Symptom-based recommendations
    if symptoms.get('frequent_urination') or symptoms.get('excessive_thirst'):
        recommendations.append("Monitor fluid intake and urination patterns - keep a daily log")
    
    if symptoms.get('unexplained_weight_loss'):
        recommendations.append("Track weight changes and discuss with healthcare provider")
    
    if symptoms.get('extreme_fatigue'):
        recommendations.append("Evaluate sleep quality and consider energy level tracking")
    
    # Lifestyle recommendations
    if "Obese" in lifestyle.get('weight_status', ''):
        recommendations.append("Focus on gradual weight loss (5-10% of body weight)")
    
    if "Rarely/Never" in lifestyle.get('exercise', ''):
        recommendations.append("Start with 150 minutes of moderate exercise per week")
    
    if "Poor" in lifestyle.get('diet', ''):
        recommendations.append("Reduce sugar and processed foods, increase vegetables and whole grains")
    
    if lifestyle.get('stress') == 'High':
        recommendations.append("Practice stress management techniques (meditation, yoga, deep breathing)")
    
    if "Poor" in lifestyle.get('sleep', ''):
        recommendations.append("Improve sleep hygiene - aim for 7-9 hours of quality sleep")
    
    # General recommendations
    recommendations.extend([
        "Stay hydrated with water instead of sugary drinks",
        "Learn about diabetes warning signs and risk factors",
        "Consider joining a diabetes prevention program if available",
        "Regular health screenings including blood pressure and cholesterol"
    ])
    
    return recommendations[:8]  # Limit to 8 recommendations

def show_learn_more():
    st.header("📚 Learn More About Diabetes")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Symptoms", "Risk Factors", "Prevention", "When to See a Doctor"])
    
    with tab1:
        st.subheader("Common Diabetes Symptoms")
        
        st.markdown("""
        **Early Warning Signs:**
        - **Frequent urination** (polyuria) - especially at night
        - **Excessive thirst** (polydipsia) - feeling constantly dehydrated
        - **Unexplained weight loss** - despite normal or increased appetite
        - **Extreme fatigue** - feeling tired even after rest
        - **Increased hunger** (polyphagia) - especially after eating
        
        **Secondary Symptoms:**
        - **Blurred vision** - difficulty focusing
        - **Slow-healing wounds** - cuts and bruises take longer to heal
        - **Frequent infections** - particularly skin, gum, and bladder infections
        - **Tingling in extremities** - numbness in hands or feet
        - **Dark skin patches** - acanthosis nigricans (neck, armpits)
        
        **Important Note:** Not everyone with diabetes experiences all symptoms, 
        and some people with Type 2 diabetes may have no symptoms initially.
        """)
    
    with tab2:
        st.subheader("Diabetes Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Non-Modifiable Risk Factors:**
            - **Age** - Risk increases after 45
            - **Family history** - Parents or siblings with diabetes
            - **Ethnicity** - Higher risk in certain groups
            - **Gestational diabetes** - Previous history
            - **PCOS** - Polycystic ovary syndrome
            """)
        
        with col2:
            st.markdown("""
            **Modifiable Risk Factors:**
            - **Weight** - Being overweight or obese
            - **Physical inactivity** - Sedentary lifestyle
            - **Diet** - High sugar, processed foods
            - **Blood pressure** - Hypertension
            - **Stress** - Chronic stress levels
            - **Sleep** - Poor sleep quality/quantity
            """)
    
    with tab3:
        st.subheader("Diabetes Prevention")
        
        st.markdown("""
        **Lifestyle Changes That Make a Difference:**
        
        **🥗 Nutrition:**
        - Choose whole grains over refined carbohydrates
        - Increase vegetable and fruit intake
        - Limit sugary drinks and processed foods
        - Control portion sizes
        - Consider the diabetes plate method
        
        **🏃‍♀️ Physical Activity:**
        - Aim for 150 minutes of moderate exercise weekly
        - Include both aerobic and strength training
        - Start slowly and gradually increase intensity
        - Find activities you enjoy to maintain consistency
        
        **⚖️ Weight Management:**
        - Even a 5-10% weight loss can significantly reduce risk
        - Focus on sustainable lifestyle changes
        - Consider working with a nutritionist
        
        **😴 Sleep & Stress:**
        - Prioritize 7-9 hours of quality sleep
        - Practice stress management techniques
        - Maintain regular sleep schedule
        """)
    
    with tab4:
        st.subheader("When to See a Healthcare Provider")
        
        st.error("""
        **Seek Immediate Medical Attention If:**
        - Multiple diabetes symptoms present
        - Symptoms are severe or worsening
        - Unexplained rapid weight loss
        - Frequent infections that don't heal
        - Vision changes or eye problems
        """)
        
        st.warning("""
        **Schedule a Check-up If:**
        - You have risk factors for diabetes
        - Family history of diabetes
        - Age 45 or older
        - Overweight with additional risk factors
        - Previous gestational diabetes
        - Haven't been screened in the past 3 years
        """)
        
        st.info("""
        **Diagnostic Tests Your Doctor May Order:**
        - **Fasting glucose test** - Blood sugar after 8+ hour fast
        - **HbA1c test** - Average blood sugar over 2-3 months
        - **Oral glucose tolerance test** - Blood sugar response to glucose drink
        - **Random glucose test** - Blood sugar at any time of day
        """)
        
        st.success("""
        **Remember:** Early detection and intervention can prevent or delay 
        Type 2 diabetes. Don't wait for symptoms to become severe.
        """)

if __name__ == "__main__":
    main()