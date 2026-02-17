# app.py - This is the main file for Hugging Face Spaces
import gradio as gr
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Predict heart disease risk based on 13 health features.
    Returns prediction, probability, and risk message.
    """
    # Create feature array in the correct order
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the features (IMPORTANT!)
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    # Create result message
    if prediction == 1:
        if probability > 0.8:
            risk_level = "🚨 VERY HIGH RISK"
            message = "Immediate medical consultation is strongly recommended."
        elif probability > 0.6:
            risk_level = "⚠️ HIGH RISK"
            message = "Please schedule a doctor's appointment soon."
        else:
            risk_level = "⚡ MODERATE-HIGH RISK"
            message = "Consider lifestyle changes and regular monitoring."
    else:
        if probability < 0.2:
            risk_level = "💚 VERY LOW RISK"
            message = "Keep up the healthy lifestyle!"
        else:
            risk_level = "💙 LOW RISK"
            message = "Continue with regular check-ups."
    
    result = f"""
## {risk_level}

**Probability of Heart Disease:** {probability*100:.1f}%

**Recommendation:** {message}

---
*This is a prediction tool and should not replace professional medical advice.*
"""
    return result

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=50, info="Patient's age in years"),
        gr.Radio(choices=[0, 1], label="Sex", value=1, info="0 = Female, 1 = Male"),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Chest Pain Type (cp)", value=0, 
                   info="0=Typical angina, 1=Atypical angina, 2=Non-anginal, 3=Asymptomatic"),
        gr.Number(label="Resting Blood Pressure (trestbps)", value=120, info="mm Hg"),
        gr.Number(label="Cholesterol (chol)", value=200, info="mg/dl"),
        gr.Radio(choices=[0, 1], label="Fasting Blood Sugar > 120 mg/dl (fbs)", value=0, 
                info="0 = No, 1 = Yes"),
        gr.Dropdown(choices=[0, 1, 2], label="Resting ECG (restecg)", value=0,
                   info="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy"),
        gr.Number(label="Max Heart Rate (thalach)", value=150, info="beats per minute"),
        gr.Radio(choices=[0, 1], label="Exercise Induced Angina (exang)", value=0,
                info="0 = No, 1 = Yes"),
        gr.Number(label="ST Depression (oldpeak)", value=0.0, info="Induced by exercise"),
        gr.Dropdown(choices=[0, 1, 2], label="ST Slope", value=0,
                   info="0=Upsloping, 1=Flat, 2=Downsloping"),
        gr.Dropdown(choices=[0, 1, 2, 3, 4], label="Number of Major Vessels (ca)", value=0,
                   info="Colored by fluoroscopy (0-4)"),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Thalassemia (thal)", value=1,
                   info="0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Unknown"),
    ],
    outputs=gr.Markdown(),
    title="🫀 Heart Disease Risk Predictor",
    description="""
Enter patient health data below to predict heart disease risk using a Random Forest ML model.

**Note:** This tool is for educational purposes only. Always consult a healthcare professional.
""",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
