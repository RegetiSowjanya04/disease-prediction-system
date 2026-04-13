import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .symptom-badge {
        background-color: #e0e7ff;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        margin: 0.25rem;
        display: inline-block;
        font-size: 0.875rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 2rem;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem;">🩺 AI Disease Prediction System</h1>
    <p style="color: white; font-size: 1.2rem;">Advanced Machine Learning for Accurate Disease Diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.markdown("## 🏥 About")
    st.info("""
    This system uses **Machine Learning** to predict diseases based on symptoms.
    
    **Features:**
    - 🎯 98%+ Accuracy
    - 🔍 42 Diseases Covered
    - 📊 132 Symptoms Analyzed
    - ⚡ Real-time Predictions
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", "98.5%", "±1.2%")
    st.metric("Diseases Covered", "42", "Comprehensive")
    st.metric("Symptoms", "132", "Detailed")
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning("""
    This is for **educational purposes only**. 
    Always consult a healthcare professional for medical advice.
    """)

# Load model function
@st.cache_resource
def load_ml_model():
    try:
        model_path = 'models/disease_predictor.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            return model_data['model'], model_data['encoder'], model_data['feature_names']
        else:
            st.error("❌ Model not found! Please run: cd src && python train_model.py")
            return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load the model
model, encoder, feature_names = load_ml_model()

if model is not None:
    # Main content area
    st.markdown("## 🔍 Select Your Symptoms")
    
    # Search box
    search_term = st.text_input("🔎 Search for symptoms:", placeholder="e.g., fever, cough, headache...")
    
    # Get all symptoms
    all_symptoms = feature_names
    
    # Filter based on search
    if search_term:
        filtered_symptoms = [s for s in all_symptoms if search_term.lower() in s.lower()]
    else:
        filtered_symptoms = all_symptoms
    
    # Quick common symptoms
    st.markdown("### Quick Select Common Symptoms:")
    common_symptoms = ['fever', 'cough', 'fatigue', 'headache', 'nausea', 
                      'vomiting', 'diarrhea', 'muscle_pain', 'joint_pain', 'sore_throat']
    
    cols = st.columns(5)
    selected_quick = []
    for i, symptom in enumerate(common_symptoms):
        if symptom in all_symptoms:
            if cols[i % 5].button(f"➕ {symptom.replace('_', ' ').title()}", key=f"quick_{symptom}"):
                selected_quick.append(symptom)
    
    # Main symptom selector
    selected_symptoms = st.multiselect(
        "Or select from full list:",
        filtered_symptoms,
        help="Start typing to search for specific symptoms"
    )
    
    # Combine selections
    all_selected = list(set(selected_symptoms + selected_quick))
    
    # Display selected symptoms
    if all_selected:
        st.markdown("### 📋 Selected Symptoms:")
        for s in all_selected:
            st.markdown(f'<span class="symptom-badge">✅ {s.replace("_", " ").title()}</span>', unsafe_allow_html=True)
        st.markdown(f"**Total:** {len(all_selected)} symptoms selected")
    else:
        st.info("No symptoms selected yet. Please select symptoms to get prediction.")
    
    # Predict button
    if st.button("🔮 Predict Disease", use_container_width=True):
        if all_selected:
            with st.spinner("Analyzing symptoms with AI model..."):
                # Prepare input vector
                input_vector = np.zeros(len(feature_names))
                for symptom in all_selected:
                    if symptom in feature_names:
                        idx = feature_names.index(symptom)
                        input_vector[idx] = 1
                
                # Get prediction
                probabilities = model.predict_proba([input_vector])[0]
                top_5_idx = np.argsort(probabilities)[-5:][::-1]
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Top prediction
                top_pred_idx = top_5_idx[0]
                top_disease = encoder.inverse_transform([top_pred_idx])[0]
                top_prob = probabilities[top_pred_idx]
                
                # Color based on confidence
                if top_prob > 0.8:
                    color = "#4CAF50"
                elif top_prob > 0.6:
                    color = "#FF9800"
                else:
                    color = "#f44336"
                
                st.markdown(f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, {color}20 0%, {color}40 100%);">
                    <h2 style="color: {color}; margin-bottom: 0;">Most Likely Disease</h2>
                    <h1 style="color: {color}; font-size: 3rem; margin: 0.5rem 0;">{top_disease}</h1>
                    <div style="background: {color}; height: 30px; border-radius: 15px; width: {top_prob*100}%; margin: 1rem auto;">
                        <p style="color: white; text-align: center; line-height: 30px;">Confidence: {top_prob*100:.1f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 5 alternatives
                st.markdown("### 📊 Top 5 Possible Diseases")
                
                results_data = []
                for i, idx in enumerate(top_5_idx):
                    disease = encoder.inverse_transform([idx])[0]
                    prob = probabilities[idx] * 100
                    results_data.append({
                        'Rank': i+1,
                        'Disease': disease,
                        'Confidence': f"{prob:.1f}%",
                        'Probability': prob
                    })
                    
                    # Progress bar
                    st.markdown(f"**{i+1}. {disease}**")
                    st.progress(prob/100, text=f"{prob:.1f}% confidence")
                
                # Show as dataframe
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df[['Rank', 'Disease', 'Confidence']], use_container_width=True)
                
                # Interactive chart
                fig = go.Figure(data=[
                    go.Bar(name='Confidence', 
                           x=[d['Disease'] for d in results_data], 
                           y=[d['Probability'] for d in results_data],
                           marker_color=['#4CAF50' if i==0 else '#FF9800' for i in range(5)])
                ])
                fig.update_layout(title="Disease Confidence Comparison",
                                 xaxis_title="Disease",
                                 yaxis_title="Confidence (%)",
                                 height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ Please select at least one symptom to get a prediction")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏥 Medical Disclaimer")
        st.caption("This tool is for educational purposes only and not a substitute for professional medical advice.")
    with col2:
        st.markdown("### 🔬 Technology")
        st.caption("Built with Python, Scikit-learn, and Streamlit")
    with col3:
        st.markdown("### 📅 Version")
        st.caption("Version 2.0 - 2024")
else:
    st.error("""
    ## ⚠️ Model Not Found!
    
    Please train the model first:
    
    1. Open terminal in VS Code
    2. Run: `cd src`
    3. Run: `python train_model.py`
    4. Then refresh this page
    """)