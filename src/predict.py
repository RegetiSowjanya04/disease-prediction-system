import numpy as np
from utils import load_model

def predict_disease(symptoms_list, model_path='../models/disease_predictor.pkl'):
    """
    Predict disease based on symptoms
    
    Parameters:
    symptoms_list: list of symptom names (e.g., ['itching', 'skin_rash', 'fatigue'])
    
    Returns:
    disease_name: predicted disease
    top_3: dictionary of top 3 diseases with probabilities
    """
    # Load model and encoder
    model, encoder, feature_names = load_model(model_path)
    
    # Create input vector (all symptoms = 0 initially)
    input_vector = np.zeros(len(feature_names))
    
    # Set 1 for symptoms that are present
    valid_symptoms = []
    for symptom in symptoms_list:
        if symptom in feature_names:
            idx = feature_names.index(symptom)
            input_vector[idx] = 1
            valid_symptoms.append(symptom)
        else:
            print(f"⚠️ Warning: '{symptom}' is not a valid symptom name")
    
    if len(valid_symptoms) == 0:
        print("❌ No valid symptoms selected!")
        return None, None
    
    # Get prediction probabilities
    probabilities = model.predict_proba([input_vector])[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = encoder.inverse_transform(top_3_idx)
    top_3_probs = probabilities[top_3_idx]
    
    # Get best prediction
    predicted_idx = np.argmax(probabilities)
    disease_name = encoder.inverse_transform([predicted_idx])[0]
    
    return disease_name, dict(zip(top_3_diseases, top_3_probs))

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("🩺 DISEASE PREDICTION SYSTEM")
    print("="*60)
    
    # Example symptoms (you can change these)
    example_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
    
    print(f"\n📋 Symptoms: {example_symptoms}")
    disease, top_3 = predict_disease(example_symptoms)
    
    if disease:
        print(f"\n🎯 Predicted Disease: {disease}")
        print("\n📊 Top 3 possibilities:")
        for i, (d, prob) in enumerate(top_3.items(), 1):
            print(f"  {i}. {d}: {prob*100:.2f}%")
    
    print("\n" + "="*60)