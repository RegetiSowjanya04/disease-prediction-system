import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import sys
from PIL import Image
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model
from src.database import Database
from src.report_generator import generate_diagnosis_report

# Page configuration
st.set_page_config(
    page_title="MediPredict AI - Advanced Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - NO TOP GAP
st.markdown("""
<style>
    /* REMOVE TOP SPACE */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Hide Streamlit default header */
    header {
        display: none !important;
    }
    
    .stApp header {
        display: none !important;
    }
    
    .stApp > header {
        display: none !important;
    }
    
    .element-container:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glass morphism effect for cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0rem;
        color: white;
    }
    
    /* Login container */
    .login-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Prediction result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    /* Symptom badge */
    .symptom-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.25rem;
        display: inline-block;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Feature card */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Hide footer */
    footer {
        visibility: hidden;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    div:empty {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Load model
@st.cache_resource
def load_ml_model():
    try:
        model_path = 'models/disease_predictor.pkl'
        if os.path.exists(model_path):
            model_data = __import__('joblib').load(model_path)
            return model_data['model'], model_data['encoder'], model_data['feature_names']
        else:
            return None, None, None
    except Exception as e:
        return None, None, None

# AI Response Function - IMPROVED VERSION (WORKS FOR MANY TOPICS)
def get_ai_response(user_input):
    user_input_lower = user_input.lower()
    
    # Fever responses
    if any(word in user_input_lower for word in ['fever', 'temperature', 'high temp', 'high temperature', 'hot', 'warm body']):
        responses = [
            "🌡️ **Fever Information:**\n\nFever is usually a sign that your body is fighting an infection. Normal body temperature is 97°F to 99°F (36.1°C to 37.2°C).\n\n**When to see a doctor:**\n• Fever above 103°F (39.4°C)\n• Fever lasting more than 3 days\n• Accompanied by severe headache or rash\n\n**Home care:** Rest, stay hydrated, and take fever reducers as needed.",
            
            "🤒 **Managing Fever:**\n\n**Quick tips:**\n• Drink plenty of water\n• Get adequate rest\n• Use a cool compress on forehead\n• Monitor temperature regularly\n• Take lukewarm bath (not cold)\n\n**When to worry:** If fever exceeds 104°F or you have difficulty breathing, seek immediate medical attention.",
            
            "🩺 **Fever Dos and Don'ts:**\n\n**Do's:**\n✅ Rest adequately\n✅ Stay hydrated with water/ORS\n✅ Use light blankets\n✅ Take lukewarm bath\n✅ Eat light, nutritious food\n\n**Don'ts:**\n❌ Overdress or use heavy blankets\n❌ Skip meals\n❌ Ignore high fever\n❌ Self-medicate without guidance"
        ]
        return random.choice(responses)
    
    # Cough/Cold responses
    elif any(word in user_input_lower for word in ['cough', 'cold', 'sneeze', 'sneezing', 'runny nose', 'congestion', 'flu']):
        responses = [
            "😷 **About Cough & Cold:**\n\nCommon cold is a viral infection of upper respiratory tract. It usually resolves in 7-10 days.\n\n**Symptoms:** Runny nose, sneezing, cough, mild fever, body aches\n\n**Home remedies:**\n• Warm honey lemon tea\n• Steam inhalation with eucalyptus oil\n• Salt water gargle\n• Rest and hydration\n• Chicken soup\n\n**See a doctor if:** Symptoms last more than 10 days, high fever, or difficulty breathing.",
            
            "🤧 **Managing Cold Symptoms:**\n\n**Do's:**\n✅ Get plenty of rest\n✅ Drink warm fluids (herbal tea, soup)\n✅ Use humidifier\n✅ Wash hands frequently\n✅ Use saline nasal spray\n\n**Don'ts:**\n❌ Smoke or be around smoke\n❌ Drink alcohol\n❌ Share utensils/cups\n❌ Go to crowded places",
            
            "🍯 **Natural Cold Remedies:**\n\n• Ginger tea with honey and lemon\n• Chicken soup (proven to help!)\n• Warm salt water gargle\n• Steam with eucalyptus oil\n• Vitamin C rich foods (oranges, kiwi)\n• Zinc lozenges\n• Garlic in food\n• Turmeric milk\n\n**Rest is the best medicine!**"
        ]
        return random.choice(responses)
    
    # Headache/Migraine responses
    elif any(word in user_input_lower for word in ['headache', 'migraine', 'head pain', 'head ache', 'throbbing head']):
        responses = [
            "🤕 **About Headaches:**\n\nCommon triggers include stress, dehydration, lack of sleep, eye strain, or caffeine withdrawal.\n\n**Relief tips:**\n• Drink water (dehydration is common cause)\n• Rest in dark, quiet room\n• Apply cold or warm compress\n• Practice deep breathing\n• Gentle neck stretches\n\n**Warning signs (see doctor immediately):** Sudden severe headache, after head injury, with fever, confusion, or vision changes.",
            
            "💆 **Headache Management:**\n\n**Natural remedies:**\n• Peppermint oil on temples\n• Lavender oil for relaxation\n• Gentle neck and shoulder stretches\n• Adequate sleep (7-8 hours)\n• Regular meal times (don't skip)\n\n**Prevention:**\n• Stay hydrated throughout day\n• Manage stress with meditation\n• Regular exercise\n• Limit screen time\n• Maintain consistent sleep schedule",
            
            "🎯 **Types of Headaches:**\n\n**Tension headache:** Pressure around forehead/back of head\n**Migraine:** Throbbing, often one side, with nausea/light sensitivity\n**Cluster:** Severe pain around one eye, occurs in groups\n**Sinus:** Pain in forehead/cheeks with congestion\n\n**Track your triggers to prevent headaches!**"
        ]
        return random.choice(responses)
    
    # Stomach/Digestive responses
    elif any(word in user_input_lower for word in ['stomach', 'nausea', 'vomiting', 'diarrhea', 'loose motion', 'upset stomach', 'indigestion', 'gas', 'bloating']):
        responses = [
            "🏥 **Digestive Issues:**\n\nCommon causes include food poisoning, viral infection, indigestion, or stress.\n\n**Home care:**\n• Stay hydrated (water, ORS, coconut water)\n• BRAT diet (Bananas, Rice, Applesauce, Toast)\n• Avoid spicy/oily/fried food\n• Rest your stomach\n• Ginger tea for nausea\n\n**Seek medical help if:** Blood in vomit/stool, severe pain, signs of dehydration (dry mouth, no urination for 8+ hours), fever over 102°F.",
            
            "🍽️ **Stomach Care Tips:**\n\n**What to eat (when recovering):**\n• Ginger tea\n• Plain yogurt (probiotics)\n• Bone broth\n• Steamed vegetables\n• Oatmeal\n• Crackers\n\n**What to avoid:**\n• Dairy products (except yogurt)\n• Caffeine\n• Alcohol\n• Fried foods\n• Spicy foods\n• Sugary drinks",
            
            "💧 **Prevent Dehydration:**\n\n**Signs of dehydration:**\n• Dark yellow urine\n• Dry mouth and lips\n• Dizziness when standing\n• Less urination (less than 4 times/day)\n• Sunken eyes\n• Fatigue\n\n**Solution:** ORS (Oral Rehydration Solution), coconut water, clear broths, water with small amount of salt and sugar."
        ]
        return random.choice(responses)
    
    # Skin/Allergy responses
    elif any(word in user_input_lower for word in ['skin', 'rash', 'itch', 'allergy', 'hives', 'redness', 'bumps', 'eczema']):
        responses = [
            "🩹 **Skin Issues:**\n\nRashes and itching can be from allergies, infections, heat, or skin conditions like eczema.\n\n**Immediate care:**\n• Apply cool compress\n• Use fragrance-free moisturizer\n• Avoid scratching (can cause infection)\n• Take oatmeal bath\n• Wear loose, cotton clothes\n\n**When to see dermatologist:** Rash spreads rapidly, with fever, blisters, doesn't improve in a week, or covers large body area.",
            
            "🧴 **Skin Care Tips:**\n\n**Prevention:**\n• Use gentle, fragrance-free soaps\n• Moisturize immediately after bathing\n• Wear soft cotton clothes\n• Avoid known allergens\n• Stay hydrated\n\n**Natural remedies for mild itching:**\n• Aloe vera gel\n• Coconut oil\n• Cold compresses\n• Oatmeal paste\n• Baking soda bath",
            
            "⚠️ **Common Allergens to Watch:**\n\n• Pollen (seasonal)\n• Dust mites\n• Pet dander (cats, dogs)\n• Certain foods (nuts, eggs, shellfish)\n• Insect stings (bees, wasps)\n• Latex\n• Certain medications\n\n**Identify and avoid triggers!**"
        ]
        return random.choice(responses)
    
    # Diabetes responses
    elif any(word in user_input_lower for word in ['diabetes', 'blood sugar', 'sugar', 'diabetic', 'high sugar', 'glucose']):
        responses = [
            "🩸 **About Diabetes:**\n\nDiabetes affects how your body uses blood sugar (glucose).\n\n**Warning signs:**\n• Frequent urination, especially at night\n• Excessive thirst\n• Unexplained weight loss\n• Blurred vision\n• Slow healing wounds\n• Tingling/numbness in hands/feet\n• Extreme fatigue\n\n**Management:**\n• Monitor blood sugar regularly\n• Healthy diet (low sugar, complex carbs)\n• Regular exercise (30 min/day)\n• Take prescribed medications/insulin\n• Regular check-ups\n\n⚠️ **Important:** Diabetes requires proper medical management. Please consult an endocrinologist.",
            
            "🥗 **Diabetes-Friendly Diet Tips:**\n\n**Foods to eat:**\n• Leafy greens (spinach, kale)\n• Whole grains (brown rice, quinoa, oats)\n• Lean protein (chicken, fish, tofu)\n• Nuts and seeds\n• Berries (in moderation)\n• Beans and legumes\n\n**Foods to limit:**\n• Sugary drinks and sweets\n• White bread, white rice, pasta\n• Fried foods\n• Processed snacks\n• High-sugar fruits (mangoes, grapes)"
        ]
        return random.choice(responses)
    
    # Blood Pressure responses
    elif any(word in user_input_lower for word in ['bp', 'blood pressure', 'hypertension', 'high bp', 'low bp']):
        responses = [
            "❤️ **About Blood Pressure:**\n\nNormal BP is around 120/80 mmHg.\n\n**High BP (Hypertension) symptoms (often none - 'silent killer'):**\n• Severe headaches\n• Shortness of breath\n• Nosebleeds\n• Chest pain\n• Vision changes\n\n**Lifestyle changes for BP control:**\n• Reduce salt intake (less than 1 tsp/day)\n• Exercise regularly (30 min, 5 days/week)\n• Limit alcohol\n• Quit smoking\n• Manage stress with meditation\n• Maintain healthy weight\n\n⚠️ Regular monitoring is essential. Consult your doctor.",
            
            "🥬 **DASH Diet for Blood Pressure:**\n\n**Eat more:**\n• Vegetables (especially leafy greens)\n• Fruits (bananas, oranges, berries)\n• Whole grains\n• Low-fat dairy\n• Lean protein\n• Nuts and seeds\n\n**Limit:**\n• Salt/sodium\n• Red meat\n• Added sugars\n• Saturated fats\n• Alcohol"
        ]
        return random.choice(responses)
    
    # Doctor/Appointment responses
    elif any(word in user_input_lower for word in ['doctor', 'appointment', 'consult', 'specialist', 'clinic', 'hospital']):
        responses = [
            "📅 **Booking a Doctor Appointment:**\n\nYou can book an appointment through our system:\n\n1. Go to **'📅 Appointments'** in the sidebar\n2. Choose your preferred doctor from our list\n3. Select date and time\n4. Fill your personal details\n5. Click 'Book Appointment'\n\n**Available specialists:**\n• General Physician - for common illnesses\n• Cardiologist - for heart-related issues\n• Dermatologist - for skin problems\n• Pediatrician - for children\n\n**Emergency:** If you have severe symptoms, please visit your nearest emergency room immediately.",
            
            "🏥 **When to See a Doctor:**\n\n**See a doctor if you have:**\n• Fever >103°F lasting more than 3 days\n• Severe headache with stiff neck\n• Chest pain or difficulty breathing\n• Blood in vomit, stool, or urine\n• Sudden vision changes\n• Severe abdominal pain\n• Signs of stroke (face drooping, arm weakness, speech difficulty)\n\n**For routine check-ups:** Once a year for adults."
        ]
        return random.choice(responses)
    
    # Prevention/Health tips
    elif any(word in user_input_lower for word in ['prevent', 'precaution', 'avoid', 'healthy', 'tip', 'tips', 'wellness', 'fitness']):
        responses = [
            "💪 **General Health Tips:**\n\n**Daily healthy habits:**\n• Drink 8-10 glasses of water daily\n• Get 7-8 hours of quality sleep\n• Exercise 30 minutes daily (walking counts!)\n• Eat balanced meals with vegetables\n• Take breaks from screens every hour\n\n**Hygiene:**\n• Wash hands frequently with soap\n• Cover mouth when coughing/sneezing\n• Keep surroundings clean\n• Don't share personal items\n\n**Mental health:**\n• Practice mindfulness or meditation\n• Stay socially connected\n• Take time for hobbies",
            
            "🌿 **Immune System Boosters:**\n\n**Foods that boost immunity:**\n• Citrus fruits (Vitamin C)\n• Garlic and ginger\n• Turmeric\n• Yogurt (probiotics)\n• Almonds (Vitamin E)\n• Green tea\n• Broccoli\n\n**Lifestyle habits:**\n• Adequate sleep (7-8 hours)\n• Stress management\n• Regular exercise\n• Stay hydrated\n• No smoking\n• Limited alcohol\n• Regular check-ups\n• Stay up-to-date with vaccinations",
            
            "🥗 **Healthy Living Checklist:**\n\n✅ Eat colorful vegetables daily\n✅ Include protein in every meal\n✅ Limit processed and packaged foods\n✅ Take regular breaks from screens\n✅ Practice mindfulness or meditation\n✅ Stay socially connected\n✅ Get fresh air and sunlight\n✅ Stretch throughout the day\n\n**Small consistent changes = Big long-term results!**"
        ]
        return random.choice(responses)
    
    # Symptoms questions
    elif any(word in user_input_lower for word in ['symptom', 'symptoms', 'sign', 'signs', 'feeling', 'unwell', 'sick']):
        responses = [
            "🔍 **Common Symptoms Guide:**\n\n**Mild symptoms (home care):**\n• Low-grade fever (<101°F)\n• Mild cough\n• Mild headache\n• Fatigue\n• Runny nose\n• Mild sore throat\n\n**Moderate symptoms (see doctor within 24-48 hours):**\n• Persistent fever >3 days\n• Severe pain anywhere\n• Difficulty breathing\n• Dehydration signs\n• Persistent vomiting/diarrhea\n\n**Emergency symptoms (go to ER immediately):**\n• Chest pain\n• Difficulty breathing\n• Sudden severe headache\n• Loss of consciousness\n• Severe bleeding\n• Sudden vision changes\n• Signs of stroke\n\n**Use our 'Predict Disease' feature for personalized analysis!**",
            
            "📋 **When to Track Your Symptoms:**\n\nKeep a symptom diary if you have:\n• Chronic conditions (migraine, IBS, arthritis)\n• Recurring symptoms\n• Unexplained symptoms\n\n**What to track:**\n• When symptoms started\n• Duration and frequency\n• Severity (scale 1-10)\n• What makes it better/worse\n• Any triggers (food, stress, activity)\n• Any other symptoms occurring together\n\n**Share this diary with your doctor for better diagnosis!**"
        ]
        return random.choice(responses)
    
    # Thank you responses
    elif any(word in user_input_lower for word in ['thank', 'thanks', 'appreciate', 'helpful', 'good', 'great']):
        responses = [
            "🌟 You're welcome! I'm glad I could help. Remember, I'm here 24/7 for your health questions. Stay healthy! 💪\n\n**Would you like to:**\n• 🔍 Check your symptoms?\n• 📚 Learn about a specific disease?\n• 💪 Get more health tips?\n• 📅 Book an appointment?",
            
            "😊 Happy to help! Your health is important. Feel free to ask anything else!\n\n**Quick links to useful features:**\n• 🔍 Predict Disease - get diagnosis\n• 📅 Book Appointment - consult a doctor\n• 💊 Health Tips - prevention advice\n• 📊 History - view past diagnoses",
            
            "🙏 Thank you for reaching out! Taking charge of your health is the first step to wellness. I'm always here when you need me!\n\n**Stay healthy and take care!** 🌟"
        ]
        return random.choice(responses)
    
    # Greeting responses
    elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greeting', 'good morning', 'good evening', 'good afternoon', 'namaste']):
        responses = [
            "👋 Hello! Welcome to MediPredict AI Health Assistant. How can I help you today?\n\n**You can ask me about:**\n• 🩺 Symptoms and conditions (fever, cough, headache)\n• 💊 Treatments and home remedies\n• 🏥 When to see a doctor\n• 💪 Health tips and prevention\n• 📊 Understanding your diagnosis\n• 📅 Booking appointments\n\n**Or try our 'Predict Disease' feature for accurate diagnosis!**",
            
            "🌟 Hey there! I'm your AI health companion. I'm here to help you understand your health concerns better.\n\n**Try asking:**\n• 'What helps with fever?'\n• 'When to see a doctor for cough?'\n• 'Tips for healthy living'\n• 'Information about diabetes'\n\n**How are you feeling today?**",
            
            "🩺 Welcome to MediPredict AI! I'm your personal health assistant.\n\n**Quick tips for best results:**\n• Be specific about your symptoms\n• Mention how long you've had them\n• Use the 'Predict Disease' feature for diagnosis\n\n**What health concern can I help you with today?**"
        ]
        return random.choice(responses)
    
    # Medicine responses
    elif any(word in user_input_lower for word in ['medicine', 'medication', 'drug', 'tablet', 'pill', 'prescription', 'pharmacy']):
        responses = [
            "💊 **Medication Safety:**\n\n**Important rules:**\n• Always take prescribed dosage as directed\n• Complete full course of antibiotics (don't stop early)\n• Check expiration dates before taking\n• Store medications properly (cool, dry place)\n• Keep out of reach of children\n\n**Never:**\n• Share prescription medications with others\n• Take someone else's medicine\n• Crush or chew pills without consulting pharmacist\n• Mix with alcohol without medical advice\n• Double dose if you miss one\n\n⚠️ Always consult your doctor or pharmacist before starting or stopping any medication.",
            
            "💊 **Before Taking Any Medicine:**\n\n**Questions to ask your doctor/pharmacist:**\n• What is this medication for?\n• How and when should I take it?\n• How long will I need to take it?\n• What are the possible side effects?\n• Can I take it with my other medications?\n• Are there any foods/drinks to avoid?\n\n**Warning signs (stop medication and call doctor):**\n• Severe allergic reaction (rash, swelling, difficulty breathing)\n• Severe nausea or vomiting\n• Unusual bleeding or bruising\n• Dark urine or yellowing of eyes/skin"
        ]
        return random.choice(responses)
    
    # Exercise/Fitness responses
    elif any(word in user_input_lower for word in ['exercise', 'workout', 'fitness', 'gym', 'walking', 'running', 'yoga']):
        responses = [
            "🏃 **Exercise Guidelines:**\n\n**Recommended amount:**\n• 150 minutes moderate exercise weekly (brisk walking, swimming, cycling)\n• OR 75 minutes vigorous exercise weekly (running, HIIT)\n• Strength training 2x per week\n• Stretch before and after exercise\n\n**Benefits of regular exercise:**\n• Better heart health\n• Weight management\n• Improved mood and reduced stress\n• Better sleep quality\n• Stronger bones and muscles\n• Reduced risk of chronic diseases\n\n**Start slow and be consistent!** Even 10-minute walks count.\n\n**Listen to your body** - rest when needed, don't push through pain.",
            
            "🧘 **Beginner's Exercise Plan:**\n\n**Week 1-2:**\n• 10-15 minute walk daily\n• Gentle stretching 5 minutes\n\n**Week 3-4:**\n• 20-30 minute walk, 5 days/week\n• Basic bodyweight exercises: squats (10 reps), push-ups (5 reps), lunges (10 each leg)\n\n**Week 5-6:**\n• 30-40 minute brisk walk or jog\n• Add light weights (1-3 lbs) if available\n• Try yoga or pilates for flexibility\n\n**Always warm up before and cool down after exercise!**\n\n💪 **Remember:** Any movement is better than no movement!"
        ]
        return random.choice(responses)
    
    # Diet/Nutrition responses
    elif any(word in user_input_lower for word in ['diet', 'food', 'eat', 'nutrition', 'meal', 'cooking', 'recipe']):
        responses = [
            "🥗 **Healthy Eating Tips:**\n\n**Do's:**\n✅ Eat colorful fruits and vegetables (5+ servings/day)\n✅ Include protein in every meal (chicken, fish, eggs, beans, tofu)\n✅ Choose whole grains (brown rice, quinoa, oats, whole wheat)\n✅ Stay hydrated (water, herbal tea)\n✅ Eat regular meals (don't skip breakfast)\n✅ Practice portion control\n\n**Don'ts:**\n❌ Skip meals (leads to overeating later)\n❌ Too much added sugar\n❌ Excessive processed foods\n❌ Late night eating close to bedtime\n❌ Emotional eating\n\n**Small changes = Big results!** Start with one change at a time.",
            
            "🥑 **Sample Healthy Meal Plan:**\n\n**Breakfast (7-8 AM):** Oatmeal with berries and nuts OR Greek yogurt with fruit\n\n**Morning Snack (10-11 AM):** Apple or handful of almonds\n\n**Lunch (12-1 PM):** Grilled chicken salad with olive oil dressing OR quinoa bowl with vegetables\n\n**Afternoon Snack (3-4 PM):** Carrot sticks with hummus OR banana\n\n**Dinner (6-7 PM):** Baked salmon with roasted vegetables and brown rice OR lentil soup with whole grain bread\n\n**Evening Snack (if hungry):** Herbal tea or small piece of dark chocolate\n\n**Drink water throughout the day (8-10 glasses)!**"
        ]
        return random.choice(responses)
    
    # Anxiety/Stress responses
    elif any(word in user_input_lower for word in ['anxiety', 'stress', 'worried', 'nervous', 'panic', 'tension', 'overwhelmed']):
        responses = [
            "😌 **Managing Stress and Anxiety:**\n\n**Immediate relief techniques:**\n• Deep breathing: Inhale for 4 seconds, hold for 4, exhale for 4\n• 5-4-3-2-1 grounding: Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste\n• Step away from the situation for 5 minutes\n• Splash cold water on your face\n\n**Long-term management:**\n• Regular exercise (reduces stress hormones)\n• Adequate sleep (7-8 hours)\n• Limit caffeine and alcohol\n• Practice mindfulness or meditation\n• Talk to someone you trust\n• Keep a worry journal\n\n**When to seek professional help:** Anxiety interfering with daily life, panic attacks, or thoughts of self-harm.\n\n**You're not alone. Many people experience stress and anxiety.**",
            
            "🧘 **5-Minute Relaxation Exercise:**\n\n1. Find a quiet place to sit comfortably\n2. Close your eyes gently\n3. Take 5 deep breaths (in through nose, out through mouth)\n4. Notice the sensation of your breath\n5. Scan your body from head to toe, relaxing each part\n6. If your mind wanders, gently bring it back to your breath\n7. Open your eyes when ready\n\n**Practice this daily for better stress management!**\n\n💙 **Remember:** It's okay to not be okay. Reach out for support when needed."
        ]
        return random.choice(responses)
    
    # Sleep responses
    elif any(word in user_input_lower for word in ['sleep', 'insomnia', 'tired', 'fatigue', 'can sleep', 'sleeping', 'awake']):
        responses = [
            "😴 **Sleep Hygiene Tips:**\n\n**For better sleep:**\n• Maintain consistent sleep schedule (even weekends)\n• Create relaxing bedtime routine (read, gentle stretch, warm bath)\n• Keep bedroom dark, quiet, and cool (65-68°F)\n• Avoid screens 1 hour before bed (blue light disrupts sleep)\n• Avoid caffeine after 2 PM\n• Limit alcohol (disrupts sleep quality)\n• Exercise daily (but not right before bed)\n\n**How much sleep do you need?**\n• Adults: 7-9 hours\n• Teens: 8-10 hours\n• Children: 9-12 hours\n\n**If you can't sleep after 20 minutes:** Get up, do something relaxing in dim light, return to bed when sleepy.",
            
            "💤 **Signs You Need More Sleep:**\n\n• Difficulty waking up\n• Daytime sleepiness\n• Moodiness or irritability\n• Trouble concentrating\n• Increased errors or accidents\n• Craving unhealthy foods\n• Weakened immune system (getting sick often)\n\n**Improve your sleep for better overall health!**"
        ]
        return random.choice(responses)
    
    # Default response for unknown questions
    else:
        responses = [
            f"🤔 I understand you're asking about '{user_input}'. Could you provide more details?\n\n**I can help with:**\n• Specific symptoms you're experiencing (fever, cough, headache, stomach pain)\n• Disease information (diabetes, BP, cold, allergies)\n• Health concerns and when to see a doctor\n• Prevention tips and healthy living\n\n**Or try our 'Predict Disease' feature for personalized diagnosis based on your symptoms!**",
            
            "💡 **To help you better, please tell me:**\n\n• What symptoms are you experiencing?\n• How long have you had them?\n• How severe are they on a scale 1-10?\n• Any other symptoms along with these?\n\n**Alternatively, use our 'Predict Disease' tool for accurate analysis based on your symptoms.**\n\n**You can also ask me about specific topics like:**\n• 'What helps with fever?'\n• 'When to see a doctor for headache?'\n• 'Diabetes information'\n• 'Healthy eating tips'",
            
            "🩺 **I specialize in these health topics:**\n\n• 🤒 Fever, cough, cold, flu\n• 🤕 Headache, migraine\n• 🏥 Stomach issues, nausea, diarrhea\n• 🩹 Skin problems, rashes, allergies\n• 🩸 Diabetes, blood pressure\n• 💪 Exercise, fitness, healthy eating\n• 😌 Stress, anxiety, sleep\n\n**What would you like to know more about?**\n\n**For diagnosis, please use the 'Predict Disease' feature!**"
        ]
        return random.choice(responses)

# Login/Signup Page
def login_signup():
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 2.5rem; margin: 0;">🏥 MediPredict AI</h1>
        <p style="font-size: 1rem; margin: 0.3rem 0;">Advanced AI-Powered Disease Diagnosis System</p>
        <p style="font-size: 0.85rem; margin: 0;">98.5% Accuracy | 42 Diseases | 132 Symptoms</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        db = Database()
                        user = db.verify_user(username, password)
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="your@email.com")
                new_password = st.text_input("Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                submit = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit:
                    if not new_username or not new_email or not new_password:
                        st.warning("Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        db = Database()
                        user_id, error = db.create_user(new_username, new_email, new_password)
                        if user_id:
                            st.success("✅ Account created successfully! Please login.")
                        else:
                            st.error(f"❌ {error}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ✨ Why Choose MediPredict AI?")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 98.5% Accuracy</h3>
            <p>Highly reliable predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Real-time Results</h3>
            <p>Instant diagnosis</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Secure & Private</h3>
            <p>Your data is safe</p>
        </div>
        """, unsafe_allow_html=True)

# Main Dashboard
def main_dashboard():
    db = Database()
    model, encoder, feature_names = load_ml_model()
    
    with st.sidebar:
        st.markdown("### 🏥 MediPredict AI")
        st.markdown("---")
        st.markdown(f"### 👋 Welcome, **{st.session_state.user['username']}**!")
        st.markdown("---")
        
        menu_options = ["🏠 Dashboard", "🔍 Predict Disease", "📊 History", "📄 Reports", "💬 AI Assistant", "📅 Appointments", "ℹ️ Disease Library"]
        menu = st.radio("📋 Navigation", menu_options, index=0)
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()
    
    if menu == "🏠 Dashboard":
        st.markdown(f"""
        <div class="hero-section">
            <h2 style="margin: 0;">Welcome, {st.session_state.user['username']}!</h2>
            <p style="margin: 0.2rem 0;">Your AI Health Assistant is ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h2>🔍</h2>
                <h3>Start Diagnosis</h3>
                <p>Check symptoms</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h2>💬</h2>
                <h3>AI Chat</h3>
                <p>Ask questions</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h2>📅</h2>
                <h3>Book</h3>
                <p>Appointments</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🩺 How It Works")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h2>1</h2>
                <h3>Select Symptoms</h3>
                <p>Choose from 132+ symptoms</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h2>2</h2>
                <h3>AI Analysis</h3>
                <p>AI analyzes patterns</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h2>3</h2>
                <h3>Get Results</h3>
                <p>Instant diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif menu == "🔍 Predict Disease" and model:
        st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0;">🔍 Disease Prediction</h2>
            <p style="margin: 0.2rem 0;">Select your symptoms</p>
        </div>
        """, unsafe_allow_html=True)
        
        search = st.text_input("🔎 Search symptoms", placeholder="Type to search...")
        
        all_symptoms = feature_names
        if search:
            filtered = [s for s in all_symptoms if search.lower() in s.lower()]
        else:
            filtered = all_symptoms
        
        selected = st.multiselect("Select symptoms", filtered)
        
        if st.button("🔮 Analyze", use_container_width=True):
            if selected:
                with st.spinner("Analyzing..."):
                    input_vec = np.zeros(len(feature_names))
                    for sym in selected:
                        if sym in feature_names:
                            input_vec[feature_names.index(sym)] = 1
                    
                    probs = model.predict_proba([input_vec])[0]
                    top5 = np.argsort(probs)[-5:][::-1]
                    
                    top_disease = encoder.inverse_transform([top5[0]])[0]
                    top_conf = probs[top5[0]] * 100
                    
                    top_preds = {}
                    for idx in top5:
                        top_preds[encoder.inverse_transform([idx])[0]] = probs[idx] * 100
                    
                    db.save_prediction(st.session_state.user['id'], selected, top_disease, top_conf, top_preds)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>Diagnosis</h2>
                        <h1 style="font-size: 2.5rem; margin: 0.5rem 0;">{top_disease}</h1>
                        <div style="background: rgba(255,255,255,0.3); border-radius: 30px; height: 35px;">
                            <div style="background: #4CAF50; width: {top_conf}%; height: 35px; border-radius: 30px; text-align: center; line-height: 35px;">
                                {top_conf:.1f}% confidence
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, (d, p) in enumerate(top_preds.items(), 1):
                        st.progress(p/100, text=f"{i}. {d}: {p:.1f}%")
                    
                    if st.button("📄 Download Report"):
                        recommendations = [
                            f"Consult a healthcare provider regarding {top_disease}",
                            "Follow prescribed treatment plan",
                            "Schedule follow-up appointment",
                            "Maintain healthy lifestyle"
                        ]
                        report_path = generate_diagnosis_report(
                            st.session_state.user['username'], selected, 
                            top_disease, top_conf, top_preds, recommendations
                        )
                        with open(report_path, "rb") as f:
                            st.download_button("📥 Download PDF", f, file_name=report_path.split('/')[-1])
            else:
                st.warning("Select at least one symptom")
    
    elif menu == "📊 History":
        st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0;">📊 History</h2>
            <p style="margin: 0.2rem 0;">Your past diagnoses</p>
        </div>
        """, unsafe_allow_html=True)
        
        history = db.get_prediction_history(st.session_state.user['id'])
        if history:
            for h in history:
                with st.expander(f"📅 {h['date']} - {h['disease']}"):
                    st.write("**Symptoms:**", ", ".join(h['symptoms']))
                    st.write("**Diagnosis:**", h['disease'])
                    st.write("**Confidence:**", f"{h['confidence']:.1f}%")
        else:
            st.info("No history yet")
    
    elif menu == "📄 Reports":
        st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0;">📄 Reports</h2>
            <p style="margin: 0.2rem 0;">Download PDF reports</p>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists("reports"):
            reports = [f for f in os.listdir("reports") if f.endswith('.pdf')]
            if reports:
                for r in reports:
                    with open(f"reports/{r}", "rb") as f:
                        st.download_button(f"📄 {r}", f, file_name=r)
            else:
                st.info("No reports yet")
    
    elif menu == "💬 AI Assistant":
        st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0;">💬 AI Assistant</h2>
            <p style="margin: 0.2rem 0;">Ask me anything about health</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "👋 Hi! I'm MediPredict AI Assistant. Ask me about:\n\n• 🤒 Fever, cough, cold\n• 🤕 Headache, migraine\n• 🏥 Stomach issues\n• 🩹 Skin problems\n• 🩸 Diabetes, blood pressure\n• 💪 Health tips & prevention\n• 😌 Stress, anxiety, sleep\n\nWhat would you like to know?"}]
        
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if q := st.chat_input("Type your health question here..."):
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = get_ai_response(q)
                    st.markdown(resp)
            st.session_state.chat.append({"role": "assistant", "content": resp})
    
    elif menu == "📅 Appointments":
        st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0;">📅 Book Appointment</h2>
            <p style="margin: 0.2rem 0;">Consult a doctor</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form("apt_form"):
                name = st.text_input("Full Name", value=st.session_state.user['username'])
                phone = st.text_input("Phone")
                doctor = st.selectbox("Doctor", ["Dr. Smith (General)", "Dr. Johnson (Cardio)", "Dr. Williams (Skin)"])
                date = st.date_input("Date")
                time = st.selectbox("Time", ["10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM"])
                
                if st.form_submit_button("Book"):
                    if name and phone:
                        db.save_appointment(st.session_state.user['id'], name, st.session_state.user['email'], phone, doctor, str(date), time, "")
                        st.success(f"Booked with {doctor} on {date} at {time}")
                    else:
                        st.warning("Fill required fields")
        
        with col2:
            st.markdown("### Your Appointments")
            apts = db.get_appointments(st.session_state.user['id'])
            if apts:
                for a in apts:
                    st.markdown(f"**{a['doctor_name']}** - {a['date']} at {a['time']}")
            else:
                st.info("No appointments")
    
    elif menu == "ℹ️ Disease Library":
        st.markdown("""
        <div class="hero-section">
            <h2 style="margin: 0;">ℹ️ Disease Info</h2>
            <p style="margin: 0.2rem 0;">Common diseases</p>
        </div>
        """, unsafe_allow_html=True)
        
        info = {
            "🦠 Fungal Infection": "Caused by fungi. Keep skin clean and dry.",
            "🤧 Common Cold": "Viral infection. Rest and stay hydrated.",
            "🩸 Diabetes": "High blood sugar. Monitor and manage diet.",
            "❤️ Hypertension": "High blood pressure. Reduce salt intake.",
            "🤕 Migraine": "Severe headache. Rest in dark room."
        }
        
        for d, i in info.items():
            with st.expander(d):
                st.write(i)

# Main
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_signup()
    else:
        main_dashboard()