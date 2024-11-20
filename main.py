import streamlit as st
import pandas as pd
import numpy as np
import dill  # Use dill to load the model
import plotly.graph_objects as go

# Custom CSS for UI Enhancements
st.markdown("""
<style>
    .custom-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #424242;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .results-container {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2196F3;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        color: #4CAF50;
        text-align: center;
    }
    .description-text {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .type-info:hover {
        background-color: #f0f8ff;
        transform: scale(1.03);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<div class="custom-title">🎭 Personality Type Predictor</div>', unsafe_allow_html=True)
st.markdown('<p class="description-text">Rate yourself to discover your personality type instantly!</p>', unsafe_allow_html=True)

# Define feature descriptions and input
feature_descriptions = {
    "Stress Level": "How often do you feel stressed? (1: Rarely, 10: Very Often)",
    "Social Activity": "How often do you enjoy hanging out? (1: Rarely, 10: Very Often)",
    "Value of Humor": "How much do you value humor? (1: Not at all, 10: Extremely Important)",
    "Club Participation": "Likelihood of joining clubs/societies (1: Unlikely, 10: Very Likely)",
    "Exploring New Places": "How much do you enjoy exploring new places? (1: Not at all, 10: Love it)",
    "Physical Activities": "Involvement in physical activities? (1: Not at all, 10: Very Active)",
    "Reading Interest": "How much do you enjoy reading? (1: Not at all, 10: Love it)",
    "Gaming Interest": "Interest in video games (1: Not Interested, 10: Very Interested)",
    "Movies/Series Interest": "Enjoyment of movies/series (1: Rarely, 10: Very Often)",
    "Conversation Depth": "Preference for conversation depth (1: Casual, 10: Deep Talks)"
}

st.markdown("### Rate Yourself")

# Collapsible inputs for better layout
user_input = []
with st.expander("Click here to rate yourself on the scales:"):
    for feature, description in feature_descriptions.items():
        st.markdown(f'<p class="feature-desc">{description}</p>', unsafe_allow_html=True)
        value = st.slider(
            feature,
            min_value=1,
            max_value=10,
            help=description
        )
        user_input.append(value)

# Predict Button
if st.button("Predict Personality Type"):
    # Load model and predict
    try:
        with open('model.joblib', 'rb') as model_file:
            model = dill.load(model_file)  # Use dill to load the model
        
        input_data = np.array([user_input])
        predictions = model.predict(input_data)[0]  # Get probabilities

        predicted_class = np.argmax(predictions)
        class_labels = {0: "Introvert", 1: "Ambivert", 2: "Extrovert"}
        personality = class_labels[predicted_class]
        confidence = predictions[predicted_class] * 100

        # Display Results
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prediction-result">You are most likely:<br><strong>{personality}</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="confidence-score">Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

        # Donut Chart for Probability Distribution
        fig = go.Figure(data=[go.Pie(
            labels=list(class_labels.values()),
            values=predictions * 100,
            hole=.5,
            marker_colors=['#FF9800', '#2196F3', '#4CAF50']
        )])
        fig.update_layout(
            title="Personality Type Probability Distribution",
            height=300,
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

# Additional Information Section
st.markdown("""
<div class="results-container">
    <h3>Understanding the Personality Types</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div class="type-info" style="padding: 1rem; border-radius: 8px; background-color: #FFF3E0;">
            <h4 style="color: #E65100;">Introvert</h4>
            <p>Quiet, prefer minimal stimulation. Recharge through solitude. Often thoughtful and observant.</p>
        </div>
        <div class="type-info" style="padding: 1rem; border-radius: 8px; background-color: #E3F2FD;">
            <h4 style="color: #1565C0;">Ambivert</h4>
            <p>Balanced personality. Enjoys socializing and personal time equally. Adaptable to situations.</p>
        </div>
        <div class="type-info" style="padding: 1rem; border-radius: 8px; background-color: #E8F5E9;">
            <h4 style="color: #2E7D32;">Extrovert</h4>
            <p>Outgoing, energized by interactions. Thrives in group settings. Highly expressive and social.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #F5F5F5; margin-top: 2rem; border-radius: 10px;">
    <p>Created with ❤️ using Streamlit | Empowering Self-Discovery</p>
</div>
""", unsafe_allow_html=True)
