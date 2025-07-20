import streamlit as st
import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add the model directory to Python path
model_dir = Path(__file__).parent / "model"
sys.path.insert(0, str(model_dir))

# Import prediction functions
try:
    from model.predict import load_models, engineer_features, make_predictions, calculate_confidence, get_feature_importance
except ImportError:
    st.error("Could not import prediction model. Please ensure model files are available.")
    st.stop()

def calculate_sentiment_score(news_headline: str, market_data: dict) -> float:
    """Calculate sentiment score based on news headline and market data"""
    if not news_headline or news_headline.strip() == "":
        return 0.0
    
    headline = news_headline.lower()
    score = 0
    
    positive_words = [
        'bull', 'bullish', 'rise', 'rising', 'increase', 'up', 'gain', 'gains', 
        'growth', 'positive', 'surge', 'rally', 'boom', 'breakthrough', 'adoption',
        'institutional', 'investment', 'buy', 'buying', 'support', 'strong',
        'record', 'high', 'milestone', 'success', 'approve', 'approved'
    ]
    
    negative_words = [
        'bear', 'bearish', 'fall', 'falling', 'decrease', 'down', 'drop', 'crash',
        'decline', 'negative', 'sell', 'selling', 'dump', 'fear', 'uncertainty',
        'regulation', 'ban', 'banned', 'hack', 'hacked', 'scam', 'fraud',
        'low', 'bottom', 'concern', 'warning', 'risk', 'volatile', 'bubble'
    ]
    
    positive_count = sum(1 for word in positive_words if word in headline)
    negative_count = sum(1 for word in negative_words if word in headline)
    
    word_sentiment = (positive_count - negative_count) * 0.2
    score += word_sentiment
    
    # Market momentum analysis
    price_range = market_data['high_price'] - market_data['low_price']
    relative_range = price_range / market_data['open_price'] if market_data['open_price'] > 0 else 0
    
    if relative_range > 0.05:
        score += -0.1 if 'volatile' in headline else 0.05
    
    # Volume analysis
    volume_factor = min(market_data['volume'] / 10000, 0.1)
    if positive_count > negative_count:
        score += volume_factor
    elif negative_count > positive_count:
        score -= volume_factor
    
    # Crypto-specific keywords
    crypto_positive = ['bitcoin', 'btc', 'cryptocurrency', 'blockchain', 'etf', 'halving']
    crypto_negative = ['regulation', 'tax', 'government', 'central bank']
    
    for word in crypto_positive:
        if word in headline and positive_count > 0:
            score += 0.1
    
    for word in crypto_negative:
        if word in headline:
            score -= 0.15
    
    # Normalize to [-1, 1]
    score = max(min(score, 1), -1)
    
    # Add randomization for non-perfect scores
    if abs(score) == 1.0:
        randomFactor = 0.85 + np.random.random() * 0.14
        score *= randomFactor
    
    return round(score, 2)

def load_custom_css():
    """Load custom CSS to match Next.js design exactly"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #1e40af 50%, #312e81 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 90rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    
    /* Header */
    .header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem 2rem;
        margin-bottom: 0;
        position: sticky;
        top: 0;
        z-index: 50;
    }
    
    /* Form and Results containers */
    .form-card, .results-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #f7931a !important;
        box-shadow: 0 0 0 2px rgba(247, 147, 26, 0.2) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.25rem 2rem !important;
        font-size: 1.125rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 40px -3px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Labels */
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #bfdbfe !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(59, 130, 246, 0.2);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 50%, #dc2626 100%);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 50px !important;
    }
    
    .stProgress .st-bp {
        background: linear-gradient(90deg, #f7931a, #ff6b35) !important;
        border-radius: 50px !important;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar (if used) */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    /* Text colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: white !important;
    }
    
    .stMarkdown p {
        color: #bfdbfe !important;
    }
    
    /* Custom classes for specific sections */
    .hero-section {
        text-align: center;
        padding: 4rem 0 2rem 0;
    }
    
    .section-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 50%, #f7931a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .section-subtitle {
        font-size: 1.25rem;
        color: #bfdbfe;
        text-align: center;
        max-width: 48rem;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render header using Streamlit components"""
    st.markdown("""
    <div class="header">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%); padding: 12px; border-radius: 16px; color: white; font-size: 1.5rem; font-weight: bold;">₿</div>
            <div>
                <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: white;">Bitcoin Predictor</h1>
                <p style="margin: 0; font-size: 0.75rem; color: #bfdbfe; font-weight: 500;">AI-Powered Predictions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Bitcoin Price Predictor",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    render_header()
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div style="display: inline-flex; align-items: center; padding: 0.5rem 1rem; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); border-radius: 50px; color: white; font-size: 0.875rem; font-weight: 500; margin-bottom: 2rem;">
            ⚡ Real-time Bitcoin Prediction
        </div>
        <h1 class="section-title">Bitcoin Price Prediction</h1>
        <p class="section-subtitle">
            Harness the power of advanced machine learning and economic indicators to predict Bitcoin prices with automatic sentiment analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Market Data Input Form
        with st.container():
            st.markdown("""
            <div class="form-card">
                <div style="display: flex; align-items: center; justify-between; margin-bottom: 2rem;">
                    <div style="display: flex; align-items: center;">
                        <div style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%); padding: 12px; border-radius: 16px; margin-right: 1rem; color: white; font-size: 1.25rem;">📈</div>
                        <h2 style="margin: 0; font-size: 1.875rem; font-weight: 800; color: white;">Market Data Input</h2>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 50px; padding: 0.25rem 0.75rem; font-size: 0.75rem; color: #bfdbfe; font-weight: 500;">
                        ✨ AI Powered
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("prediction_form"):
                # First row
                input_col1, input_col2 = st.columns(2)
                with input_col1:
                    st.markdown("**💰 Open Price (USD)**")
                    open_price = st.number_input("Open Price", min_value=0.01, value=45000.0, step=100.0, label_visibility="collapsed", key="open")
                
                with input_col2:
                    st.markdown("**📈 High Price (USD)**")
                    high_price = st.number_input("High Price", min_value=0.01, value=46000.0, step=100.0, label_visibility="collapsed", key="high")
                
                # Second row
                input_col3, input_col4 = st.columns(2)
                with input_col3:
                    st.markdown("**📉 Low Price (USD)**")
                    low_price = st.number_input("Low Price", min_value=0.01, value=44000.0, step=100.0, label_visibility="collapsed", key="low")
                
                with input_col4:
                    st.markdown("**📊 Volume (BTC)**")
                    volume = st.number_input("Volume", min_value=0.0, value=1500.0, step=100.0, label_visibility="collapsed", key="volume")
                
                # News headline
                st.markdown("**💬 News Headline (Optional)** *📊 Sentiment calculated automatically*")
                news_headline = st.text_area("News Headline", placeholder="Enter relevant news headline for sentiment analysis...", label_visibility="collapsed", height=100)
                
                # Submit button
                submitted = st.form_submit_button("🚀 Predict Bitcoin Price")
    
    with col2:
        # Prediction Results
        st.markdown("""
        <div class="results-card">
            <div style="display: flex; align-items: center; margin-bottom: 2rem;">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 12px; border-radius: 16px; margin-right: 1rem; color: white; font-size: 1.25rem;">🎯</div>
                <h2 style="margin: 0; font-size: 1.875rem; font-weight: 800; color: white;">Prediction Results</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if submitted:
            # Validation
            if high_price < low_price:
                st.error("❌ High price cannot be less than low price!")
                st.stop()
            
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or volume < 0:
                st.error("❌ Price values must be positive and volume must be non-negative!")
                st.stop()
            
            # Loading state
            with st.spinner("🔄 Analyzing market data and making predictions..."):
                time.sleep(1)  # Brief pause for UX
                
                try:
                    # Calculate sentiment
                    market_data = {
                        'open_price': open_price,
                        'high_price': high_price, 
                        'low_price': low_price,
                        'volume': volume
                    }
                    
                    sentiment_score = calculate_sentiment_score(news_headline, market_data)
                    
                    # Prepare input data
                    input_data = {
                        'open_price': open_price,
                        'close_price': open_price,
                        'high_price': high_price,
                        'low_price': low_price,
                        'volume': volume,
                        'sentiment_score': sentiment_score
                    }
                    
                    # Load models and make prediction
                    models = load_models()
                    features = engineer_features(input_data)
                    predictions = make_predictions(models, features)
                    confidence = calculate_confidence(predictions, features)
                    feature_importance = get_feature_importance(models)
                    
                    # Main prediction display
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="position: relative; z-index: 1;">
                            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                                <span style="margin-right: 0.5rem; font-size: 1.25rem;">⚡</span>
                                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600;">Predicted Bitcoin Price</h3>
                            </div>
                            <div class="prediction-price">
                                ${predictions['meta_model']:,.2f}
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(8px); border-radius: 50px; padding: 0.75rem 1.5rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.5rem;">
                                <span>🏆</span>
                                <span>Confidence: {confidence*100:.1f}%</span>
                            </div>
                            <div style="margin-top: 1rem; font-size: 0.875rem;">
                                Sentiment Score: {sentiment_score:.2f}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model Performance Metrics
                    st.subheader("📈 Model Performance")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown("""
                        <div class="metric-container" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 100%);">
                            <div style="font-size: 0.875rem; color: #bfdbfe; font-weight: 500; margin-bottom: 0.5rem;">RMSE</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: white;">97.08</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown("""
                        <div class="metric-container" style="background: linear-gradient(135deg, rgba(147, 51, 234, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);">
                            <div style="font-size: 0.875rem; color: #bfdbfe; font-weight: 500; margin-bottom: 0.5rem;">MAE</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: white;">52.16</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown("""
                        <div class="metric-container" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);">
                            <div style="font-size: 0.875rem; color: #bfdbfe; font-weight: 500; margin-bottom: 0.5rem;">R² Score</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: white;">0.995</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Individual Model Predictions
                    st.subheader("🔍 Individual Model Predictions")
                    pred_data = {
                        'Model': ['Random Forest', 'Ridge Regression', 'XGBoost', 'Meta Model'],
                        'Prediction': [
                            f"${predictions['random_forest']:,.2f}",
                            f"${predictions['ridge']:,.2f}",
                            f"${predictions['xgboost']:,.2f}",
                            f"${predictions['meta_model']:,.2f}"
                        ]
                    }
                    pred_df = pd.DataFrame(pred_data)
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    # Feature Importance
                    st.subheader("🎯 Key Feature Importance")
                    for i, item in enumerate(feature_importance[:8]):
                        importance_pct = item['importance']
                        feature_name = item['feature'].replace('_', ' ').title()
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="color: #bfdbfe; font-weight: 500;">{feature_name}</span>
                                <span style="color: white; font-weight: 600;">{importance_pct*100:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(importance_pct, text="")
                    
                    st.success("✅ Prediction completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
        
        else:
            # Default state
            st.markdown("""
            <div style="text-align: center; padding: 4rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">🎯</div>
                <h3 style="color: white; margin-bottom: 1rem;">Ready to predict Bitcoin prices</h3>
                <p style="color: #bfdbfe;">Enter market data to get started</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(8px); padding: 5rem 0; margin: 4rem 0; border-radius: 24px;">
        <div style="text-align: center; margin-bottom: 4rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Powered by Advanced AI
            </h2>
            <p style="font-size: 1.25rem; color: #bfdbfe; max-width: 32rem; margin: 0 auto;">
                Our sophisticated ensemble model combines multiple algorithms for superior accuracy
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%); border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);">
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 5rem; height: 5rem; margin: 0 auto 1.5rem auto; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                🧠
            </div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Machine Learning
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                Ensemble model combining Random Forest, XGBoost, and Ridge Regression for optimal predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);">
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 5rem; height: 5rem; margin: 0 auto 1.5rem auto; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                💾
            </div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Economic Data
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                Trained on comprehensive economic indicators with automated news sentiment analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%); border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);">
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 5rem; height: 5rem; margin: 0 auto 1.5rem auto; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                ⚡
            </div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Real-time
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                Get instant predictions with high confidence scores based on current market conditions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(20px); border-top: 1px solid rgba(255, 255, 255, 0.1); padding: 3rem 2rem; margin-top: 4rem; border-radius: 24px;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%); padding: 12px; border-radius: 16px; margin-right: 1rem; color: white; font-size: 1.25rem; font-weight: bold;">₿</div>
                <h3 style="color: white; font-size: 1.25rem; font-weight: 800; margin: 0;">Bitcoin Price Predictor</h3>
            </div>
            <p style="color: #bfdbfe; max-width: 32rem; margin: 0 auto 2rem auto; line-height: 1.6;">
                Advanced AI-powered Bitcoin price prediction system using ensemble machine learning models,
                developed as part of NUST's Introduction to Data Science course project.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Team section
    team_col1, team_col2, team_col3 = st.columns(3)
    
    with team_col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255, 255, 255, 0.05); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); width: 4rem; height: 4rem; border-radius: 50%; margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.125rem; color: white;">MM</div>
            <h5 style="color: white; font-weight: 500; margin: 0 0 0.5rem 0;">Muhammad Muntazar</h5>
            <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">Data Acquisition & Preprocessing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with team_col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255, 255, 255, 0.05); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="background: linear-gradient(135deg, #10b981, #06b6d4); width: 4rem; height: 4rem; border-radius: 50%; margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.125rem; color: white;">HA</div>
            <h5 style="color: white; font-weight: 500; margin: 0 0 0.5rem 0;">Hafiz Abdul Basit</h5>
            <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">Data Transformation Engineer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with team_col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255, 255, 255, 0.05); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="background: linear-gradient(135deg, #f7931a, #dc2626); width: 4rem; height: 4rem; border-radius: 50%; margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.125rem; color: white;">MW</div>
            <h5 style="color: white; font-weight: 500; margin: 0 0 0.5rem 0;">Muhammad Wasif Shakeel</h5>
            <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">AI Model Engineer</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Final footer
    st.markdown("""
    <div style="text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 2rem; margin-top: 2rem;">
        <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0 0 0.5rem 0;">
            © 2024 Bitcoin Prediction App - NUST Data Science Project
        </p>
        <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">
            Made for academic research and learning
        </p>
        <div style="margin-top: 1rem;">
            <a href="https://github.com/mwasifshkeel/bitcoin-economic-predictor" target="_blank" 
               style="display: inline-flex; align-items: center; background: linear-gradient(135deg, #374151, #1f2937); 
                      color: white; text-decoration: none; padding: 0.75rem 1.5rem; border-radius: 12px; 
                      font-weight: 500;">
                🔗 View Project on GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()