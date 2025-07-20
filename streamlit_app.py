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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Global Styles - Exact match */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 25%, #312e81 75%, #1e1b4b 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 95rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    .stToolbar {display: none;}
    
    /* Header - Exact match */
    .app-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(24px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem 0;
        margin-bottom: 0;
        position: sticky;
        top: 0;
        z-index: 50;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        max-width: 95rem;
        margin: 0 auto;
        padding: 0 2rem;
        gap: 1rem;
    }
    
    .header-icon {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);
        padding: 0.75rem;
        border-radius: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 32px rgba(247, 147, 26, 0.3);
    }
    
    .header-text h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .header-text p {
        margin: 0;
        font-size: 0.75rem;
        color: #60a5fa;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 0;
        position: relative;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 2rem;
        color: white;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        gap: 0.5rem;
    }
    
    .hero-title {
        font-size: clamp(3rem, 8vw, 5rem);
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 30%, #f7931a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin: 0 0 1.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #bfdbfe;
        max-width: 48rem;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Card Styles - Exact match */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(24px);
        border-radius: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: between;
        margin-bottom: 2rem;
    }
    
    .card-title {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .card-icon {
        padding: 0.75rem;
        border-radius: 1rem;
        color: white;
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .card-title-text {
        font-size: 1.875rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    
    .card-badge {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2rem;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #bfdbfe;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    /* Form Styles - Exact match */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 1rem !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 1rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input::placeholder, .stNumberInput input::placeholder, .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #f7931a !important;
        box-shadow: 0 0 0 3px rgba(247, 147, 26, 0.1) !important;
        background: rgba(255, 255, 255, 0.15) !important;
    }
    
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #bfdbfe !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Button Styles - Exact match */
    .stButton button {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 1.25rem 1.5rem !important;
        font-size: 1.125rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px -3px rgba(247, 147, 26, 0.4) !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 20px 40px -3px rgba(247, 147, 26, 0.5) !important;
        background: linear-gradient(135deg, #e8850e 0%, #ea580c 100%) !important;
    }
    
    .stButton button:active {
        transform: translateY(0px) scale(1) !important;
    }
    
    /* Prediction Results - Exact match */
    .prediction-main-card {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 50%, #dc2626 100%);
        border-radius: 1.5rem;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(247, 147, 26, 0.4);
    }
    
    .prediction-main-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 8rem;
        height: 8rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: 0;
    }
    
    .prediction-main-card::after {
        content: '';
        position: absolute;
        bottom: -25%;
        left: -25%;
        width: 6rem;
        height: 6rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: 0;
    }
    
    .prediction-content {
        position: relative;
        z-index: 1;
    }
    
    .prediction-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .prediction-price {
        font-size: clamp(3rem, 8vw, 4rem);
        font-weight: 900;
        margin: 1rem 0;
        line-height: 1;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 2rem;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metrics Grid - Exact match */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        text-align: center;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
    }
    
    /* Feature Importance - Exact match */
    .feature-item {
        margin-bottom: 1rem;
    }
    
    .feature-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .feature-name {
        color: #bfdbfe;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .feature-value {
        color: white;
        font-weight: 700;
        font-size: 0.875rem;
    }
    
    /* Progress bars - Exact match */
    .stProgress .st-bo {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.5rem !important;
        height: 0.5rem !important;
    }
    
    .stProgress .st-bp {
        border-radius: 0.5rem !important;
        height: 0.5rem !important;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
    }
    
    .loading-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Team Section */
    .team-card {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.1);
    }
    
    .team-avatar {
        width: 4rem;
        height: 4rem;
        border-radius: 50%;
        margin: 0 auto 1rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.125rem;
        color: white;
    }
    
    .team-name {
        color: white;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    
    .team-role {
        color: #bfdbfe;
        font-size: 0.875rem;
        margin: 0;
    }
    
    /* Footer */
    .app-footer {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(24px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1.5rem;
        padding: 3rem 2rem 2rem 2rem;
        margin-top: 4rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
        
        .hero-section {
            padding: 2rem 0;
        }
        
        .metrics-grid {
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }
        
        .prediction-price {
            font-size: 2.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render header exactly matching Next.js design"""
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <div class="header-icon">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="currentColor">
                    <path d="M16 0C7.164 0 0 7.164 0 16s7.164 16 16 16 16-7.164 16-16S24.836 0 16 0zm7.189 17.297c-.48 1.927-1.861 2.35-3.456 1.838l-.706-2.825c.896.224 1.841-.139 2.059-1.094.218-.955-.338-1.675-1.234-1.9l.337-1.35zm-2.25 9.023c-.579 2.322-2.242 2.832-4.166 2.216l-.851-3.407c1.08.27 2.218-.168 2.483-1.318.265-1.15-.408-2.018-1.487-2.288l.407-1.629z"/>
                </svg>
            </div>
            <div class="header-text">
                <h1>Bitcoin Predictor</h1>
                <p>AI-Powered Predictions</p>
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
        <div class="hero-badge">
            <span>⚡</span>
            <span>Real-time Bitcoin Prediction</span>
        </div>
        <h1 class="hero-title">Bitcoin Price Prediction</h1>
        <p class="hero-subtitle">
            Harness the power of advanced machine learning and economic indicators to predict Bitcoin prices with automatic sentiment analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content - Two column layout exactly like Next.js
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Market Data Input Card - Exact match
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">
                    <div class="card-icon" style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);">
                        📈
                    </div>
                    <h2 class="card-title-text">Market Data Input</h2>
                </div>
                <div class="card-badge">
                    <span>✨</span>
                    <span>AI Powered</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form", clear_on_submit=False):
            # Input fields with proper styling and icons
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                st.markdown("**💰 Open Price (USD)**")
                open_price = st.number_input("", min_value=0.01, value=45000.0, step=100.0, 
                                           label_visibility="collapsed", key="open_input",
                                           placeholder="45000.00")
            
            with col1_2:
                st.markdown("**📈 High Price (USD)**")
                high_price = st.number_input("", min_value=0.01, value=46000.0, step=100.0, 
                                           label_visibility="collapsed", key="high_input",
                                           placeholder="46000.00")
            
            col1_3, col1_4 = st.columns(2)
            
            with col1_3:
                st.markdown("**📉 Low Price (USD)**")
                low_price = st.number_input("", min_value=0.01, value=44000.0, step=100.0, 
                                          label_visibility="collapsed", key="low_input",
                                          placeholder="44000.00")
            
            with col1_4:
                st.markdown("**📊 Volume (BTC)**")
                volume = st.number_input("", min_value=0.0, value=1500.0, step=100.0, 
                                       label_visibility="collapsed", key="volume_input",
                                       placeholder="1500.00")
            
            st.markdown("**💬 News Headline (Optional)** *📊 Sentiment calculated automatically*")
            news_headline = st.text_area("", height=80, label_visibility="collapsed", 
                                       placeholder="Enter relevant news headline for sentiment analysis...",
                                       key="news_input")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Predict Bitcoin Price")
    
    with col2:
        # Prediction Results Card - Exact match
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">
                    <div class="card-icon" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                        🎯
                    </div>
                    <h2 class="card-title-text">Prediction Results</h2>
                </div>
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
            
            # Loading state - exact match
            with st.spinner(""):
                st.markdown("""
                <div class="loading-container">
                    <div class="loading-icon">🔄</div>
                    <h3 style="color: white; margin: 0 0 0.5rem 0;">Analyzing market data...</h3>
                    <p style="color: #bfdbfe; margin: 0;">This may take a few moments</p>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(2)  # Simulate processing time
                
                try:
                    # Calculate sentiment and make predictions
                    market_data = {
                        'open_price': open_price,
                        'high_price': high_price, 
                        'low_price': low_price,
                        'volume': volume
                    }
                    
                    sentiment_score = calculate_sentiment_score(news_headline, market_data)
                    
                    input_data = {
                        'open_price': open_price,
                        'close_price': open_price,
                        'high_price': high_price,
                        'low_price': low_price,
                        'volume': volume,
                        'sentiment_score': sentiment_score
                    }
                    
                    models = load_models()
                    features = engineer_features(input_data)
                    predictions = make_predictions(models, features)
                    confidence = calculate_confidence(predictions, features)
                    feature_importance = get_feature_importance(models)
                    
                    # Clear loading and show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.stop()
            
            # Show prediction results after processing
            if 'predictions' in locals():
                # Main prediction card - exact match
                st.markdown(f"""
                <div class="prediction-main-card">
                    <div class="prediction-content">
                        <div class="prediction-header">
                            <span>⚡</span>
                            <span>Predicted Bitcoin Price</span>
                        </div>
                        <div class="prediction-price">
                            ${predictions['meta_model']:,.2f}
                        </div>
                        <div class="confidence-badge">
                            <span>🏆</span>
                            <span>Confidence: {confidence*100:.1f}%</span>
                        </div>
                        <div style="font-size: 0.875rem; opacity: 0.9;">
                            Sentiment Score: {sentiment_score:.2f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Model Performance Metrics - exact match
                st.markdown("### 📈 Model Performance Metrics")
                st.markdown("""
                <div class="metrics-grid">
                    <div class="metric-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);">
                        <div class="metric-label" style="color: #93c5fd;">Root Mean Square Error</div>
                        <div class="metric-value">97.08</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, rgba(147, 51, 234, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);">
                        <div class="metric-label" style="color: #c4b5fd;">Mean Absolute Error</div>
                        <div class="metric-value">52.16</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(34, 197, 94, 0.2) 100%);">
                        <div class="metric-label" style="color: #86efac;">R² Score</div>
                        <div class="metric-value">0.995</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual Model Predictions
                st.markdown("### 🔍 Individual Model Predictions")
                pred_df = pd.DataFrame({
                    'Model': ['Random Forest', 'Ridge Regression', 'XGBoost', 'Meta Model'],
                    'Prediction': [
                        f"${predictions['random_forest']:,.2f}",
                        f"${predictions['ridge']:,.2f}",
                        f"${predictions['xgboost']:,.2f}",
                        f"${predictions['meta_model']:,.2f}"
                    ]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Feature Importance - exact match
                st.markdown("### 🎯 Key Feature Importance")
                
                # Create gradient colors for progress bars
                gradient_colors = [
                    "linear-gradient(90deg, #dc2626, #f97316)",  # Red to Orange
                    "linear-gradient(90deg, #f97316, #fbbf24)",  # Orange to Yellow  
                    "linear-gradient(90deg, #fbbf24, #84cc16)",  # Yellow to Lime
                    "linear-gradient(90deg, #84cc16, #22c55e)",  # Lime to Green
                    "linear-gradient(90deg, #22c55e, #06b6d4)",  # Green to Cyan
                    "linear-gradient(90deg, #06b6d4, #3b82f6)",  # Cyan to Blue
                    "linear-gradient(90deg, #3b82f6, #6366f1)",  # Blue to Indigo
                    "linear-gradient(90deg, #6366f1, #9333ea)"   # Indigo to Purple
                ]
                
                for i, item in enumerate(feature_importance[:8]):
                    importance_pct = item['importance']
                    feature_name = item['feature'].replace('_', ' ').title()
                    color = gradient_colors[min(i, len(gradient_colors)-1)]
                    
                    st.markdown(f"""
                    <div class="feature-item">
                        <div class="feature-header">
                            <span class="feature-name">{feature_name}</span>
                            <span class="feature-value">{importance_pct*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Custom progress bar with gradient
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem; height: 0.5rem; overflow: hidden; margin-bottom: 1rem;">
                        <div style="background: {color}; height: 100%; width: {importance_pct*100:.1f}%; border-radius: 0.5rem; transition: width 1s ease-out;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("✅ Prediction completed successfully!")
        
        else:
            # Default empty state - exact match
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">🎯</div>
                <h3 style="color: white; margin-bottom: 1rem; font-size: 1.25rem; font-weight: 700;">Ready to predict Bitcoin prices</h3>
                <p style="color: #bfdbfe; margin: 0;">Enter market data to get started</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Section - exact match
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px); padding: 5rem 2rem; margin: 4rem 0; border-radius: 1.5rem; border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="text-align: center; margin-bottom: 4rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Powered by Advanced AI
            </h2>
            <p style="font-size: 1.25rem; color: #bfdbfe; max-width: 32rem; margin: 0 auto; line-height: 1.6;">
                Our sophisticated ensemble model combines multiple algorithms for superior accuracy
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature cards - exact match  
    feat_col1, feat_col2, feat_col3 = st.columns(3, gap="large")
    
    with feat_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%); border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); transition: transform 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 5rem; height: 5rem; margin: 0 auto 1.5rem auto; display: flex; align-items: center; justify-content: center; font-size: 2.5rem;">
                🧠
            </div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Machine Learning
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6; margin: 0;">
                Ensemble model combining Random Forest, XGBoost, and Ridge Regression for optimal predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); transition: transform 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 5rem; height: 5rem; margin: 0 auto 1.5rem auto; display: flex; align-items: center; justify-content: center; font-size: 2.5rem;">
                💾
            </div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Economic Data
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6; margin: 0;">
                Trained on comprehensive economic indicators with automated news sentiment analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%); border-radius: 1rem; padding: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); transition: transform 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 5rem; height: 5rem; margin: 0 auto 1.5rem auto; display: flex; align-items: center; justify-content: center; font-size: 2.5rem;">
                ⚡
            </div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Real-time
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6; margin: 0;">
                Get instant predictions with high confidence scores based on current market conditions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close features section
    
    # Footer - exact match
    st.markdown("""
    <div class="app-footer">
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%); padding: 0.75rem; border-radius: 1rem; margin-right: 1rem; color: white; font-size: 1.5rem; font-weight: bold;">₿</div>
                <h3 style="color: white; font-size: 1.25rem; font-weight: 800; margin: 0;">Bitcoin Price Predictor</h3>
            </div>
            <p style="color: #bfdbfe; max-width: 48rem; margin: 0 auto 2rem auto; line-height: 1.6;">
                Advanced AI-powered Bitcoin price prediction system using ensemble machine learning models,
                developed as part of NUST's Introduction to Data Science course project.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Team section - exact match
    team_col1, team_col2, team_col3 = st.columns(3, gap="large")
    
    with team_col1:
        st.markdown("""
        <div class="team-card">
            <div class="team-avatar" style="background: linear-gradient(135deg, #3b82f6, #8b5cf6);">MM</div>
            <h5 class="team-name">Muhammad Muntazar</h5>
            <p class="team-role">Data Acquisition & Preprocessing Specialist</p>
        </div>
        """, unsafe_allow_html=True)
    
    with team_col2:
        st.markdown("""
        <div class="team-card">
            <div class="team-avatar" style="background: linear-gradient(135deg, #10b981, #06b6d4);">HA</div>
            <h5 class="team-name">Hafiz Abdul Basit</h5>
            <p class="team-role">Data Transformation Engineer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with team_col3:
        st.markdown("""
        <div class="team-card">
            <div class="team-avatar" style="background: linear-gradient(135deg, #f7931a, #dc2626);">MW</div>
            <h5 class="team-name">Muhammad Wasif Shakeel</h5>
            <p class="team-role">AI Model Engineer</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Final footer section
    st.markdown("""
        <div style="text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 2rem; margin-top: 2rem;">
            <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0 0 0.5rem 0;">
                © 2024 Bitcoin Prediction App - NUST Data Science Project
            </p>
            <p style="color: #64748b; font-size: 0.875rem; margin: 0 0 1rem 0;">
                Made for academic research and learning
            </p>
            <a href="https://github.com/mwasifshkeel/bitcoin-economic-predictor" target="_blank" 
               style="display: inline-flex; align-items: center; background: linear-gradient(135deg, #374151, #1f2937); 
                      color: white; text-decoration: none; padding: 0.75rem 1.5rem; border-radius: 0.75rem; 
                      font-weight: 600; gap: 0.5rem; transition: all 0.3s ease; border: 1px solid rgba(255, 255, 255, 0.1);" 
               onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.3)'" 
               onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='none'">
                🔗 View Project on GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()