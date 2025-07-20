import streamlit as st
import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

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
    """Load custom CSS to match Next.js design"""
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #1e40af 50%, #312e81 100%);
        color: white;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 90rem;
    }
    
    /* Header Styles */
    .header-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0 2rem;
    }
    
    .bitcoin-icon {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);
        padding: 12px;
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.75rem;
        color: #bfdbfe;
        font-weight: 500;
        margin: 0;
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
        backdrop-filter: blur(8px);
        border-radius: 50px;
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 50%, #f7931a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        line-height: 1.1;
    }
    
    .hero-description {
        font-size: 1.25rem;
        color: #bfdbfe;
        max-width: 48rem;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
    }
    
    /* Form Container */
    .form-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .form-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .form-title {
        display: flex;
        align-items: center;
        font-size: 1.875rem;
        font-weight: 800;
        color: white;
    }
    
    .form-icon {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);
        padding: 12px;
        border-radius: 16px;
        margin-right: 1rem;
    }
    
    .ai-badge {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #bfdbfe;
        font-weight: 500;
    }
    
    /* Input Styles */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        color: white !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #f7931a !important;
        box-shadow: 0 0 0 2px rgba(247, 147, 26, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #bfdbfe !important;
    }
    
    /* Label Styles */
    .stTextInput > label,
    .stNumberInput > label,
    .stTextArea > label {
        color: #bfdbfe !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        font-size: 1.125rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 40px -3px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Results Container */
    .results-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
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
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 8rem;
        height: 8rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        transform: translate(4rem, -4rem);
    }
    
    .prediction-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 6rem;
        height: 6rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        transform: translate(-3rem, 3rem);
    }
    
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 100%);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #bfdbfe;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
    }
    
    /* Feature Importance */
    .feature-item {
        margin-bottom: 1rem;
    }
    
    .feature-header {
        display: flex;
        justify-content: between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .feature-name {
        color: #bfdbfe;
        font-weight: 500;
        text-transform: capitalize;
    }
    
    .feature-value {
        color: white;
        font-weight: 600;
    }
    
    .feature-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        height: 12px;
        overflow: hidden;
    }
    
    .feature-progress {
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease-out;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        animation: spin 1s linear infinite;
        display: inline-block;
    }
    
    /* Features Section */
    .features-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(8px);
        padding: 5rem 0;
        margin: 4rem 0;
        border-radius: 24px;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 4rem;
    }
    
    .feature-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
    }
    
    .feature-icon {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        width: 5rem;
        height: 5rem;
        margin: 0 auto 1.5rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Footer */
    .footer-section {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem 0;
        margin-top: 4rem;
        border-radius: 24px 24px 0 0;
    }
    
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .team-member {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .prediction-price {
            font-size: 2.5rem;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
        }
        
        .team-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the header section"""
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <div class="bitcoin-icon">
                ₿
            </div>
            <div>
                <h1 class="header-title">Bitcoin Predictor</h1>
                <p class="header-subtitle">AI-Powered Predictions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_hero():
    """Render the hero section"""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">
            ⚡ Real-time Bitcoin Prediction
        </div>
        <h1 class="hero-title">Bitcoin Price Prediction</h1>
        <p class="hero-description">
            Harness the power of advanced machine learning and economic indicators to predict Bitcoin prices with automatic sentiment analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_form():
    """Render the prediction form matching Next.js design"""
    with st.container():
        st.markdown("""
        <div class="form-header">
            <div class="form-title">
                <div class="form-icon">📈</div>
                Market Data Input
            </div>
            <div class="ai-badge">
                ✨ AI Powered
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # First row - Open Price and High Price
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**💰 Open Price (USD)**")
                open_price = st.number_input("", min_value=0.01, value=45000.0, step=100.0, key="open", label_visibility="collapsed")
            
            with col2:
                st.markdown("**📈 High Price (USD)**")
                high_price = st.number_input("", min_value=0.01, value=46000.0, step=100.0, key="high", label_visibility="collapsed")
            
            # Second row - Low Price and Volume
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**📉 Low Price (USD)**")
                low_price = st.number_input("", min_value=0.01, value=44000.0, step=100.0, key="low", label_visibility="collapsed")
            
            with col4:
                st.markdown("**📊 Volume (BTC)**")
                volume = st.number_input("", min_value=0.0, value=1500.0, step=100.0, key="volume", label_visibility="collapsed")
            
            # News headline
            st.markdown("**💬 News Headline (Optional)** *📊 Sentiment calculated automatically*")
            news_headline = st.text_input("", placeholder="Enter relevant news headline for sentiment analysis...", key="news", label_visibility="collapsed")
            
            # Submit button
            submit_button = st.form_submit_button("📈 Predict Bitcoin Price")
    
    return submit_button, open_price, high_price, low_price, volume, news_headline

def render_loading():
    """Render loading state"""
    st.markdown("""
    <div class="results-container">
        <div style="text-align: center; padding: 2rem 0;">
            <div class="loading-spinner" style="font-size: 2rem; margin-bottom: 1rem;">⚡</div>
            <h3 style="color: #bfdbfe; margin-bottom: 1rem;">Analyzing market data...</h3>
            <div style="background: rgba(255, 255, 255, 0.1); height: 4px; border-radius: 2px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #f7931a, #ff6b35); height: 100%; width: 70%; animation: pulse 2s infinite; border-radius: 2px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_results(prediction_data):
    """Render prediction results matching Next.js design"""
    if prediction_data is None:
        st.markdown("""
        <div class="results-container">
            <div style="text-align: center; padding: 4rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">🎯</div>
                <h3 style="color: white; margin-bottom: 1rem;">Ready to predict Bitcoin prices</h3>
                <p style="color: #bfdbfe;">Enter market data to get started</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <span style="margin-right: 0.5rem;">⚡</span>
                <h3>Predicted Bitcoin Price</h3>
            </div>
            <div class="prediction-price">
                ${prediction_data['prediction']:,.2f}
            </div>
            <div class="confidence-badge">
                <span>🏆</span>
                <span>Confidence: {prediction_data['confidence']*100:.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("""
    <h3 style="color: white; margin: 2rem 0 1rem 0; display: flex; align-items: center;">
        📊 Model Performance Metrics
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 100%);">
            <div class="metric-label">Root Mean Square Error</div>
            <div class="metric-value">{prediction_data['model_performance']['rmse']:.2f}</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(147, 51, 234, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value">{prediction_data['model_performance']['mae']:.2f}</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{prediction_data['model_performance']['r2']:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance
    if 'feature_importance' in prediction_data:
        st.markdown("""
        <h3 style="color: white; margin: 2rem 0 1rem 0;">🎯 Key Feature Importance</h3>
        """, unsafe_allow_html=True)
        
        colors = [
            'linear-gradient(90deg, #dc2626, #f7931a)',
            'linear-gradient(90deg, #f7931a, #fbbf24)',
            'linear-gradient(90deg, #fbbf24, #84cc16)',
            'linear-gradient(90deg, #84cc16, #10b981)',
            'linear-gradient(90deg, #10b981, #06b6d4)',
            'linear-gradient(90deg, #06b6d4, #3b82f6)',
            'linear-gradient(90deg, #3b82f6, #6366f1)',
            'linear-gradient(90deg, #6366f1, #8b5cf6)',
        ]
        
        for i, item in enumerate(prediction_data['feature_importance'][:8]):
            color = colors[i] if i < len(colors) else colors[-1]
            st.markdown(f"""
            <div class="feature-item">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="feature-name">{item['feature'].replace('_', ' ').title()}</span>
                    <span class="feature-value">{item['importance']*100:.1f}%</span>
                </div>
                <div class="feature-bar">
                    <div class="feature-progress" style="width: {item['importance']*100}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_features():
    """Render features section"""
    st.markdown("""
    <div class="features-section">
        <div style="text-align: center; margin-bottom: 4rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Powered by Advanced AI
            </h2>
            <p style="font-size: 1.25rem; color: #bfdbfe; max-width: 32rem; margin: 0 auto;">
                Our sophisticated ensemble model combines multiple algorithms for superior accuracy
            </p>
        </div>
        
        <div class="features-grid">
            <div class="feature-card" style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);">
                <div class="feature-icon">
                    🧠
                </div>
                <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                    Machine Learning
                </h3>
                <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                    Ensemble model combining Random Forest, XGBoost, and Ridge Regression for optimal predictions
                </p>
            </div>
            
            <div class="feature-card" style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);">
                <div class="feature-icon">
                    💾
                </div>
                <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                    Economic Data
                </h3>
                <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                    Trained on comprehensive economic indicators with automated news sentiment analysis
                </p>
            </div>
            
            <div class="feature-card" style="background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);">
                <div class="feature-icon">
                    ⚡
                </div>
                <h3 style="font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                    Real-time
                </h3>
                <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                    Get instant predictions with high confidence scores based on current market conditions
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render footer section"""
    st.markdown("""
    <div class="footer-section">
        <div style="text-center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <div class="bitcoin-icon" style="margin-right: 1rem;">₿</div>
                <h3 style="color: white; font-size: 1.25rem; font-weight: 800; margin: 0;">Bitcoin Price Predictor</h3>
            </div>
            <p style="color: #bfdbfe; max-width: 32rem; margin: 0 auto 2rem auto; line-height: 1.6;">
                Advanced AI-powered Bitcoin price prediction system using ensemble machine learning models,
                developed as part of NUST's Introduction to Data Science course project.
            </p>
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 2rem; margin-bottom: 2rem;">
            <h4 style="color: white; text-align: center; margin-bottom: 2rem;">Meet the Team</h4>
            <div class="team-grid">
                <div class="team-member">
                    <div class="team-avatar" style="background: linear-gradient(135deg, #3b82f6, #8b5cf6);">MM</div>
                    <h5 style="color: white; font-weight: 500; margin: 0 0 0.5rem 0;">Muhammad Muntazar</h5>
                    <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">Data Acquisition & Preprocessing Specialist</p>
                </div>
                <div class="team-member">
                    <div class="team-avatar" style="background: linear-gradient(135deg, #10b981, #06b6d4);">HA</div>
                    <h5 style="color: white; font-weight: 500; margin: 0 0 0.5rem 0;">Hafiz Abdul Basit</h5>
                    <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">Data Transformation Engineer</p>
                </div>
                <div class="team-member">
                    <div class="team-avatar" style="background: linear-gradient(135deg, #f7931a, #dc2626);">MW</div>
                    <h5 style="color: white; font-weight: 500; margin: 0 0 0.5rem 0;">Muhammad Wasif Shakeel</h5>
                    <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">AI Model Engineer</p>
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin-bottom: 2rem;">
            <a href="https://github.com/mwasifshkeel/bitcoin-economic-predictor" target="_blank" 
               style="display: inline-flex; align-items: center; background: linear-gradient(135deg, #374151, #1f2937); 
                      color: white; text-decoration: none; padding: 0.75rem 1.5rem; border-radius: 12px; 
                      transition: all 0.3s ease; font-weight: 500;">
                🔗 View Project on GitHub
            </a>
        </div>
        
        <div style="border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 2rem; text-align: center;">
            <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0 0 0.5rem 0;">
                © 2024 Bitcoin Prediction App - NUST Data Science Project
            </p>
            <p style="color: #bfdbfe; font-size: 0.875rem; margin: 0;">
                Made for academic research and learning
            </p>
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
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    
    # Render header
    render_header()
    
    # Render hero section
    render_hero()
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        submit_button, open_price, high_price, low_price, volume, news_headline = render_prediction_form()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit_button:
            # Validation
            if high_price < low_price:
                st.error("❌ High price cannot be less than low price!")
                return
            
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or volume < 0:
                st.error("❌ Price values must be positive and volume must be non-negative!")
                return
            
            # Set loading state
            st.session_state.is_loading = True
            st.rerun()
    
    with col2:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        if st.session_state.is_loading:
            render_loading()
            
            # Perform prediction
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
                    'close_price': open_price,  # Use open price as close price default
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
                
                # Store result
                st.session_state.prediction_result = {
                    'prediction': predictions['meta_model'],
                    'confidence': confidence,
                    'model_performance': {
                        'rmse': 97.08,
                        'mae': 52.16,
                        'r2': 0.995
                    },
                    'feature_importance': feature_importance,
                    'individual_predictions': predictions,
                    'calculated_sentiment': sentiment_score
                }
                
                # Clear loading state
                st.session_state.is_loading = False
                st.rerun()
                
            except Exception as e:
                st.session_state.is_loading = False
                st.error(f"❌ Error making prediction: {str(e)}")
                st.rerun()
        else:
            render_prediction_results(st.session_state.prediction_result)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Render features section
    render_features()
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
