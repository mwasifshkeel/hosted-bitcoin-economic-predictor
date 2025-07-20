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
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 25%, #312e81 75%, #1e1b4b 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 90rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    .stToolbar {display: none;}
    
    /* Remove default margins and padding */
    .block-container > div {
        padding-top: 0rem;
    }
    
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        max-width: 90rem;
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
        width: 3rem;
        height: 3rem;
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
        padding: 3rem 2rem 4rem 2rem;
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
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 30%, #f7931a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin: 0 0 1.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.125rem;
        color: #bfdbfe;
        max-width: 48rem;
        margin: 0 auto 2rem auto;
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
        margin-bottom: 2rem;
        height: fit-content;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .card-title {
        display: flex;
        align-items: center;
        gap: 1rem;
        flex: 1;
    }
    
    .card-icon {
        padding: 0.75rem;
        border-radius: 1rem;
        color: white;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        min-width: 3rem;
        height: 3rem;
    }
    
    .card-title-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    
    .card-badge {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2rem;
        padding: 0.5rem 1rem;
        font-size: 0.75rem;
        color: #bfdbfe;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        white-space: nowrap;
    }
    
    /* Input Labels with Icons */
    .input-label {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        color: #bfdbfe;
        font-weight: 600;
        font-size: 0.875rem;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .label-icon {
        padding: 0.25rem;
        border-radius: 0.5rem;
        color: white;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 1.5rem;
        height: 1.5rem;
    }
    
    /* Form Styles - Fixed */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 1rem !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        min-height: 3rem !important;
        box-sizing: border-box !important;
    }
    
    .stTextArea > div > div > textarea {
        min-height: 5rem !important;
        resize: vertical !important;
    }
    
    .stTextInput > div > div > input::placeholder, 
    .stNumberInput > div > div > input::placeholder, 
    .stTextArea > div > div > textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 0.8 !important;
    }
    
    .stTextInput > div > div > input:focus, 
    .stNumberInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: #f7931a !important;
        box-shadow: 0 0 0 3px rgba(247, 147, 26, 0.2) !important;
        background: rgba(255, 255, 255, 0.15) !important;
        outline: none !important;
    }
    
    /* Button Styles - Fixed */
    .stButton > button {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.125rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px -3px rgba(247, 147, 26, 0.4) !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        position: relative !important;
        overflow: hidden !important;
        min-height: 3.5rem !important;
        margin-top: 1rem !important;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 40px -3px rgba(247, 147, 26, 0.6) !important;
        background: linear-gradient(135deg, #e8850e 0%, #ea580c 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Form layout fixes */
    .stForm {
        border: none !important;
        background: transparent !important;
    }
    
    /* Column spacing fixes */
    .stColumn {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    .stColumn:first-child {
        padding-left: 0 !important;
    }
    
    .stColumn:last-child {
        padding-right: 0 !important;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #bfdbfe;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        color: #60a5fa;
    }
    
    .empty-state h3 {
        color: white;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .empty-state p {
        color: #94a3b8;
        margin: 0;
        font-size: 1rem;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 2rem;
        text-align: center;
    }
    
    .loading-spinner {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Prediction Results - Exact match */
    .prediction-main-card {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 50%, #dc2626 100%);
        border-radius: 1.5rem;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(247, 147, 26, 0.5);
    }
    
    .prediction-main-card::before {
        content: '';
        position: absolute;
        top: -3rem;
        right: -3rem;
        width: 6rem;
        height: 6rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: 0;
    }
    
    .prediction-main-card::after {
        content: '';
        position: absolute;
        bottom: -2rem;
        left: -2rem;
        width: 4rem;
        height: 4rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: 0;
    }
    
    .prediction-content {
        position: relative;
        z-index: 1;
    }
    
    .prediction-price {
        font-size: clamp(2.5rem, 6vw, 3.5rem);
        font-weight: 900;
        margin: 1rem 0;
        line-height: 1;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Team Cards - Fixed */
    .team-card {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .team-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    
    .team-avatar {
        width: 4rem;
        height: 4rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.125rem;
        margin: 0 auto 1rem auto;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .team-name {
        color: white;
        font-weight: 600;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
    }
    
    .team-role {
        color: #bfdbfe;
        font-size: 0.875rem;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Features Section - Fixed spacing */
    .features-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        padding: 4rem 2rem;
        margin: 3rem 0;
        border-radius: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-card {
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .feature-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
    }
    
    .feature-icon {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        width: 4rem;
        height: 4rem;
        margin: 0 auto 1.5rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Footer improvements */
    .app-footer {
        padding: 3rem 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
            padding: 2rem 1rem 3rem 1rem;
        }
        
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .card-header {
            flex-direction: column;
            text-align: center;
            gap: 0.5rem;
        }
        
        .card-title {
            justify-content: center;
        }
        
        .features-section {
            padding: 3rem 1rem;
        }
        
        .prediction-main-card {
            padding: 1.5rem;
        }
        
        .stColumn {
            margin-bottom: 1rem;
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
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    
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
        # Market Data Input Card - Fixed styling
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
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form", clear_on_submit=False):
            # Input fields with proper styling and icons
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                st.markdown("""
                <div class="input-label">
                    <div class="label-icon" style="background: #10b981;">💰</div>
                    <span>Open Price (USD)</span>
                </div>
                """, unsafe_allow_html=True)
                open_price = st.number_input("", min_value=0.01, value=45000.0, step=100.0, 
                                           label_visibility="collapsed", key="open_input",
                                           help="The opening price of Bitcoin for the trading period")
            
            with col1_2:
                st.markdown("""
                <div class="input-label">
                    <div class="label-icon" style="background: #ef4444;">📈</div>
                    <span>High Price (USD)</span>
                </div>
                """, unsafe_allow_html=True)
                high_price = st.number_input("", min_value=0.01, value=46000.0, step=100.0, 
                                           label_visibility="collapsed", key="high_input",
                                           help="The highest price reached during the trading period")
            
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
            
            col1_3, col1_4 = st.columns(2)
            
            with col1_3:
                st.markdown("""
                <div class="input-label">
                    <div class="label-icon" style="background: #3b82f6;">📉</div>
                    <span>Low Price (USD)</span>
                </div>
                """, unsafe_allow_html=True)
                low_price = st.number_input("", min_value=0.01, value=44000.0, step=100.0, 
                                          label_visibility="collapsed", key="low_input",
                                          help="The lowest price reached during the trading period")
            
            with col1_4:
                st.markdown("""
                <div class="input-label">
                    <div class="label-icon" style="background: #8b5cf6;">📊</div>
                    <span>Volume (BTC)</span>
                </div>
                """, unsafe_allow_html=True)
                volume = st.number_input("", min_value=0.0, value=1500.0, step=100.0, 
                                       label_visibility="collapsed", key="volume_input",
                                       help="Trading volume in Bitcoin during the period")
            
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="input-label">
                <div class="label-icon" style="background: #6366f1;">💬</div>
                <span>News Headline (Optional)</span>
                <span style="color: #fbbf24; margin-left: 0.5rem; font-size: 0.75rem;">📊 Sentiment calculated automatically</span>
            </div>
            """, unsafe_allow_html=True)
            news_headline = st.text_area("", height=100, label_visibility="collapsed", 
                                       placeholder="Enter relevant news headline for sentiment analysis...",
                                       key="news_input",
                                       help="Enter a news headline to incorporate sentiment analysis into the prediction")
            
            submitted = st.form_submit_button("🚀 Predict Bitcoin Price")
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close glass-card
    
    with col2:
        # Prediction Results Card - Fixed styling
        st.markdown("""
        <div class="glass-card" style="min-height: 400px;">
            <div class="card-header">
                <div class="card-title">
                    <div class="card-icon" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                        🎯
                    </div>
                    <h2 class="card-title-text">Prediction Results</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if submitted:
            # Validation
            if high_price < low_price:
                st.error("❌ High price cannot be less than low price!")
            elif open_price <= 0 or high_price <= 0 or low_price <= 0 or volume < 0:
                st.error("❌ Price values must be positive and volume must be non-negative!")
            else:
                # Set loading state
                st.session_state.is_loading = True
                st.rerun()
        
        # Handle different states
        if st.session_state.is_loading:
            # Loading state - exact match
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner">⚙️</div>
                <h3 style="color: #60a5fa; margin: 0 0 0.5rem 0; font-weight: 600;">Analyzing market data...</h3>
                <p style="color: #bfdbfe; margin: 0;">This may take a few moments</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate processing
            time.sleep(3)
            
            try:
                # Get current form values from session state
                open_price = st.session_state.open_input
                high_price = st.session_state.high_input 
                low_price = st.session_state.low_input
                volume = st.session_state.volume_input
                news_headline = st.session_state.news_input or ""
                
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
                
                # Store results in session state
                st.session_state.prediction_result = {
                    'prediction': predictions['meta_model'],
                    'confidence': confidence,
                    'sentiment_score': sentiment_score,
                    'predictions': predictions,
                    'feature_importance': feature_importance
                }
                st.session_state.is_loading = False
                st.rerun()
                
            except Exception as e:
                st.session_state.is_loading = False
                st.error(f"❌ Error making prediction: {str(e)}")
                st.rerun()
        
        elif st.session_state.prediction_result:
            # Show prediction results
            result = st.session_state.prediction_result
            
            # Main prediction card - exact match
            st.markdown(f"""
            <div class="prediction-main-card">
                <div class="prediction-content">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem; font-size: 1.25rem; font-weight: 600;">
                        <span>⚡</span>
                        <span>Predicted Bitcoin Price</span>
                    </div>
                    <div class="prediction-price">
                        ${result['prediction']:,.2f}
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); border-radius: 2rem; padding: 0.75rem 1.5rem; font-weight: 700; display: inline-flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                        <span>🏆</span>
                        <span>Confidence: {result['confidence']*100:.1f}%</span>
                    </div>
                    <div style="font-size: 0.875rem; opacity: 0.9;">
                        Sentiment Score: {result['sentiment_score']:.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Model Performance Metrics
            st.markdown("### 📈 Model Performance")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("RMSE", "97.08", help="Root Mean Square Error")
            with metric_col2:
                st.metric("MAE", "52.16", help="Mean Absolute Error") 
            with metric_col3:
                st.metric("R² Score", "0.995", help="Coefficient of determination")
            
            # Individual Model Predictions
            st.markdown("### 🔍 Individual Model Predictions")
            pred_df = pd.DataFrame({
                'Model': ['Random Forest', 'Ridge Regression', 'XGBoost', 'Meta Model'],
                'Prediction': [
                    f"${result['predictions']['random_forest']:,.2f}",
                    f"${result['predictions']['ridge']:,.2f}",
                    f"${result['predictions']['xgboost']:,.2f}",
                    f"${result['predictions']['meta_model']:,.2f}"
                ]
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # Feature Importance
            st.markdown("### 🎯 Key Feature Importance")
            for i, item in enumerate(result['feature_importance'][:5]):
                importance_pct = item['importance']
                feature_name = item['feature'].replace('_', ' ').title()
                st.progress(importance_pct, text=f"{feature_name}: {importance_pct*100:.1f}%")
            
            st.success("✅ Prediction completed successfully!")
            
            # Reset button
            if st.button("🔄 Make Another Prediction"):
                st.session_state.prediction_result = None
                st.session_state.is_loading = False
                st.rerun()
        
        else:
            # Empty state - exact match from images
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🎯</div>
                <h3>Ready to predict Bitcoin prices</h3>
                <p>Enter market data to get started</p>
            </div>
            """, unsafe_allow_html=True)

    # Features Section - Fixed layout
    st.markdown("""
    <div class="features-section">
        <div style="text-align: center; margin-bottom: 3rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 1rem;">
                Powered by Advanced AI
            </h2>
            <p style="font-size: 1.25rem; color: #bfdbfe; max-width: 32rem; margin: 0 auto; line-height: 1.6;">
                Our sophisticated ensemble model combines multiple algorithms for superior accuracy
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards - Fixed layout
    feat_col1, feat_col2, feat_col3 = st.columns(3, gap="large")
    
    with feat_col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);">
            <div>
                <div class="feature-icon">🧠</div>
                <h3 class="feature-title">Machine Learning</h3>
            </div>
            <p class="feature-description">
                Ensemble model combining Random Forest, XGBoost, and Ridge Regression for optimal predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);">
            <div>
                <div class="feature-icon">💾</div>
                <h3 class="feature-title">Economic Data</h3>
            </div>
            <p class="feature-description">
                Trained on comprehensive economic indicators with automated news sentiment analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);">
            <div>
                <div class="feature-icon">⚡</div>
                <h3 class="feature-title">Real-time</h3>
            </div>
            <p class="feature-description">
                Get instant predictions with high confidence scores based on current market conditions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer section - Fixed layout
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
    
    # Team section - Fixed cards
    st.markdown("<h4 style='text-align: center; color: white; font-size: 1.125rem; font-weight: 700; margin-bottom: 2rem;'>Meet the Team</h4>", unsafe_allow_html=True)
    
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
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()