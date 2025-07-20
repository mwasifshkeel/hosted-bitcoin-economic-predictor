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

def main():
    # Page configuration
    st.set_page_config(
        page_title="Bitcoin Price Predictor",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f7931a 0%, #ff6b35 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("₿ Bitcoin Price Predictor")
    st.markdown("### AI-Powered Bitcoin Price Prediction using Economic Indicators")
    
    # Sidebar for team info
    with st.sidebar:
        st.header("👥 Meet the Team")
        st.markdown("""
        **NUST Data Science Project**
        
        🔬 **Muhammad Muntazar**  
        *Data Acquisition & Preprocessing*
        
        🛠️ **Hafiz Abdul Basit**  
        *Data Transformation Engineer*
        
        🤖 **Muhammad Wasif Shakeel**  
        *AI Model Engineer*
        
        ---
        
        🎯 **Model Performance**
        - RMSE: 97.08
        - MAE: 52.16  
        - R² Score: 0.995
        
        ---
        
        🔗 [GitHub Repository](https://github.com/mwasifshkeel/bitcoin-economic-predictor)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Market Data Input")
        
        with st.form("prediction_form"):
            open_price = st.number_input(
                "Open Price (USD)", 
                min_value=0.01, 
                value=45000.0, 
                step=100.0,
                help="The opening price of Bitcoin"
            )
            
            high_price = st.number_input(
                "High Price (USD)", 
                min_value=0.01, 
                value=46000.0, 
                step=100.0,
                help="The highest price reached"
            )
            
            low_price = st.number_input(
                "Low Price (USD)", 
                min_value=0.01, 
                value=44000.0, 
                step=100.0,
                help="The lowest price reached"
            )
            
            volume = st.number_input(
                "Volume (BTC)", 
                min_value=0.0, 
                value=1500.0, 
                step=100.0,
                help="Trading volume in Bitcoin"
            )
            
            news_headline = st.text_area(
                "News Headline (Optional)",
                placeholder="Enter relevant news headline for sentiment analysis...",
                help="News headline will be analyzed for sentiment impact on price"
            )
            
            submit_button = st.form_submit_button("🚀 Predict Bitcoin Price", type="primary")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if submit_button:
            # Validation
            if high_price < low_price:
                st.error("❌ High price cannot be less than low price!")
                return
            
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or volume < 0:
                st.error("❌ Price values must be positive and volume must be non-negative!")
                return
            
            # Show loading spinner
            with st.spinner("🔄 Analyzing market data and making predictions..."):
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
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>💰 Predicted Bitcoin Price</h2>
                        <h1>${predictions['meta_model']:,.2f}</h1>
                        <p><strong>Confidence: {confidence*100:.1f}%</strong></p>
                        <p>Sentiment Score: {sentiment_score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model performance metrics
                    st.subheader("📈 Model Performance")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("RMSE", "97.08", help="Root Mean Square Error")
                    with metric_col2:
                        st.metric("MAE", "52.16", help="Mean Absolute Error")
                    with metric_col3:
                        st.metric("R² Score", "0.995", help="Coefficient of determination")
                    
                    # Individual model predictions
                    st.subheader("🔍 Individual Model Predictions")
                    pred_df = pd.DataFrame({
                        'Model': ['Random Forest', 'Ridge Regression', 'XGBoost', 'Meta Model'],
                        'Prediction': [
                            f"${predictions['random_forest']:,.2f}",
                            f"${predictions['ridge']:,.2f}",
                            f"${predictions['xgboost']:,.2f}",
                            f"${predictions['meta_model']:,.2f}"
                        ]
                    })
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Feature importance
                    st.subheader("🎯 Key Feature Importance")
                    for i, item in enumerate(feature_importance[:5]):  # Top 5 features
                        importance_pct = item['importance'] * 100
                        st.progress(item['importance'], text=f"{item['feature'].replace('_', ' ').title()}: {importance_pct:.1f}%")
                    
                    # Success message
                    st.success("✅ Prediction completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.error("Please check your input data and try again.")
        
        else:
            st.info("👆 Enter market data and click 'Predict' to get Bitcoin price prediction")
            st.markdown("""
            **How it works:**
            1. Enter current market data (OHLCV)
            2. Optionally add news headline for sentiment analysis  
            3. Our AI ensemble model processes the data
            4. Get prediction with confidence score
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎓 NUST Introduction to Data Science Course Project 2024</p>
        <p>⚡ Powered by Ensemble Machine Learning Models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
