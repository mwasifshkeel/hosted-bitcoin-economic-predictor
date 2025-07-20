# Bitcoin Price Predictor - Streamlit App

An AI-powered Bitcoin price prediction application built with Streamlit and ensemble machine learning models.

## Features

- 🎯 Real-time Bitcoin price prediction
- 📊 Interactive market data input form
- 🤖 Ensemble ML models (Random Forest, XGBoost, Ridge)
- 📈 Automatic sentiment analysis from news headlines
- 📋 Model performance metrics and feature importance
- 🎨 Modern, responsive UI

## Team

- **Muhammad Muntazar** - Data Acquisition & Preprocessing
- **Hafiz Abdul Basit** - Data Transformation Engineer  
- **Muhammad Wasif Shakeel** - AI Model Engineer

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/mwasifshkeel/bitcoin-economic-predictor
cd bitcoin-economic-predictor/ProjectSite
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment

### Streamlit Cloud Deployment

1. Fork this repository to your GitHub account

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Click "New app" and connect your GitHub repository

4. Set the following:
   - Repository: `your-username/bitcoin-economic-predictor`
   - Branch: `main`
   - Main file path: `ProjectSite/streamlit_app.py`

5. Click "Deploy!"

### Heroku Deployment

1. Create a `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

### Docker Deployment

1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t bitcoin-predictor .
docker run -p 8501:8501 bitcoin-predictor
```

## Model Information

- **Algorithm**: Ensemble of Random Forest, XGBoost, and Ridge Regression
- **Features**: OHLCV data + engineered features + sentiment analysis
- **Performance**: RMSE: 97.08, MAE: 52.16, R²: 0.995
- **Confidence Scoring**: Based on model agreement and input quality

## Usage

1. Enter market data:
   - Open Price (USD)
   - High Price (USD)  
   - Low Price (USD)
   - Volume (BTC)

2. Optionally add a news headline for sentiment analysis

3. Click "Predict Bitcoin Price"

4. View prediction results with confidence scores and feature importance

## API Endpoints

The app runs entirely in Streamlit with no separate API endpoints needed.

## License

This project is for educational purposes as part of NUST's Data Science curriculum.
