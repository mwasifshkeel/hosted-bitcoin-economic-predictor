# Bitcoin Price Prediction Web Application

A full-stack Next.js web application that deploys machine learning models for Bitcoin price prediction using economic indicators.

## Features

- **Interactive Prediction Interface**: Enter market data and get instant price predictions
- **Real-time Results**: Display predicted prices with confidence intervals
- **Model Performance Metrics**: View RMSE: 97.08, MAE: 52.16, and R²: 0.995 scores
- **Feature Importance**: Understand which factors most influence predictions
- **Responsive Design**: Works on desktop and mobile devices
- **Full-Stack Next.js**: Backend API and frontend in a single application

## Tech Stack

- **Framework**: Next.js 14 with TypeScript
- **Frontend**: React, Tailwind CSS
- **Backend**: Next.js API Routes
- **Styling**: Tailwind CSS with custom Bitcoin theme
- **Icons**: Lucide React

## Setup Instructions

### Prerequisites

- Node.js (v18+)

### 1. Install Dependencies

```bash
npm install
```

### 2. Run the Application

```bash
# Development mode
npm run dev

# Production build
npm run build
npm start
```

### 3. Access the Application

Open [http://localhost:3000](http://localhost:3000) in your browser.

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Get price predictions

## Input Parameters

- **Open Price**: Bitcoin opening price (USD)
- **High Price**: Highest price in period (USD)  
- **Low Price**: Lowest price in period (USD)
- **Volume**: Trading volume (BTC)
- **News Headline**: Optional news text (sentiment calculated automatically)

## Output

- **Predicted Price**: Model's price prediction
- **Confidence**: Prediction confidence score
- **Calculated Sentiment**: Automatically computed sentiment score from news
- **Performance Metrics**: Model accuracy metrics (RMSE: 97.08, MAE: 52.16, R²: 0.995)
- **Feature Importance**: Most influential factors

## Architecture

### Frontend (React/Next.js)
- `app/page.tsx` - Main application page
- `components/` - Reusable UI components
- Responsive design with Tailwind CSS

### Backend (Next.js API Routes)
- `app/api/predict/route.ts` - Prediction endpoint
- `app/api/health/route.ts` - Health check endpoint
- Feature engineering and model logic

## Customization

### Styling
- Modify `tailwind.config.js` for custom colors/themes
- Update components for different layouts

### Prediction Logic
- Update feature engineering in `app/api/predict/route.ts`
- Integrate with actual ML models if available

### UI Components
- Customize forms in `components/PredictionForm.tsx`
- Modify results display in `components/PredictionResults.tsx`

## Deployment

### Vercel (Recommended)
```bash
npm run build
# Deploy to Vercel
```

### Other Platforms
- Netlify
- Railway
- AWS/GCP/Azure

## Benefits of Next.js Approach

1. **Unified Stack**: Single technology for frontend and backend
2. **Better Performance**: Optimized bundling and server-side rendering
3. **Easier Deployment**: Deploy entire app as one unit
4. **Type Safety**: Full TypeScript support across stack
5. **Built-in Optimization**: Image optimization, code splitting, etc.

## Team

- Muhammad Muntazar
- Hafiz Abdul Basit 
- Muhammad Wasif Shakeel

## License

Academic project for NUST Introduction to Data Science course.
- Muhammad Wasif Shakeel

## License

Academic project for NUST Introduction to Data Science course.
