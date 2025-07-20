'use client'

import { useState } from 'react'
import PredictionForm from '../components/PredictionForm'
import PredictionResults from '../components/PredictionResults'
import Header from '../components/Header'
import Footer from '../components/Footer'
import { Brain, Database, Zap, Activity } from 'lucide-react'

interface PredictionData {
  prediction: number
  confidence: number
  model_performance: {
    rmse: number
    mae: number
    r2: number
  }
  feature_importance?: Array<{
    feature: string
    importance: number
  }>
}

export default function Home() {
  const [predictionResult, setPredictionResult] = useState<PredictionData | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handlePrediction = (result: PredictionData) => {
    setPredictionResult(result)
  }

  const handleLoading = (loading: boolean) => {
    setIsLoading(loading)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
      <Header />
      
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-bitcoin/10 to-blue-600/10"></div>
        <div className="relative container mx-auto px-4 py-16">
          <div className="text-center mb-16">
            <div className="inline-flex items-center px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-white mb-6">
              <Activity className="w-4 h-4 mr-2" />
              <span className="text-sm text-white font-medium">Real-time Bitcoin Prediction</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-blue-100 to-bitcoin bg-clip-text text-transparent mb-6">
              Bitcoin Price Prediction
            </h1>
            <p className="text-xl text-blue-100 max-w-3xl mx-auto leading-relaxed">
              Harness the power of advanced machine learning and economic indicators to predict Bitcoin prices with automatic sentiment analysis
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
            <div className="order-2 lg:order-1">
              <PredictionForm 
                onPrediction={handlePrediction}
                onLoading={handleLoading}
              />
            </div>
            
            <div className="order-1 lg:order-2">
              <PredictionResults 
                result={predictionResult}
                isLoading={isLoading}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 bg-white/5 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">
              Powered by Advanced AI
            </h2>
            <p className="text-xl text-blue-100 max-w-2xl mx-auto">
              Our sophisticated ensemble model combines multiple algorithms for superior accuracy
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="group">
              <div className="bg-gradient-to-br from-bitcoin to-orange-500 rounded-2xl p-8 text-center transform group-hover:scale-105 transition-all duration-300 shadow-2xl">
                <div className="bg-white/20 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                  <Brain className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  Machine Learning
                </h3>
                <p className="text-orange-100 leading-relaxed">
                  Ensemble model combining Random Forest, XGBoost, and Ridge Regression for optimal predictions
                </p>
              </div>
            </div>
            
            <div className="group">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl p-8 text-center transform group-hover:scale-105 transition-all duration-300 shadow-2xl">
                <div className="bg-white/20 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                  <Database className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  Economic Data
                </h3>
                <p className="text-blue-100 leading-relaxed">
                  Trained on comprehensive economic indicators with automated news sentiment analysis
                </p>
              </div>
            </div>
            
            <div className="group">
              <div className="bg-gradient-to-br from-green-500 to-teal-600 rounded-2xl p-8 text-center transform group-hover:scale-105 transition-all duration-300 shadow-2xl">
                <div className="bg-white/20 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                  <Zap className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  Real-time
                </h3>
                <p className="text-green-100 leading-relaxed">
                  Get instant predictions with high confidence scores based on current market conditions
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </main>
  )
}