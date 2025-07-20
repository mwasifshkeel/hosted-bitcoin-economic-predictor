'use client'

import { useState } from 'react'
import { TrendingUp, DollarSign, BarChart3, MessageSquare, Sparkles, ArrowRight } from 'lucide-react'

interface PredictionFormProps {
  onPrediction: (result: any) => void
  onLoading: (loading: boolean) => void
}

export default function PredictionForm({ onPrediction, onLoading }: PredictionFormProps) {
  const [formData, setFormData] = useState({
    open_price: '',
    high_price: '',
    low_price: '',
    volume: '',
    news_headline: ''
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    onLoading(true)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          open_price: parseFloat(formData.open_price),
          high_price: parseFloat(formData.high_price),
          low_price: parseFloat(formData.low_price),
          volume: parseFloat(formData.volume),
          news_headline: formData.news_headline,
        }),
      })

      if (!response.ok) {
        throw new Error('Prediction request failed')
      }

      const result = await response.json()
      onPrediction(result)
    } catch (error) {
      console.error('Prediction error:', error)
      // Handle error appropriately
    } finally {
      onLoading(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  return (
    <div className="bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl p-8 border border-white/20">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-3xl font-bold text-white flex items-center">
          <div className="bg-gradient-to-r from-bitcoin to-orange-500 p-3 rounded-2xl mr-4">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          Market Data Input
        </h2>
        <div className="bg-white/10 rounded-full px-3 py-1">
          <Sparkles className="w-4 h-4 text-bitcoin inline mr-1" />
          <span className="text-xs text-blue-100 font-medium">AI Powered</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-blue-100 mb-3">
              <div className="flex items-center">
                <div className="bg-green-500 rounded-lg p-1 mr-2">
                  <DollarSign className="w-4 h-4 text-white" />
                </div>
                Open Price (USD)
              </div>
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.01"
                value={formData.open_price}
                onChange={(e) => handleInputChange('open_price', e.target.value)}
                className="w-full px-4 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-bitcoin focus:border-transparent transition-all duration-300"
                placeholder="45000.00"
                required
              />
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-bitcoin/20 to-transparent opacity-0 hover:opacity-100 transition-opacity pointer-events-none"></div>
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-blue-100 mb-3">
              <div className="flex items-center">
                <div className="bg-red-500 rounded-lg p-1 mr-2">
                  <TrendingUp className="w-4 h-4 text-white" />
                </div>
                High Price (USD)
              </div>
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.01"
                value={formData.high_price}
                onChange={(e) => handleInputChange('high_price', e.target.value)}
                className="w-full px-4 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-bitcoin focus:border-transparent transition-all duration-300"
                placeholder="46000.00"
                required
              />
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-bitcoin/20 to-transparent opacity-0 hover:opacity-100 transition-opacity pointer-events-none"></div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-blue-100 mb-3">
              <div className="flex items-center">
                <div className="bg-blue-500 rounded-lg p-1 mr-2">
                  <TrendingUp className="w-4 h-4 text-white rotate-180" />
                </div>
                Low Price (USD)
              </div>
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.01"
                value={formData.low_price}
                onChange={(e) => handleInputChange('low_price', e.target.value)}
                className="w-full px-4 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-bitcoin focus:border-transparent transition-all duration-300"
                placeholder="44000.00"
                required
              />
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-bitcoin/20 to-transparent opacity-0 hover:opacity-100 transition-opacity pointer-events-none"></div>
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-blue-100 mb-3">
              <div className="flex items-center">
                <div className="bg-purple-500 rounded-lg p-1 mr-2">
                  <BarChart3 className="w-4 h-4 text-white" />
                </div>
                Volume (BTC)
              </div>
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.01"
                value={formData.volume}
                onChange={(e) => handleInputChange('volume', e.target.value)}
                className="w-full px-4 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-bitcoin focus:border-transparent transition-all duration-300"
                placeholder="1500.00"
                required
              />
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-bitcoin/20 to-transparent opacity-0 hover:opacity-100 transition-opacity pointer-events-none"></div>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-semibold text-blue-100 mb-3">
            <div className="flex items-center">
              <div className="bg-indigo-500 rounded-lg p-1 mr-2">
                <MessageSquare className="w-4 h-4 text-white" />
              </div>
              News Headline (Optional)
              <span className="ml-2 text-xs text-yellow-300">📊 Sentiment calculated automatically</span>
            </div>
          </label>
          <div className="relative">
            <input
              type="text"
              value={formData.news_headline}
              onChange={(e) => handleInputChange('news_headline', e.target.value)}
              className="w-full px-4 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-bitcoin focus:border-transparent transition-all duration-300"
              placeholder="Enter relevant news headline for sentiment analysis..."
            />
          </div>
        </div>

        <button
          type="submit"
          className="w-full bg-gradient-to-r from-bitcoin to-orange-500 hover:from-bitcoin-dark hover:to-orange-600 text-white font-bold py-5 px-6 rounded-2xl transition-all duration-300 flex items-center justify-center group shadow-2xl transform hover:scale-105"
        >
          <TrendingUp className="mr-3 w-6 h-6 group-hover:animate-bounce" />
          <span className="text-lg">Predict Bitcoin Price</span>
          <ArrowRight className="ml-3 w-5 h-5 group-hover:translate-x-1 transition-transform" />
        </button>
      </form>
    </div>
  )
}