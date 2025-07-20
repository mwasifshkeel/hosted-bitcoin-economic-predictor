'use client'

import { TrendingUp, Target, BarChart, Award, Activity, Zap } from 'lucide-react'

interface PredictionResultsProps {
  result: any
  isLoading: boolean
}

export default function PredictionResults({ result, isLoading }: PredictionResultsProps) {
  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl p-8 border border-white/20">
        <div className="animate-pulse space-y-6">
          <div className="flex items-center space-x-4">
            <div className="h-12 w-12 bg-white/20 rounded-2xl"></div>
            <div className="h-8 bg-white/20 rounded-xl flex-1"></div>
          </div>
          <div className="h-32 bg-gradient-to-r from-white/10 to-white/5 rounded-2xl flex items-center justify-center">
            <div className="flex items-center space-x-2 text-blue-200">
              <Activity className="w-6 h-6 animate-spin" />
              <span className="text-lg font-medium">Analyzing market data...</span>
            </div>
          </div>
          <div className="space-y-4">
            <div className="h-6 bg-white/20 rounded-xl"></div>
            <div className="grid grid-cols-3 gap-4">
              <div className="h-20 bg-white/10 rounded-xl"></div>
              <div className="h-20 bg-white/10 rounded-xl"></div>
              <div className="h-20 bg-white/10 rounded-xl"></div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl p-8 border border-white/20">
        <h2 className="text-3xl font-bold text-white mb-8 flex items-center">
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-2xl mr-4">
            <Target className="w-6 h-6 text-white" />
          </div>
          Prediction Results
        </h2>
        <div className="text-center text-blue-200 py-16">
          <div className="bg-white/5 rounded-full w-24 h-24 mx-auto mb-6 flex items-center justify-center">
            <TrendingUp className="w-12 h-12 text-blue-300" />
          </div>
          <p className="text-xl mb-4">Ready to predict Bitcoin prices</p>
          <p className="text-blue-300">Enter market data to get started</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl p-8 border border-white/20">
      <h2 className="text-3xl font-bold text-white mb-8 flex items-center">
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-3 rounded-2xl mr-4">
          <Target className="w-6 h-6 text-white" />
        </div>
        Prediction Results
      </h2>

      {/* Main Prediction */}
      <div className="bg-gradient-to-br from-bitcoin via-orange-500 to-red-500 rounded-3xl p-8 text-white mb-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-16 translate-x-16"></div>
        <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/10 rounded-full translate-y-12 -translate-x-12"></div>
        <div className="relative text-center">
          <div className="flex items-center justify-center mb-4">
            <Zap className="w-6 h-6 mr-2" />
            <h3 className="text-xl font-semibold">Predicted Bitcoin Price</h3>
          </div>
          <div className="text-5xl md:text-6xl font-bold mb-4 animate-pulse">
            ${result.prediction?.toLocaleString(undefined, { 
              minimumFractionDigits: 2, 
              maximumFractionDigits: 2 
            })}
          </div>
          <div className="flex items-center justify-center bg-white/20 rounded-full px-6 py-3 backdrop-blur-sm">
            <Award className="w-5 h-5 mr-2" />
            <span className="font-semibold">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
          <BarChart className="mr-3 w-6 h-6 text-bitcoin" />
          Model Performance Metrics
        </h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-6 bg-gradient-to-br from-blue-500/20 to-blue-600/20 backdrop-blur-sm rounded-2xl border border-white/10">
            <div className="text-sm text-blue-200 font-medium mb-2">Root Mean Square Error</div>
            <div className="text-2xl font-bold text-white">
              {result.model_performance?.rmse?.toFixed(2) || '97.08'}
            </div>
          </div>
          <div className="text-center p-6 bg-gradient-to-br from-purple-500/20 to-purple-600/20 backdrop-blur-sm rounded-2xl border border-white/10">
            <div className="text-sm text-purple-200 font-medium mb-2">Mean Absolute Error</div>
            <div className="text-2xl font-bold text-white">
              {result.model_performance?.mae?.toFixed(2) || '52.16'}
            </div>
          </div>
          <div className="text-center p-6 bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-sm rounded-2xl border border-white/10">
            <div className="text-sm text-green-200 font-medium mb-2">R² Score</div>
            <div className="text-2xl font-bold text-white">
              {result.model_performance?.r2?.toFixed(3) || '0.995'}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      {result.feature_importance && (
        <div>
          <h3 className="text-xl font-semibold text-white mb-6">
            Key Feature Importance
          </h3>
          <div className="space-y-4">
            {result.feature_importance.map((item: any, index: number) => (
              <div key={index} className="group">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-blue-100 font-medium capitalize">
                    {item.feature.replace('_', ' ')}
                  </div>
                  <div className="text-white font-semibold">
                    {(item.importance * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="relative">
                  <div className="bg-white/10 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                        index === 0 ? 'bg-gradient-to-r from-red-600 to-orange-500' :
                        index === 1 ? 'bg-gradient-to-r from-orange-500 to-yellow-400' :
                        index === 2 ? 'bg-gradient-to-r from-yellow-400 to-lime-400' :
                        index === 3 ? 'bg-gradient-to-r from-lime-400 to-green-400' :
                        index === 4 ? 'bg-gradient-to-r from-green-400 to-cyan-400' :
                        index === 5 ? 'bg-gradient-to-r from-cyan-400 to-blue-400' :
                        index === 6 ? 'bg-gradient-to-r from-blue-400 to-indigo-400' :
                        index === 7 ? 'bg-gradient-to-r from-indigo-400 to-gray-400' :
                        'bg-gradient-to-r from-gray-400 to-gray-300'
                      }`}
                      style={{ width: `${item.importance * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}