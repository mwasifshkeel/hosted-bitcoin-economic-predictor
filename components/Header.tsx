import { Bitcoin, Menu } from 'lucide-react'

export default function Header() {
  return (
    <header className="bg-white/10 backdrop-blur-xl border-b border-white/20 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-bitcoin to-orange-500 p-3 rounded-2xl">
              <Bitcoin className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
                Bitcoin Predictor
              </h1>
              <p className="text-xs text-blue-200 font-medium">
                AI-Powered Predictions
              </p>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}