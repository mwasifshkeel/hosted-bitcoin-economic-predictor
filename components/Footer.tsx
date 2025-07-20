import { Bitcoin, Github } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="bg-slate-900/80 backdrop-blur-xl border-t border-white/10">
      <div className="container mx-auto px-4 py-12">
        {/* Project Info Section */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <div className="bg-gradient-to-r from-bitcoin to-orange-500 p-3 rounded-2xl">
              <Bitcoin className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white">Bitcoin Price Predictor</h3>
          </div>
          <p className="text-blue-200 mb-6 max-w-2xl mx-auto">
            Advanced AI-powered Bitcoin price prediction system using ensemble machine learning models,
            developed as part of NUST's Introduction to Data Science course project.
          </p>
        </div>

        {/* Team Section */}
        <div className="bg-white/5 rounded-2xl p-8 mb-8">
          <h4 className="text-lg font-semibold text-white mb-6 text-center">Meet the Team</h4>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center">
                <span className="text-white font-bold text-lg">MM</span>
              </div>
              <h5 className="text-white font-medium">Muhammad Muntazar</h5>
              <p className="text-blue-200 text-sm">Data Acquisition & Preprocessing Specialist</p>
            </div>
            <div className="text-center">
              <div className="bg-gradient-to-r from-green-500 to-teal-600 w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center">
                <span className="text-white font-bold text-lg">HA</span>
              </div>
              <h5 className="text-white font-medium">Hafiz Abdul Basit</h5>
              <p className="text-blue-200 text-sm">Data Transformation Engineer</p>
            </div>
            <div className="text-center">
              <div className="bg-gradient-to-r from-orange-500 to-red-600 w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center">
                <span className="text-white font-bold text-lg">MW</span>
              </div>
              <h5 className="text-white font-medium">Muhammad Wasif Shakeel</h5>
              <p className="text-blue-200 text-sm">AI Model Engineer</p>
            </div>
          </div>
        </div>

        {/* GitHub Link */}
        <div className="text-center mb-8">
          <a 
            href="https://github.com/mwasifshkeel/bitcoin-economic-predictor" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-3 bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-700 hover:to-gray-800 px-6 py-3 rounded-xl transition-all duration-300 transform hover:scale-105"
          >
            <Github className="w-5 h-5 text-white" />
            <span className="text-white font-medium">View Project on GitHub</span>
          </a>
        </div>

        {/* Copyright */}
        <div className="border-t border-white/10 pt-8 text-center">
          <p className="text-blue-200 text-sm mb-2">
            © 2024 Bitcoin Prediction App - NUST Data Science Project
          </p>
          <div className="flex items-center justify-center text-blue-200 text-sm">
            <span>Made for academic research and learning</span>
          </div>
        </div>
      </div>
    </footer>
  )
}