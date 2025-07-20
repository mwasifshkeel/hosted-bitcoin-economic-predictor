import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

interface PredictionInput {
  open_price: number
  close_price: number
  high_price: number
  low_price: number
  volume: number
  news_headline?: string
}

interface PredictionResult {
  prediction: number
  confidence: number
  model_performance: {
    rmse: number
    mae: number
    r2: number
  }
  feature_importance: Array<{
    feature: string
    importance: number
  }>
  individual_predictions: {
    random_forest: number
    ridge: number
    xgboost: number
    meta_model: number
  }
  status: string
}

function executePythonScript(inputData: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonScriptPath = path.join(process.cwd(), 'model', 'predict.py')
    
    const pythonCommands = ['python', 'python3', 'py']
    let currentCommand = 0
    
    function tryPythonCommand() {
      if (currentCommand >= pythonCommands.length) {
        reject(new Error('No working Python installation found'))
        return
      }
      
      const pythonProcess = spawn(pythonCommands[currentCommand], [pythonScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      })
      
      let outputData = ''
      let errorData = ''
      
      pythonProcess.stdout.on('data', (data) => {
        outputData += data.toString()
      })
      
      pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString()
      })
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(outputData)
            resolve(result)
          } catch (parseError) {
            reject(new Error(`Failed to parse Python output: ${parseError.message}\nOutput: ${outputData}`))
          }
        } else {
          console.error(`Python stderr: ${errorData}`)
          console.error(`Python stdout: ${outputData}`)
          
          currentCommand++
          if (currentCommand < pythonCommands.length) {
            tryPythonCommand()
          } else {
            reject(new Error(`Python script failed with code ${code}. Error: ${errorData}. Output: ${outputData}`))
          }
        }
      })
      
      pythonProcess.on('error', (error) => {
        console.error(`Python process error: ${error.message}`)
        currentCommand++
        if (currentCommand < pythonCommands.length) {
          tryPythonCommand()
        } else {
          reject(new Error(`Failed to start Python process: ${error.message}`))
        }
      })
      
      try {
        pythonProcess.stdin.write(JSON.stringify(inputData))
        pythonProcess.stdin.end()
      } catch (writeError) {
        reject(new Error(`Failed to write to Python process: ${writeError.message}`))
      }
    }
    
    tryPythonCommand()
  })
}

function calculateSentimentScore(newsHeadline: string, marketData: any): number {
  /**
   * Calculate sentiment score based on news headline and market data
   * Mimics the logic from the notebook's calculate_sentiment_score function
   */
  let score = 0;
  
  if (!newsHeadline || newsHeadline.trim().length === 0) {
    return 0.0; // Neutral sentiment for empty headlines
  }
  
  const headline = newsHeadline.toLowerCase();
  
  // Factor 1: Base sentiment from keyword analysis
  const positiveWords = [
    'bull', 'bullish', 'rise', 'rising', 'increase', 'up', 'gain', 'gains', 
    'growth', 'positive', 'surge', 'rally', 'boom', 'breakthrough', 'adoption',
    'institutional', 'investment', 'buy', 'buying', 'support', 'strong',
    'record', 'high', 'milestone', 'success', 'approve', 'approved'
  ];
  
  const negativeWords = [
    'bear', 'bearish', 'fall', 'falling', 'decrease', 'down', 'drop', 'crash',
    'decline', 'negative', 'sell', 'selling', 'dump', 'fear', 'uncertainty',
    'regulation', 'ban', 'banned', 'hack', 'hacked', 'scam', 'fraud',
    'low', 'bottom', 'concern', 'warning', 'risk', 'volatile', 'bubble'
  ];
  
  // Count positive and negative words
  let positiveCount = 0;
  let negativeCount = 0;
  
  positiveWords.forEach(word => {
    if (headline.includes(word)) positiveCount++;
  });
  
  negativeWords.forEach(word => {
    if (headline.includes(word)) negativeCount++;
  });
  
  // Base sentiment score from word analysis
  const wordSentiment = (positiveCount - negativeCount) * 0.2;
  score += wordSentiment;
  
  // Factor 2: Market momentum analysis
  const priceChange = marketData.high_price - marketData.low_price;
  const priceRange = marketData.high_price - marketData.low_price;
  const relativeRange = priceRange / marketData.open_price;
  
  // Add market momentum factor
  if (relativeRange > 0.05) {
    score += headline.includes('volatile') ? -0.1 : 0.05;
  }
  
  // Factor 3: Volume analysis
  const volumeFactor = Math.min(marketData.volume / 10000, 0.1); // Cap volume influence
  if (positiveCount > negativeCount) {
    score += volumeFactor;
  } else if (negativeCount > positiveCount) {
    score -= volumeFactor;
  }
  
  // Factor 4: Specific Bitcoin/crypto keywords
  const cryptoPositive = ['bitcoin', 'btc', 'cryptocurrency', 'blockchain', 'etf', 'halving'];
  const cryptoNegative = ['regulation', 'tax', 'government', 'central bank'];
  
  cryptoPositive.forEach(word => {
    if (headline.includes(word) && positiveCount > 0) score += 0.1;
  });
  
  cryptoNegative.forEach(word => {
    if (headline.includes(word)) score -= 0.15;
  });
  
  // Normalize score to range between -1 and 1
  score = Math.max(Math.min(score, 1), -1);
  
  // Add some randomization to avoid perfect scores
  if (Math.abs(score) === 1.0) {
    const randomFactor = 0.85 + Math.random() * 0.14;
    score *= randomFactor;
  }
  
  return Math.round(score * 100) / 100;
}

export async function POST(request: NextRequest) {
  try {
    const data: PredictionInput = await request.json()
    
    // Validate required fields
    const requiredFields = ['open_price', 'high_price', 'low_price', 'volume']
    for (const field of requiredFields) {
      if (!(field in data) || data[field] === null || data[field] === undefined) {
        return NextResponse.json(
          { error: `Missing or invalid required field: ${field}` },
          { status: 400 }
        )
      }
    }
    
    // Validate data ranges
    if (data.open_price <= 0 || data.high_price <= 0 || data.low_price <= 0 || data.volume < 0) {
      return NextResponse.json(
        { error: 'Price values must be positive and volume must be non-negative' },
        { status: 400 }
      )
    }
    
    if (data.high_price < data.low_price) {
      return NextResponse.json(
        { error: 'High price cannot be less than low price' },
        { status: 400 }
      )
    }
    
    // Calculate sentiment score from news headline
    const sentiment_score = calculateSentimentScore(data.news_headline || '', {
      open_price: data.open_price,
      high_price: data.high_price,
      low_price: data.low_price,
      volume: data.volume
    });
    
    // Set defaults and add calculated sentiment
    const inputData = {
      open_price: data.open_price,
      close_price: data.close_price || data.open_price,
      high_price: data.high_price,
      low_price: data.low_price,
      volume: data.volume,
      sentiment_score: sentiment_score,
      news_headline: data.news_headline ?? ''
    }
    
    // Execute Python prediction script
    console.log('Executing prediction with data:', inputData)
    const result = await executePythonScript(inputData)
    console.log('Prediction result:', result)
    
    return NextResponse.json({
      prediction: result.prediction,
      confidence: result.confidence,
      model_performance: result.model_performance,
      feature_importance: result.feature_importance,
      individual_predictions: result.individual_predictions,
      calculated_sentiment: sentiment_score, // Include calculated sentiment in response
      using_fallback: result.using_fallback || false,
      status: 'success'
    })
    
  } catch (error) {
    console.error('Prediction error:', error)
    
    if (error.message.includes('No working Python installation found')) {
      return NextResponse.json(
        { 
          error: 'Python environment not available. Please ensure Python is installed and accessible.',
          details: error.message,
          status: 'error' 
        },
        { status: 500 }
      )
    }
    
    if (error.message.includes('Python script failed')) {
      return NextResponse.json(
        { 
          error: 'Model prediction script encountered an error.',
          details: error.message,
          status: 'error' 
        },
        { status: 500 }
      )
    }
    
    return NextResponse.json(
      { 
        error: 'Internal server error during prediction',
        details: error.message,
        status: 'error' 
      },
      { status: 500 }
    )
  }
}
