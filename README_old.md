# Stock Volatility Forecasting Dashboard

A Streamlit dashboard for predicting stock volatility using a trained Transformer model.

## Features

- 📊 Upload CSV files with stock data (OHLCV format)
- 🤖 Automatic feature engineering (25+ features)
- 📈 Predict volatility for next 10 or 20 days
- 📉 Interactive visualizations with historical data
- 📥 Download forecast results as CSV

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the dashboard:
```bash
streamlit run app.py
```

2. Upload your stock data CSV file with these columns:
   - Date
   - Open
   - High
   - Low
   - Close
   - Volume

3. Select prediction horizon (10 or 20 days)

4. Click "Generate Prediction" to see forecasts

## Model Information

- **Architecture**: Transformer
- **d_model**: 64
- **Layers**: 2
- **Attention Heads**: 4
- **Sequence Length**: 20 days
- **Model Path**: `sp500_models_tuned/T_dm64_l2_lr1e4_best.pth`

## CSV Format Example

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.0,152.0,149.0,151.0,1000000
2024-01-02,151.0,153.0,150.0,152.0,1100000
2024-01-03,152.0,154.0,151.0,153.0,1200000
```

## Requirements

- Minimum 20 days of historical data
- Daily OHLCV format
- Chronological order

## Output

The dashboard provides:
- 5-day and 10-day volatility predictions
- Volatility forecast chart with historical comparison
- Risk level assessment (Low/Moderate/High)
- Downloadable forecast CSV
- Interpretation guidance

## Notes

- Volatility represents the standard deviation of returns
- Higher volatility indicates greater uncertainty and larger potential price swings
- The model uses the last 20 days of data to make predictions
