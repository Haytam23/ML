# S&P 500 Volatility Forecasting - Streamlit Deployment

## Project Overview

This project implements a Transformer-based machine learning model for forecasting S&P 500 volatility, deployed as an interactive web application using Streamlit.

### Model Specifications
- **Architecture**: Transformer Encoder
- **Best Configuration**: d_model=128, 2 encoder layers, dropout=0.2
- **Input**: Last 60 days of engineered features (27 features total)
- **Output**: 5-day ahead forecast of 30-day rolling volatility
- **Confidence Intervals**: 95% prediction intervals based on historical residuals

## Features

### 1. Real-Time Inference
- Upload historical S&P 500 data (CSV format)
- Automatic feature engineering (27 technical features)
- Point predictions with 95% confidence intervals
- 5-day ahead volatility forecasts

### 2. Feature Engineering
The model uses 27 engineered features:
- **Basic Features (4)**: Returns, Log Returns, HL Range, OC Range
- **Rolling Return Statistics (12)**: Mean, Std, Min, Max over 5, 10, 20-day windows
- **Rolling Volume Statistics (6)**: Mean, Std over 5, 10, 20-day windows
- **Rolling Volatility (3)**: Annualized volatility over 5, 10, 20-day windows
- **Momentum Indicators (2)**: 5-day and 10-day momentum

### 3. Visualization
- Historical true vs predicted volatility comparison
- 5-day forecast with confidence intervals
- Risk level assessment (Low/Moderate/High)
- Interactive Plotly charts

### 4. Prediction Intervals
95% confidence intervals calculated using:
- Historical prediction residuals from validation set
- Normal distribution assumption
- Z-score = 1.96 for 95% confidence level

## File Structure

```
├── app_v2.py                  # Main Streamlit application
├── sample_data.csv            # Example S&P 500 data
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── sp500_models_tuned/
    └── T_dm64_l2_lr1e4_best.pth  # Trained model weights
```

## Installation & Usage

### Local Deployment

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the app**:
```bash
streamlit run app_v2.py
```

3. **Access dashboard**:
- Open browser to `http://localhost:8501`
- Upload CSV file with S&P 500 data
- Generate 5-day volatility forecast

### Google Colab Deployment

1. **Upload files to Colab**:
```python
from google.colab import files
uploaded = files.upload()  # Upload app_v2.py
```

2. **Install dependencies**:
```python
!pip install streamlit plotly scipy
```

3. **Run with tunneling**:
```python
!streamlit run app_v2.py & npx localtunnel --port 8501
```

4. **Access via public URL**: Click the generated URL to access the dashboard

## Input Data Format

CSV file with the following columns:
```
Date,Open,High,Low,Close,Volume
2024-01-01,4500.50,4520.30,4495.20,4510.00,3500000000
2024-01-02,4510.00,4535.80,4505.00,4530.25,3600000000
...
```

**Requirements**:
- Minimum 60 days of historical data (80-90 days recommended)
- Daily OHLCV format
- Chronological order

## Model Performance

### Architecture Details
- Input dimension: 27 features
- Sequence length: 60 days
- d_model: 64 (configurable to 128)
- Attention heads: 4
- Encoder layers: 2
- Dropout: 0.1 (configurable to 0.2)
- Feedforward dimension: 128

### Training
- Dataset: S&P 500 historical data
- Features: 27 engineered technical indicators
- Target: 30-day rolling volatility
- Horizon: 5-day ahead forecast
- Validation: Time-based train/val/test split

## Key Components

### 1. Feature Engineering (`FeatureEngineer` class)
- Calculates 27 technical features from OHLCV data
- Handles missing values and normalization
- Creates sequences for model input

### 2. Model Architecture (`TransformerModel` class)
- Transformer encoder with configurable parameters
- Batch-first implementation
- Dropout regularization for robustness

### 3. Prediction with Intervals (`predict_with_intervals`)
- Point predictions from trained model
- 95% confidence intervals using historical residuals
- Scipy stats for normal distribution calculations

### 4. Visualization (`plot_*` functions)
- Interactive Plotly charts
- True vs predicted comparisons
- Confidence interval shading
- 5-day forecast visualization

## Deployment Options

### 1. Local Development
Best for testing and development:
- Fast iteration
- Full debugging capabilities
- Access to local file system

### 2. Google Colab
Best for demonstration and sharing:
- No local setup required
- GPU acceleration available
- Temporary public URL via tunneling
- Easy sharing with stakeholders

### 3. Production Deployment
Options for production use:
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Full cloud infrastructure
- **Docker**: Containerized deployment

## Usage Example

```python
# 1. Load your S&P 500 data
df = pd.read_csv('sp500_historical_data.csv')

# 2. Feature engineering
engineer = FeatureEngineer(sequence_length=60)
df_features = engineer.engineer_features(df)
X, feature_cols, mean, std = engineer.create_sequences(df_features)

# 3. Load model
model, device = load_model('sp500_models_tuned/T_dm64_l2_lr1e4_best.pth')

# 4. Generate prediction
point_pred, lower, upper = predict_with_intervals(
    model, X, device, historical_residuals, confidence=0.95
)

print(f"5-day forecast: {point_pred:.4f}")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
```

## Troubleshooting

### Issue: "Not enough data"
**Solution**: Upload at least 80-90 days of historical data

### Issue: "Feature mismatch"
**Solution**: Ensure CSV has all required columns (Date, Open, High, Low, Close, Volume)

### Issue: "Model loading error"
**Solution**: Verify model path points to correct .pth file

## Future Enhancements

1. **Multiple Horizons**: Add 1-day, 3-day, 7-day forecasts
2. **Ensemble Models**: Combine multiple model predictions
3. **Real-time Data**: Connect to financial data APIs
4. **Model Retraining**: Automated periodic retraining
5. **Portfolio Integration**: Multi-asset volatility forecasting

## References

- Original training notebook: `01_train_evaluate_models.ipynb`
- Model checkpoints: `sp500_models_tuned/` directory
- Streamlit documentation: https://docs.streamlit.io

## License

This project is for educational and research purposes.
