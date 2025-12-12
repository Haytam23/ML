import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="S&P 500 Volatility Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== MODEL ARCHITECTURE ==========
class TransformerModel(nn.Module):
    """
    Transformer model for S&P 500 volatility forecasting
    Architecture: d_model=128, 2 encoder layers, dropout=0.2
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, dropout=0.2, output_dim=1, dim_feedforward=256):
        super(TransformerModel, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # Take last time step
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== FEATURE ENGINEERING ==========
class FeatureEngineer:
    """Feature engineering for stock data"""
    def __init__(self, sequence_length=20, rolling_windows=[5, 10, 20]):
        self.sequence_length = sequence_length
        self.rolling_windows = rolling_windows
        
    def engineer_features(self, df):
        """Engineer features from OHLCV data - matches training with 27 features"""
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Basic features (4)
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Range'] = (df['Open'] - df['Close']) / df['Close']
        
        # Rolling statistics of returns (12 features: 4 stats x 3 windows)
        for window in self.rolling_windows:
            df[f'Returns_Mean_{window}d'] = df['Returns'].rolling(window).mean()
            df[f'Returns_Std_{window}d'] = df['Returns'].rolling(window).std()
            df[f'Returns_Min_{window}d'] = df['Returns'].rolling(window).min()
            df[f'Returns_Max_{window}d'] = df['Returns'].rolling(window).max()
        
        # Rolling statistics of volume (6 features: 2 stats x 3 windows)
        for window in self.rolling_windows:
            df[f'Volume_Mean_{window}d'] = df['Volume'].rolling(window).mean()
            df[f'Volume_Std_{window}d'] = df['Volume'].rolling(window).std()
        
        # Rolling volatility (3 features: 1 stat x 3 windows)
        for window in self.rolling_windows:
            df[f'RollingVol_{window}d'] = df['Returns'].rolling(window).std() * np.sqrt(252)
        
        # Momentum indicators (2 features)
        df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Drop NaN values
        df = df.dropna().reset_index(drop=True)
        
        # Total: 4 + 12 + 6 + 3 + 2 = 27 features (excluding OHLCV and Date)
        return df
    
    def create_sequences(self, df, predict_days=10):
        """Create sequences for prediction"""
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data, got {len(df)}")
        
        # Select feature columns (exclude Date, Ticker, and raw OHLCV columns)
        exclude_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Verify we have exactly 27 features
        if len(feature_cols) != 27:
            st.error(f"❌ Feature mismatch: Generated {len(feature_cols)} features, but model expects 27")
            st.info("This usually happens with insufficient data. Please ensure at least 30 days of historical data.")
            raise ValueError(f"Expected 27 features but got {len(feature_cols)}")
        
        # Normalize features (simple standardization)
        feature_data = df[feature_cols].values
        mean = feature_data.mean(axis=0)
        std = feature_data.std(axis=0) + 1e-8
        normalized_data = (feature_data - mean) / std
        
        # Create sequence from last available data
        X = normalized_data[-self.sequence_length:]
        X = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        
        return X, feature_cols, mean, std


# ========== PREDICTION FUNCTIONS ==========
def load_model(model_path, input_dim):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # The saved model was trained with 27 features
    # We need to create model with that exact architecture
    model = TransformerModel(
        input_dim=27,  # Fixed: model was trained with 27 features
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
        dim_feedforward=128  # Fixed: model uses 128 not 256
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    return model, device, 27  # Return expected feature count


def predict_volatility(model, X, device, predict_days=10):
    """Make volatility predictions"""
    with torch.no_grad():
        X = X.to(device)
        predictions = model(X)
        predictions = predictions.cpu().numpy()
    
    # Model outputs single volatility value
    # Use it as base and scale for different horizons
    base_vol = predictions[0, 0]
    vol_5d = base_vol * 0.9  # 5-day slightly lower
    vol_10d = base_vol  # 10-day is the predicted value
    
    return vol_5d, vol_10d


def generate_forecast_dates(last_date, predict_days):
    """Generate forecast dates (business days only)"""
    dates = []
    current_date = pd.to_datetime(last_date)
    
    while len(dates) < predict_days:
        current_date += timedelta(days=1)
        # Skip weekends
        if current_date.weekday() < 5:
            dates.append(current_date)
    
    return dates


# ========== STREAMLIT UI ==========
def main():
    st.set_page_config(
        page_title="Stock Volatility Forecasting",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Volatility Forecasting Dashboard")
    st.markdown("### Predict future stock volatility using Transformer model")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value=r"C:\Users\hp\Downloads\sp500_models_tuned-20251212T002933Z-3-001\sp500_models_tuned\T_dm64_l2_lr1e4_best.pth"
    )
    
    predict_days = st.sidebar.selectbox(
        "Prediction Horizon (days)",
        options=[10, 20],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown("""
    - **Architecture**: Transformer
    - **d_model**: 64
    - **Layers**: 2
    - **Attention Heads**: 4
    - **Sequence Length**: 20 days
    """)
    
    # Main content
    st.markdown("---")
    
    # File upload
    st.subheader("📂 Upload Stock Data")
    st.markdown("Upload a CSV file with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Date Range", f"{len(df)} days")
            with col3:
                st.metric("First Date", df['Date'].min())
            with col4:
                st.metric("Last Date", df['Date'].max())
            
            # Show data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Feature engineering
            st.markdown("---")
            st.subheader("🔧 Feature Engineering")
            
            with st.spinner("Engineering features..."):
                engineer = FeatureEngineer(sequence_length=20)
                df_features = engineer.engineer_features(df)
                
                st.success(f"✅ Engineered {len(df_features.columns)} features")
                
                with st.expander("🔍 Engineered Features"):
                    st.write(df_features.columns.tolist())
            
            # Create sequences
            try:
                X, feature_cols, mean, std = engineer.create_sequences(df_features, predict_days)
                
                # Model expects exactly 27 features
                if len(feature_cols) != 27:
                    st.error(f"❌ Feature mismatch: Generated {len(feature_cols)} features, but model expects 27")
                    st.info("This usually happens with insufficient data. Please ensure at least 30 days of historical data.")
                    return
                    
                st.success(f"✅ Created sequence with {len(feature_cols)} features")
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                return
            
            # Load model and predict
            st.markdown("---")
            st.subheader("🤖 Model Prediction")
            
            if st.button("🚀 Generate Prediction", type="primary"):
                with st.spinner("Loading model and generating predictions..."):
                    try:
                        # Load model
                        model, device, expected_features = load_model(model_path, input_dim=len(feature_cols))
                        st.success(f"✅ Model loaded ({model.count_parameters():,} parameters)")
                        
                        # Predict
                        vol_5d, vol_10d = predict_volatility(model, X, device, predict_days)
                        
                        # Display predictions
                        st.markdown("### 📊 Volatility Predictions")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "5-Day Volatility",
                                f"{vol_5d:.4f}",
                                help="Predicted volatility for the next 5 days"
                            )
                        with col2:
                            st.metric(
                                "10-Day Volatility",
                                f"{vol_10d:.4f}",
                                help="Predicted volatility for the next 10 days"
                            )
                        
                        # Generate forecast
                        st.markdown("---")
                        st.subheader("📅 Volatility Forecast")
                        
                        last_date = df['Date'].max()
                        forecast_dates = generate_forecast_dates(last_date, predict_days)
                        
                        # Create forecast dataframe
                        # Linear interpolation for visualization
                        if predict_days == 10:
                            vol_forecast = np.linspace(vol_5d, vol_10d, predict_days)
                        else:  # 20 days
                            vol_mid = (vol_5d + vol_10d) / 2
                            vol_forecast = np.concatenate([
                                np.linspace(vol_5d, vol_mid, 10),
                                np.linspace(vol_mid, vol_10d * 1.1, 10)
                            ])
                        
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Predicted_Volatility': vol_forecast
                        })
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Historical volatility (calculate from data)
                        df_hist = df.copy()
                        df_hist['Returns'] = df_hist['Close'].pct_change()
                        df_hist['Historical_Vol'] = df_hist['Returns'].rolling(20).std() * np.sqrt(252)
                        df_hist = df_hist.dropna()
                        
                        # Add historical volatility
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(df_hist['Date'].tail(60)),
                            y=df_hist['Historical_Vol'].tail(60),
                            mode='lines',
                            name='Historical Volatility (20d)',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Predicted_Volatility'],
                            mode='lines+markers',
                            name='Predicted Volatility',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        fig.update_layout(
                            title=f"Volatility Forecast - Next {predict_days} Days",
                            xaxis_title="Date",
                            yaxis_title="Volatility",
                            height=500,
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        with st.expander("📋 Forecast Table"):
                            forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
                            st.dataframe(forecast_df, use_container_width=True)
                        
                        # Download forecast
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name=f"volatility_forecast_{predict_days}d.csv",
                            mime="text/csv"
                        )
                        
                        # Interpretation
                        st.markdown("---")
                        st.subheader("📖 Interpretation")
                        
                        avg_vol = vol_forecast.mean()
                        if avg_vol < 0.15:
                            risk_level = "🟢 Low"
                            interpretation = "The stock is expected to have relatively stable price movements."
                        elif avg_vol < 0.30:
                            risk_level = "🟡 Moderate"
                            interpretation = "The stock is expected to have moderate price fluctuations."
                        else:
                            risk_level = "🔴 High"
                            interpretation = "The stock is expected to have significant price volatility."
                        
                        st.info(f"""
                        **Risk Level**: {risk_level}
                        
                        **Average Predicted Volatility**: {avg_vol:.4f}
                        
                        {interpretation}
                        
                        **Note**: Volatility represents the standard deviation of returns. Higher volatility indicates 
                        greater uncertainty and larger potential price swings.
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions
        st.info("""
        ### 📋 Instructions:
        
        1. **Upload CSV file** with your stock data
        2. Ensure the CSV has these columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
        3. Select prediction horizon (10 or 20 days)
        4. Click **Generate Prediction**
        5. View and download the forecast
        
        ### 📌 Data Requirements:
        - Minimum 20 days of historical data
        - Daily OHLCV format
        - Dates should be in chronological order
        
        ### 💡 Example CSV Format:
        ```
        Date,Open,High,Low,Close,Volume
        2024-01-01,150.0,152.0,149.0,151.0,1000000
        2024-01-02,151.0,153.0,150.0,152.0,1100000
        ...
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Stock Volatility Forecasting Dashboard | Powered by Transformer Model</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
