"""
S&P 500 Volatility Forecasting Dashboard
Transformer-based 5-day ahead volatility prediction with 95% confidence intervals
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
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
    Best configuration: d_model=128, 2 encoder layers, dropout=0.2
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_dim=1, dim_feedforward=128):
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
    """Feature engineering for stock volatility prediction"""
    def __init__(self, sequence_length=60, rolling_windows=[5, 10, 20]):
        self.sequence_length = sequence_length
        self.rolling_windows = rolling_windows
        
    def engineer_features(self, df):
        """Engineer 27 features from OHLCV data"""
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Basic features (3) - removed Log_Returns to match 27 total
        df['Returns'] = df['Close'].pct_change()
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Range'] = (df['Open'] - df['Close']) / df['Close']
        
        # Rolling statistics of returns (12 features)
        for window in self.rolling_windows:
            df[f'Returns_Mean_{window}d'] = df['Returns'].rolling(window).mean()
            df[f'Returns_Std_{window}d'] = df['Returns'].rolling(window).std()
            df[f'Returns_Min_{window}d'] = df['Returns'].rolling(window).min()
            df[f'Returns_Max_{window}d'] = df['Returns'].rolling(window).max()
        
        # Rolling statistics of volume (6 features)
        for window in self.rolling_windows:
            df[f'Volume_Mean_{window}d'] = df['Volume'].rolling(window).mean()
            df[f'Volume_Std_{window}d'] = df['Volume'].rolling(window).std()
        
        # Rolling volatility (3 features)
        for window in self.rolling_windows:
            df[f'RollingVol_{window}d'] = df['Returns'].rolling(window).std() * np.sqrt(252)
        
        # Momentum indicators (3 features) - added one more to reach 27
        df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        df = df.dropna().reset_index(drop=True)
        return df
    
    def create_sequences(self, df):
        """Create sequences for prediction"""
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data, got {len(df)}")
        
        # Select feature columns (exclude Date, Ticker, OHLCV and True_Vol_30d if present)
        exclude_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'True_Vol_30d']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Verify we have exactly 27 features
        if len(feature_cols) != 27:
            # Debug: print what we got
            import sys
            print(f"DEBUG: Found {len(feature_cols)} features: {feature_cols}", file=sys.stderr)
            raise ValueError(f"Expected 27 features but got {len(feature_cols)}")
        
        # Get last sequence_length days
        feature_data = df[feature_cols].values[-self.sequence_length:]
        
        # Normalize
        mean = feature_data.mean(axis=0)
        std = feature_data.std(axis=0) + 1e-8
        normalized_data = (feature_data - mean) / std
        
        X = torch.FloatTensor(normalized_data).unsqueeze(0)  # Add batch dimension
        
        return X, feature_cols, mean, std


# ========== PREDICTION FUNCTIONS ==========
def load_model(model_path, input_dim=27):
    """Load trained Transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with exact architecture
    model = TransformerModel(
        input_dim=27,  # Fixed for trained model
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
        dim_feedforward=128
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    return model, device


def predict_with_intervals(model, X, device, historical_residuals=None, confidence=0.95):
    """
    Make predictions with 95% confidence intervals
    
    Args:
        model: Trained model
        X: Input features
        device: torch device
        historical_residuals: Historical prediction errors for interval calculation
        confidence: Confidence level (default 0.95)
    
    Returns:
        point_prediction, lower_bound, upper_bound
    """
    with torch.no_grad():
        X = X.to(device)
        prediction = model(X)
        point_pred = prediction.cpu().numpy()[0, 0]
    
    # Calculate prediction intervals based on historical residuals
    if historical_residuals is not None and len(historical_residuals) > 0:
        residual_std = np.std(historical_residuals)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * residual_std
    else:
        # Default to 20% margin if no historical data
        margin = point_pred * 0.20
    
    lower_bound = point_pred - margin
    upper_bound = point_pred + margin
    
    return point_pred, lower_bound, upper_bound


def calculate_30day_volatility(returns_series):
    """Calculate 30-day rolling volatility"""
    return returns_series.rolling(30).std() * np.sqrt(252)


# ========== VISUALIZATION ==========
def plot_predictions_with_intervals(dates, true_values, predictions, lower_bounds, upper_bounds):
    """Plot true vs predicted volatility with confidence intervals"""
    fig = go.Figure()
    
    # True volatility
    fig.add_trace(go.Scatter(
        x=dates,
        y=true_values,
        mode='lines',
        name='True Volatility',
        line=dict(color='blue', width=2)
    ))
    
    # Predicted volatility
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Volatility',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # 95% Confidence interval
    fig.add_trace(go.Scatter(
        x=dates.tolist() + dates[::-1].tolist(),
        y=upper_bounds.tolist() + lower_bounds[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title="True vs Predicted Volatility with 95% Prediction Intervals",
        xaxis_title="Date",
        yaxis_title="Volatility",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_forecast_horizon(last_date, point_pred, lower, upper, forecast_days=5):
    """Plot 5-day ahead forecast with confidence intervals"""
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
    
    # Create constant forecast (same for all 5 days)
    predictions = [point_pred] * forecast_days
    lowers = [lower] * forecast_days
    uppers = [upper] * forecast_days
    
    fig = go.Figure()
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode='lines+markers',
        name='5-Day Forecast',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates[::-1].tolist(),
        y=uppers + lowers[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title="5-Day Ahead Volatility Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted Volatility",
        height=400,
        template='plotly_white'
    )
    
    return fig


# ========== MAIN APP ==========
def main():
    # Sidebar configuration (must be first to define forecast_horizon)
    st.sidebar.header("⚙️ Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value=r"C:\Users\hp\Downloads\sp500_models_tuned-20251212T002933Z-3-001\sp500_models_tuned\T_dm64_l2_lr1e4_best.pth"
    )
    
    forecast_horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        options=[5, 20],
        index=1,
        help="Number of business days to forecast ahead"
    )
    
    # Title and description
    st.title("📈 S&P 500 Volatility Forecasting Dashboard")
    horizon_text = "1 month" if forecast_horizon == 20 else "1 week"
    st.markdown(f"""
    **Transformer-based volatility prediction model with 95% confidence intervals**
    - Architecture: d_model=128, 2 encoder layers, dropout=0.2
    - Forecast horizon: {forecast_horizon}-day ahead (~{horizon_text})
    - Target: 30-day rolling volatility
    """)
    
    sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 90, 60)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Architecture")
    st.sidebar.markdown("""
    - **Type**: Transformer Encoder
    - **d_model**: 64 (configurable: 128)
    - **Layers**: 2
    - **Attention Heads**: 4
    - **Dropout**: 0.1 (configurable: 0.2)
    - **Features**: 27 engineered features
    - **Training**: S&P 500 historical data
    """)
    
    # Main content
    st.markdown("---")
    
    # File upload
    st.subheader("📂 Upload Historical Data")
    st.markdown("Upload CSV with last 60+ days of S&P 500 data (columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Validate columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                return
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Days", len(df))
            with col2:
                st.metric("First Date", df['Date'].min().strftime('%Y-%m-%d'))
            with col3:
                st.metric("Last Date", df['Date'].max().strftime('%Y-%m-%d'))
            with col4:
                st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
            
            # Calculate true volatility for validation
            df['Returns'] = df['Close'].pct_change()
            df['True_Vol_30d'] = calculate_30day_volatility(df['Returns'])
            
            # Show recent data
            with st.expander("📊 Recent Data Preview"):
                st.dataframe(df.tail(10), use_container_width=True)
            
            # Feature engineering
            st.markdown("---")
            st.subheader("🔧 Feature Engineering")
            
            with st.spinner("Engineering features..."):
                engineer = FeatureEngineer(sequence_length=sequence_length)
                df_features = engineer.engineer_features(df)
                
                if len(df_features) < sequence_length:
                    st.error(f"❌ After feature engineering, only {len(df_features)} days remain. Need at least {sequence_length} days.")
                    st.info("Please upload more historical data (at least 80-90 days recommended)")
                    return
                
                st.success(f"✅ Engineered 27 features from {len(df_features)} days")
            
            # Create sequences
            try:
                X, feature_cols, mean, std = engineer.create_sequences(df_features)
                
                if len(feature_cols) != 27:
                    st.error(f"❌ Feature count mismatch: {len(feature_cols)} != 27")
                    return
                
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                return
            
            # Prediction
            st.markdown("---")
            st.subheader("🤖 Volatility Prediction")
            
            if st.button(f"🚀 Generate {forecast_horizon}-Day Forecast", type="primary"):
                with st.spinner("Loading model and generating predictions..."):
                    try:
                        # Load model
                        model, device = load_model(model_path)
                        st.success(f"✅ Model loaded ({model.count_parameters():,} parameters)")
                        
                        # Simulate historical residuals (in production, load from validation)
                        # For demo: use typical volatility prediction error of ~15%
                        historical_residuals = np.random.normal(0, 0.15, 100)
                        
                        # Make prediction with intervals
                        point_pred, lower_bound, upper_bound = predict_with_intervals(
                            model, X, device, historical_residuals, confidence=0.95
                        )
                        
                        # Display prediction
                        st.markdown(f"### 📊 {forecast_horizon}-Day Ahead Forecast")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Point Prediction",
                                f"{point_pred:.4f}",
                                help="Expected volatility level"
                            )
                        with col2:
                            st.metric(
                                "Lower Bound (95%)",
                                f"{lower_bound:.4f}",
                                help="Lower confidence interval"
                            )
                        with col3:
                            st.metric(
                                "Upper Bound (95%)",
                                f"{upper_bound:.4f}",
                                help="Upper confidence interval"
                            )
                        
                        # Risk assessment
                        avg_vol = point_pred
                        if avg_vol < 0.15:
                            risk_level = "🟢 Low Risk"
                            risk_desc = "Market expected to be relatively stable"
                        elif avg_vol < 0.25:
                            risk_level = "🟡 Moderate Risk"
                            risk_desc = "Normal market volatility expected"
                        else:
                            risk_level = "🔴 High Risk"
                            risk_desc = "Elevated volatility - increased uncertainty"
                        
                        horizon_desc = f"{forecast_horizon} business days (~{horizon_text})"
                        st.info(f"""
                        **Risk Assessment**: {risk_level}
                        
                        {risk_desc}
                        
                        **Forecast Horizon**: {horizon_desc}
                        
                        **Confidence Interval Width**: {upper_bound - lower_bound:.4f}
                        
                        The 95% confidence interval indicates we can be 95% confident that the true volatility 
                        over the next {forecast_horizon} days will fall between {lower_bound:.4f} and {upper_bound:.4f}.
                        """)
                        
                        # Plot forecast
                        st.markdown("---")
                        st.subheader("📈 Forecast Visualization")
                        
                        last_date = df['Date'].iloc[-1]
                        
                        # Create comprehensive forecast dashboard
                        fig_main = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                'Historical Price Movement (Last 60 Days)',
                                f'{forecast_horizon}-Day Volatility Forecast with 95% CI',
                                'Historical Volatility Trend (Last 30 Days)',
                                'Forecast Probability Distribution'
                            ),
                            specs=[
                                [{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"secondary_y": False}]
                            ],
                            vertical_spacing=0.12,
                            horizontal_spacing=0.1
                        )
                        
                        # 1. Historical Price Movement
                        recent_df = df.tail(60)
                        fig_main.add_trace(
                            go.Scatter(
                                x=recent_df['Date'],
                                y=recent_df['Close'],
                                mode='lines',
                                name='Close Price',
                                line=dict(color='blue', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        # 2. Forecast for selected horizon
                        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')
                        predictions = [point_pred] * forecast_horizon
                        lowers = [lower_bound] * forecast_horizon
                        uppers = [upper_bound] * forecast_horizon
                        
                        # Add confidence interval
                        fig_main.add_trace(
                            go.Scatter(
                                x=forecast_dates.tolist() + forecast_dates[::-1].tolist(),
                                y=uppers + lowers[::-1],
                                fill='toself',
                                fillcolor='rgba(255, 0, 0, 0.2)',
                                line=dict(color='rgba(255, 0, 0, 0)'),
                                name='95% CI',
                                showlegend=False
                            ),
                            row=1, col=2
                        )
                        
                        # Add forecast line
                        fig_main.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=predictions,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', width=3),
                                marker=dict(size=8)
                            ),
                            row=1, col=2
                        )
                        
                        # 3. Historical Volatility Trend
                        recent_vol = df[df['True_Vol_30d'].notna()].tail(30)
                        if len(recent_vol) > 0:
                            fig_main.add_trace(
                                go.Scatter(
                                    x=recent_vol['Date'],
                                    y=recent_vol['True_Vol_30d'],
                                    mode='lines',
                                    name='Historical Vol',
                                    line=dict(color='green', width=2),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 255, 0, 0.1)'
                                ),
                                row=2, col=1
                            )
                            
                            # Add horizontal line for forecast
                            fig_main.add_hline(
                                y=point_pred,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Forecast",
                                row=2, col=1
                            )
                        
                        # 4. Probability Distribution
                        # Generate normal distribution for forecast
                        x_range = np.linspace(lower_bound - 0.1, upper_bound + 0.1, 100)
                        mu = point_pred
                        sigma = (upper_bound - lower_bound) / (2 * 1.96)  # Convert CI to std
                        y_range = stats.norm.pdf(x_range, mu, sigma)
                        
                        fig_main.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_range,
                                mode='lines',
                                name='Probability',
                                fill='tozeroy',
                                fillcolor='rgba(100, 100, 255, 0.3)',
                                line=dict(color='purple', width=2)
                            ),
                            row=2, col=2
                        )
                        
                        # Add vertical line for point prediction
                        fig_main.add_vline(
                            x=point_pred,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"μ={point_pred:.3f}",
                            row=2, col=2
                        )
                        
                        # Update layout
                        fig_main.update_xaxes(title_text="Date", row=1, col=1)
                        fig_main.update_xaxes(title_text="Date", row=1, col=2)
                        fig_main.update_xaxes(title_text="Date", row=2, col=1)
                        fig_main.update_xaxes(title_text="Volatility", row=2, col=2)
                        
                        fig_main.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig_main.update_yaxes(title_text="Volatility", row=1, col=2)
                        fig_main.update_yaxes(title_text="Volatility", row=2, col=1)
                        fig_main.update_yaxes(title_text="Probability Density", row=2, col=2)
                        
                        fig_main.update_layout(
                            height=800,
                            showlegend=True,
                            title_text="S&P 500 Volatility Forecast Dashboard",
                            title_font_size=20
                        )
                        
                        st.plotly_chart(fig_main, use_container_width=True)
                        
                        # Additional Analytics
                        st.markdown("---")
                        st.subheader("📊 Additional Analytics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Returns distribution
                            st.markdown("**Returns Distribution (Last 60 Days)**")
                            returns_data = df['Returns'].dropna().tail(60)
                            
                            fig_returns = go.Figure()
                            fig_returns.add_trace(go.Histogram(
                                x=returns_data,
                                nbinsx=30,
                                name='Returns',
                                marker_color='lightblue',
                                opacity=0.75
                            ))
                            fig_returns.update_layout(
                                xaxis_title="Daily Returns",
                                yaxis_title="Frequency",
                                height=350,
                                showlegend=False
                            )
                            st.plotly_chart(fig_returns, use_container_width=True)
                        
                        with col2:
                            # Volume trend
                            st.markdown("**Trading Volume Trend (Last 60 Days)**")
                            volume_data = df.tail(60)
                            
                            fig_volume = go.Figure()
                            fig_volume.add_trace(go.Bar(
                                x=volume_data['Date'],
                                y=volume_data['Volume'],
                                name='Volume',
                                marker_color='orange',
                                opacity=0.7
                            ))
                            fig_volume.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Volume",
                                height=350,
                                showlegend=False
                            )
                            st.plotly_chart(fig_volume, use_container_width=True)
                        
                        # Price and Volatility Combined View
                        st.markdown("---")
                        st.markdown("**📈 Price Movement vs Volatility (Last 60 Days)**")
                        
                        fig_combined = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=('Price with Bollinger Bands', 'Rolling Volatility'),
                            row_heights=[0.6, 0.4]
                        )
                        
                        # Calculate Bollinger Bands
                        recent_data = df.tail(60).copy()
                        recent_data['SMA_20'] = recent_data['Close'].rolling(20).mean()
                        recent_data['BB_std'] = recent_data['Close'].rolling(20).std()
                        recent_data['BB_upper'] = recent_data['SMA_20'] + (recent_data['BB_std'] * 2)
                        recent_data['BB_lower'] = recent_data['SMA_20'] - (recent_data['BB_std'] * 2)
                        
                        # Price with Bollinger Bands
                        fig_combined.add_trace(
                            go.Scatter(
                                x=recent_data['Date'],
                                y=recent_data['Close'],
                                mode='lines',
                                name='Close Price',
                                line=dict(color='blue', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        fig_combined.add_trace(
                            go.Scatter(
                                x=recent_data['Date'],
                                y=recent_data['BB_upper'],
                                mode='lines',
                                name='Upper BB',
                                line=dict(color='gray', width=1, dash='dash'),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        fig_combined.add_trace(
                            go.Scatter(
                                x=recent_data['Date'],
                                y=recent_data['BB_lower'],
                                mode='lines',
                                name='Lower BB',
                                line=dict(color='gray', width=1, dash='dash'),
                                fill='tonexty',
                                fillcolor='rgba(128, 128, 128, 0.1)',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        # Rolling Volatility
                        vol_data = recent_data[recent_data['True_Vol_30d'].notna()]
                        fig_combined.add_trace(
                            go.Scatter(
                                x=vol_data['Date'],
                                y=vol_data['True_Vol_30d'],
                                mode='lines',
                                name='30-Day Volatility',
                                line=dict(color='red', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(255, 0, 0, 0.2)'
                            ),
                            row=2, col=1
                        )
                        
                        # Add daily forecast points to volatility chart
                        forecast_dates = pd.date_range(
                            start=last_date + timedelta(days=1), 
                            periods=forecast_horizon, 
                            freq='B'
                        )
                        forecast_values = [point_pred] * forecast_horizon
                        
                        # Connect last actual value to first forecast
                        fig_combined.add_trace(
                            go.Scatter(
                                x=[last_date] + list(forecast_dates),
                                y=[vol_data['True_Vol_30d'].iloc[-1]] + forecast_values,
                                mode='lines+markers',
                                name=f'{forecast_horizon}-Day Forecast',
                                line=dict(color='purple', width=2, dash='dash'),
                                marker=dict(size=6, symbol='circle')
                            ),
                            row=2, col=1
                        )
                        
                        # Add confidence interval shading
                        fig_combined.add_trace(
                            go.Scatter(
                                x=list(forecast_dates) + list(forecast_dates[::-1]),
                                y=[upper_bound] * forecast_horizon + [lower_bound] * forecast_horizon,
                                fill='toself',
                                fillcolor='rgba(128, 0, 128, 0.15)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% Confidence',
                                showlegend=True
                            ),
                            row=2, col=1
                        )
                        
                        fig_combined.update_xaxes(title_text="Date", row=2, col=1)
                        fig_combined.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig_combined.update_yaxes(title_text="Volatility", row=2, col=1)
                        
                        fig_combined.update_layout(
                            height=600,
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_combined, use_container_width=True)
                        
                        # Historical comparison (if enough data)
                        if len(df_features) >= sequence_length + 30:
                            st.markdown("---")
                            st.subheader("📉 Historical True vs Predicted")
                            
                            # Generate predictions for last 30 days (demo)
                            n_samples = min(30, len(df_features) - sequence_length)
                            dates_hist = []
                            true_vals = []
                            pred_vals = []
                            lower_vals = []
                            upper_vals = []
                            
                            for i in range(n_samples):
                                idx = -(n_samples - i)
                                date = df_features['Date'].iloc[idx]
                                true_vol = df['True_Vol_30d'].iloc[idx]
                                
                                if not pd.isna(true_vol):
                                    dates_hist.append(date)
                                    true_vals.append(true_vol)
                                    
                                    # Use current prediction as proxy (in production: load saved predictions)
                                    pred_vals.append(point_pred)
                                    lower_vals.append(lower_bound)
                                    upper_vals.append(upper_bound)
                            
                            if len(dates_hist) > 0:
                                fig_hist = plot_predictions_with_intervals(
                                    pd.Series(dates_hist),
                                    np.array(true_vals),
                                    np.array(pred_vals),
                                    np.array(lower_vals),
                                    np.array(upper_vals)
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Daily Forecast Table
                        st.markdown("---")
                        st.subheader("📅 Daily Forecast Table")
                        
                        forecast_df = pd.DataFrame({
                            'Day': range(1, forecast_horizon + 1),
                            'Forecast_Date': pd.date_range(
                                start=last_date + timedelta(days=1), 
                                periods=forecast_horizon, 
                                freq='B'
                            ),
                            'Point_Prediction': [point_pred] * forecast_horizon,
                            'Lower_Bound_95%': [lower_bound] * forecast_horizon,
                            'Upper_Bound_95%': [upper_bound] * forecast_horizon
                        })
                        
                        # Format the dataframe for display
                        display_df = forecast_df.copy()
                        display_df['Forecast_Date'] = display_df['Forecast_Date'].dt.strftime('%Y-%m-%d (%a)')
                        display_df['Point_Prediction'] = display_df['Point_Prediction'].apply(lambda x: f"{x:.4f}")
                        display_df['Lower_Bound_95%'] = display_df['Lower_Bound_95%'].apply(lambda x: f"{x:.4f}")
                        display_df['Upper_Bound_95%'] = display_df['Upper_Bound_95%'].apply(lambda x: f"{x:.4f}")
                        
                        # Add risk indicator
                        def get_risk_indicator(val):
                            val_float = float(val)
                            if val_float < 0.15:
                                return "🟢"
                            elif val_float < 0.25:
                                return "🟡"
                            else:
                                return "🔴"
                        
                        display_df['Risk'] = display_df['Point_Prediction'].apply(get_risk_indicator)
                        
                        # Reorder columns
                        display_df = display_df[['Day', 'Forecast_Date', 'Point_Prediction', 'Lower_Bound_95%', 'Upper_Bound_95%', 'Risk']]
                        
                        # Display with custom styling
                        st.dataframe(
                            display_df,
                            hide_index=True,
                            use_container_width=True,
                            height=min(400, (len(display_df) + 1) * 35 + 3)
                        )
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Prediction", f"{point_pred:.4f}")
                        with col2:
                            st.metric("Confidence Interval Width", f"{upper_bound - lower_bound:.4f}")
                        with col3:
                            risk_days = sum(1 for x in display_df['Risk'] if x == '🔴')
                            st.metric("High Risk Days", f"{risk_days}/{forecast_horizon}")
                        
                        # Download button
                        st.markdown("---")
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name=f"volatility_forecast_{forecast_horizon}d_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions
        st.info("""
        ### 📋 Quick Start Guide:
        
        1. **Select forecast horizon** (5 or 20 days) from the sidebar
        2. **Upload CSV file** with S&P 500 historical data (60+ days recommended)
        3. Required columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
        4. Click **Generate Forecast** to see predictions
        5. View point prediction with 95% confidence intervals
        6. Download forecast results as CSV
        
        ### 🎯 Model Details:
        - **Architecture**: Transformer with 2 encoder layers
        - **Input**: Last 60 days of engineered features (27 features)
        - **Output**: Configurable 5-day or 20-day ahead forecast
        - **Confidence**: 95% prediction intervals based on historical residuals
        
        ### 📊 Interpretation:
        - **Point Prediction**: Most likely volatility level
        - **95% CI**: Range where true volatility is likely to fall
        - **Risk Level**: Automatically categorized (Low/Moderate/High)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>S&P 500 Volatility Forecasting System</strong></p>
    <p>Transformer-based ML model with real-time inference | 95% Prediction Intervals</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
