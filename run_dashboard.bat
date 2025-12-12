@echo off
echo ========================================
echo Stock Volatility Forecasting Dashboard
echo ========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo ========================================
echo Starting Streamlit Dashboard...
echo ========================================
echo.
streamlit run app.py
