# SMCI Stock Price Predictor

A deep learning project that uses LSTM neural networks to forecast the future prices of SMCI (Super Micro Computer Inc.) stock, leveraging historical stock data and technical indicators. Designed for financial analysis and time-series forecasting using Python and modern ML libraries.

## Description

This project implements an end-to-end LSTM (Long Short-Term Memory) neural network to predict the stock prices for SMCI, using historical market data enriched with commonly used technical indicators. The workflow covers data fetching, feature engineering, model training, prediction, and optional visualization. It helps investors and ML enthusiasts understand and forecast stock price movements.

## Features

- Download historical price data for SMCI from Yahoo Finance.
- Compute technical indicators for enhanced feature sets.
- Train LSTM-based deep learning models for time-series forecasting.
- Make terminal-based stock price predictions.
- Support for GPU acceleration on Apple Silicon (M1/M2).

## Installation

1.Clone the reposiroty :
```bash
git clone https://github.com/Aamir2424/smci-stock-predictor.git
cd smci-stock-predictor
```
Install dependencies:
```bash
pip install yfinance pandas numpy scikit-learn ta matplotlib seaborn tensorflow
```
# Run the script
```bash
python stock_analysis.py
```

## Technologies / Built With

- Python 3.x
- TensorFlow / Keras
- pandas, numpy
- yfinance (Yahoo Finance data downloader)
- ta (technical indicators)
- scikit-learn
- matplotlib (optional for plotting)
