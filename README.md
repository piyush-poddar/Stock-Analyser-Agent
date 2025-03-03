# Stock-Analyser-Agent

## 📌 Overview

This is a Stock Market Analysis Chatbot which provides real-time stock market analysis and responds to user queries regarding stock prices, trends, and basic market insights. It is powered by a LangGraph agent, which integrates the Gemini API with the yFinance API to provide comprehensive stock analysis. 

Additionally, it generates line and candlestick charts using Matplotlib to visualize stock trends over time. To enhance user experience, the chatbot is built with Streamlit, providing an interactive and user-friendly interface.

## ✨ Features

- 📈 **Real-time Stock Data**: Fetches the latest stock prices using yFinance.

- 🤖 **AI-Powered Analysis**: Uses the Gemini API for intelligent stock-related insights.

- 💬 **Chatbot Interface**: Built with Streamlit for an interactive experience.

- 🔍 **Stock Trend Analysis**: Provides insights into stock movement over time.

- 📊 **Graphical Representation**: Generates line and candlestick charts for trend visualization.

## 🛠️ Tech Stack

- **Python** - Backend processing

- **LangGraph & LangChain** - For Developing the AI Agent with Stock Analysis Tool

- **yFinance** - Fetching real-time stock market data

- **Gemini API** - LLM Powering the Agent

- **Matplotlib** - For visualizing stock trends

- **Streamlit** - For chatbot UI

## 🚀 Installation

1. Clone the repository:
```Bash
git clone https://github.com/piyush-poddar/Stock-Analyser-Agent.git

cd Stock-Analyser-Agent
```

2. Install required dependencies:
```Bash
pip install -r requirements.txt
```

3. Run the Streamlit chatbot:
```Bash
streamlit run app.py
```

## 📌 Usage

- Enter queries like:

    - "Analyse Tesla Stock for me"
    - "Analyse Apple for 3 months and give me a candle chart"
    - "Compare Google and Microsoft Stocks for the past year"

- The bot responds with stock prices, analysis, and visual charts depending on the query.

## 🔮 Future Enhancements

- 📜 Sentiment analysis for stock predictions.

- 📡 Live Portfolio Tracking

- 📊 Enhanced Visualization & UI

- 🤖 AI-based Stock Price Predictions

## 🤝 Contributing

1. Fork the repo and create a new branch.

2. Make improvements and test thoroughly.

3. Submit a pull request with a detailed description.

### Made with ❤️ by Piyush Poddar
