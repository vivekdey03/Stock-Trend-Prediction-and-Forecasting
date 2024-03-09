import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from stocknews import StockNews
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import requests


# Load CSS file
def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

@st.cache_data
def get_data():
    path= 'stock.csv'
    return pd.read_csv(path, low_memory=False)

st.title('Stock Trend Prediction & Forecasting')
ticker = st.sidebar.text_input('Stock Picker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Data For Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)
fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)
st.plotly_chart(fig)

#Sub Headings
pricing_data, fundamental_data, charts, forecasting, news = st.tabs(["Pricing Data", "Fundamental Data", "Charts", "Forecasting", "Top 10 News"])

# Code for Pricing Data Information
with pricing_data:
    st.header("Price Movements")
    data2 = data
    data2['% Change'] = data['Adj Close'].shift(1)
    data2.dropna(inplace = True)
    if not data2.empty:
        st.write(data2)
        annual_return = data2['% Change'].mean()*252*100
        st.write("Annual Return is ",annual_return,"%")
        stdev = np.std(data2['% Change'])*np.sqrt(252)
        st.write("Standerd Deviation is ",stdev*100,"%")
        st.write("Risk Adj. Return is ",annual_return/(stdev*100),"%")
    else:
        st.write("No data available for the selected stock")

# Code for Chart Information
with charts:
    #Open price vs Close price
    st.write("###")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))

    fig.update_layout(title_text='Open and Close Prices Over Time', xaxis_title='Date', yaxis_title='Price',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # 100-day moving average chart

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    data['SMA100'] = data['Close'].rolling(window=100).mean()
    fig1.add_trace(go.Scatter(x=data.index, y=data['SMA100'], mode='lines', name='SMA100'))

    fig1.update_layout(title_text='Close and 100-MA Over Time', xaxis_title='Date', yaxis_title='Price',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # 200-day moving average chart

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    fig2.add_trace(go.Scatter(x=data.index, y=data['SMA200'], mode='lines', name='SMA200'))

    fig2.update_layout(title_text='Close and 200-MA Over Time', xaxis_title='Date', yaxis_title='Price',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

    # 100-day vs 200-day moving average chart vs Close price

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    data['SMA100'] = data['Close'].rolling(window=100).mean()
    fig3.add_trace(go.Scatter(x=data.index, y=data['SMA100'], mode='lines', name='SMA100'))
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    fig3.add_trace(go.Scatter(x=data.index, y=data['SMA200'], mode='lines', name='SMA200'))

    fig3.update_layout(title_text='Close and 100-MA vs 200-MA Over Time', xaxis_title='Date', yaxis_title='Price',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

# Code for Forecasting Information
with forecasting:
    st.header("Stock Price Forecasting")

    # Load the data for forecasting
    df_prophet = data[['Adj Close']].reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Adj Close': 'y'})

    # User input for forecasting period
    forecast_years = st.slider("Select the number of years for forecasting:", 1, 10, 1)

    # Forecasting with Prophet
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=365 * forecast_years)  # Forecast for the selected number of years
    forecast = model.predict(future)

    # Actual vs Predicted Prices
    actual_prices = data[['Adj Close']].reset_index()
    actual_prices = df_prophet.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    actual_vs_predicted = pd.merge(actual_prices, forecast[['ds', 'yhat']], on='ds', how='inner')

    # Calculate Accuracy Metrics
    mse = ((actual_vs_predicted['y'] - actual_vs_predicted['yhat']) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(actual_vs_predicted['y'] - actual_vs_predicted['yhat']).mean()

    # Display Accuracy Metrics
    st.subheader("Accuracy Metrics")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")


    # Plotting the forecast
    st.subheader("Actual vs Predicted Stock Prices Forecast")
    fig_prophet = plot_plotly(model, forecast)
    st.plotly_chart(fig_prophet)

    # Display components of the forecast
    st.subheader("Forecast Components")
    fig_components = model.plot_components(forecast)
    st.write(fig_components)

# Code for Fundamental Data Information
key = '8LKLIWFD66VHX354'  #API Key
with fundamental_data:
    
    fd = FundamentalData(key,output_format = 'pandas')

    #Balance Sheet
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)

    #Income Statement
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    #Cash Flow Statement
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)

# Code for Financial News Information
with news:
    #Top 10 Financial News
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news = False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'# {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        titile_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {titile_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

#End of Program