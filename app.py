# 1. app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from data_preparation import DataPreparation
from model_training import CNNTrading
from backtesting import TradingBacktester

def get_valid_dates():
    """Get valid date range for trading"""
    end_date = datetime.now() - timedelta(days=1)
    while end_date.weekday() > 4:
        end_date -= timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    while start_date.weekday() > 4:
        start_date -= timedelta(days=1)
    return start_date.date(), end_date.date()

def initialize_page():
    st.set_page_config(
        page_title="CNN Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .big-font {font-size:24px !important; font-weight: bold;}
        .metric-container {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;}
        .profit {color: #4CAF50;}
        .loss {color: #f44336;}
        </style>
    """, unsafe_allow_html=True)

def create_chart(data, trades, title):
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Portfolio Value')
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        if trades is not None and not trades.empty:
            buys = trades[trades['action'] == 'buy']
            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys['date'],
                        y=buys['price'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=15, color='green'),
                        name='Buy'
                    ),
                    row=1, col=1
                )

            sells = trades[trades['action'] == 'sell']
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells['date'],
                        y=sells['price'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=15, color='red'),
                        name='Sell'
                    ),
                    row=1, col=1
                )

        if 'portfolio_value' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['portfolio_value'],
                    name='Portfolio Value',
                    line=dict(color='#2196F3', width=2)
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )

        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def show_metrics(metrics):
    try:
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**Total Return**")
            value = metrics['total_return'] * 100
            color = 'profit' if value > 0 else 'loss'
            st.markdown(f'<p class="big-font {color}">{value:.2f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[1]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**Sharpe Ratio**")
            st.markdown(f'<p class="big-font">{metrics["sharpe_ratio"]:.2f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[2]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**Max Drawdown**")
            st.markdown(f'<p class="big-font loss">{metrics["max_drawdown"]*100:.2f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[3]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**Number of Trades**")
            st.markdown(f'<p class="big-font">{metrics["num_trades"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def run_pipeline(symbols, start_date, end_date, initial_capital):
    try:
        results = {}
        
        # Prepare data
        data_prep = DataPreparation(symbols=symbols,
                                  start_date=start_date,
                                  end_date=end_date)
        
        # Create dataset
        dataset_info = data_prep.prepare_dataset(save_path='trading_dataset')
        
        # Train/load model
        trader = CNNTrading(dataset_path='trading_dataset')
        if sum(len(info) for info in dataset_info.values()) > 0:
            model = trader.build_model()
            model.save('trading_model.h5')
        
        # Run backtests
        backtester = TradingBacktester(
            model_path='trading_model.h5',
            initial_capital=initial_capital
        )
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, start_date, end_date, progress=False)
                backtest_results = backtester.backtest(symbol, start_date, end_date)
                
                if backtest_results is not None:
                    results[symbol] = {
                        'metrics': backtest_results['metrics'],
                        'data': pd.concat([
                            data,
                            pd.Series(backtest_results['portfolio_value'], 
                                    name='portfolio_value')
                        ], axis=1),
                        'trades': backtest_results['trades']
                    }
                    st.success(f"Successfully processed {symbol}")
                    
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
                results[symbol] = {
                    'metrics': None,
                    'data': None,
                    'trades': pd.DataFrame()
                }
        
        return results
        
    except Exception as e:
        st.error(f"Error in pipeline: {str(e)}")
        return None

def app():
    initialize_page()
    
    st.title('CNN Trading System')
    st.markdown('---')

    default_start, default_end = get_valid_dates()

    st.sidebar.header('Settings')
    
    symbols = st.sidebar.multiselect(
        'Select Symbols',
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        default=['AAPL', 'MSFT']
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            'Start Date',
            value=default_start,
            max_value=default_end
        )
    with col2:
        end_date = st.date_input(
            'End Date',
            value=default_end,
            min_value=start_date,
            max_value=default_end
        )
    
    initial_capital = st.sidebar.number_input(
        'Initial Capital ($)',
        value=100000,
        step=10000,
        min_value=10000
    )
    
    if st.sidebar.button('Run Backtest', use_container_width=True):
        if not symbols:
            st.warning("Please select at least one symbol")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner('Running backtest...'):
            try:
                # Check data availability
                status_text.text("Checking data availability...")
                progress_bar.progress(10)
                
                valid_data = {}
                for symbol in symbols:
                    try:
                        data = yf.download(symbol, start_date, end_date, progress=False)
                        if not data.empty:
                            valid_data[symbol] = data
                            st.success(f"✓ {symbol}: {len(data)} days of data loaded")
                        else:
                            st.warning(f"✗ {symbol}: No data available")
                    except Exception as e:
                        st.warning(f"Error loading {symbol}: {str(e)}")
                
                if not valid_data:
                    st.error("No valid data available for selected symbols")
                    return
                
                progress_bar.progress(30)
                status_text.text("Running analysis...")
                
                results = run_pipeline(
                    symbols=list(valid_data.keys()),
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    initial_capital=initial_capital
                )
                
                progress_bar.progress(70)
                status_text.text("Generating results...")
                
                if results:
                    for symbol in valid_data.keys():
                        if symbol in results:
                            st.header(f'{symbol} Analysis')
                            
                            if 'metrics' in results[symbol] and results[symbol]['metrics']:
                                show_metrics(results[symbol]['metrics'])
                                
                                # Trade summary
                                if 'trades' in results[symbol] and not results[symbol]['trades'].empty:
                                    trades_df = results[symbol]['trades']
                                    buys = trades_df[trades_df['action'] == 'buy']
                                    sells = trades_df[trades_df['action'] == 'sell']
                                    
                                    cols = st.columns(4)
                                    cols[0].metric("Buy Trades", len(buys))
                                    cols[1].metric("Sell Trades", len(sells))
                                    if not buys.empty:
                                        cols[2].metric("Avg Buy Price", f"${buys['price'].mean():.2f}")
                                    if not sells.empty:
                                        cols[3].metric("Avg Sell Price", f"${sells['price'].mean():.2f}")
                            
                            if 'data' in results[symbol] and 'trades' in results[symbol]:
                                fig = create_chart(
                                    results[symbol]['data'],
                                    results[symbol]['trades'],
                                    f'{symbol} Trading Activity'
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            if 'trades' in results[symbol] and not results[symbol]['trades'].empty:
                                st.subheader('Trade History')
                                st.dataframe(
                                    results[symbol]['trades'].style.format({
                                        'price': '${:.2f}',
                                        'value': '${:.2f}'
                                    })
                                )
                            
                            st.markdown('---')
                        else:
                            st.warning(f"No results available for {symbol}")
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                else:
                    st.error("No results returned from backtest")
                
            except Exception as e:
                st.error(f"Error during backtest: {str(e)}")
                st.exception(e)
            
            finally:
                progress_bar.empty()
                status_text.empty()

if __name__ == "__main__":
    app()