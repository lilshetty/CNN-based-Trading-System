import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from data_preparation import DataPreparation
import matplotlib.pyplot as plt

class TradingBacktester:
    def __init__(self, model_path='trading_model.h5', 
                 initial_capital=100000,
                 position_size=0.1):
        self.model = tf.keras.models.load_model(model_path)
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.data_prep = DataPreparation()
        
    def prepare_single_prediction(self, data_window):
        """Prepare single window of data for prediction"""
        indicators = self.data_prep.calculate_indicators(data_window)
        if indicators is None:
            return None
        
        image_data = self.data_prep.create_image(indicators)
        return np.expand_dims(image_data, axis=(0, -1))
    
    def backtest(self, symbol, start_date, end_date):
        """Run backtest on historical data"""
        try:
            # Fetch data
            self.data = yf.download(symbol, start_date, end_date)
            
            # Initialize results tracking
            positions = pd.Series(index=self.data.index, data=0)
            portfolio_value = pd.Series(index=self.data.index, data=self.initial_capital)
            cash = self.initial_capital
            shares = 0
            trades = []
            
            # Trading loop
            window_size = 15
            for i in range(window_size, len(self.data)):
                window_data = self.data.iloc[i-window_size:i]
                
                X = self.prepare_single_prediction(window_data)
                if X is not None:
                    prediction = self.model.predict(X, verbose=0)
                    signal = np.argmax(prediction[0])
                    
                    current_price = self.data['Close'].iloc[i]
                    date = self.data.index[i]
                    
                    # Trading logic
                    if signal == 1 and shares == 0:  # Buy
                        shares = int((cash * self.position_size) // current_price)
                        if shares > 0:
                            cash -= shares * current_price
                            trades.append({
                                'date': date,
                                'action': 'buy',
                                'price': current_price,
                                'shares': shares,
                                'value': shares * current_price
                            })
                            print(f"BUY: {shares} shares at {current_price}")
                    
                    elif signal == 2 and shares > 0:  # Sell
                        cash += shares * current_price
                        trades.append({
                            'date': date,
                            'action': 'sell',
                            'price': current_price,
                            'shares': shares,
                            'value': shares * current_price
                        })
                        print(f"SELL: {shares} shares at {current_price}")
                        shares = 0
                
                positions.iloc[i] = shares
                portfolio_value.iloc[i] = cash + (shares * current_price)
            
            # Calculate metrics
            returns = portfolio_value.pct_change().dropna()
            metrics = {
                'total_return': (portfolio_value.iloc[-1] - self.initial_capital) / self.initial_capital,
                'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() != 0 else 0,
                'max_drawdown': (portfolio_value / portfolio_value.cummax() - 1).min(),
                'num_trades': len(trades)
            }
            
            return {
                'metrics': metrics,
                'portfolio_value': portfolio_value,
                'positions': positions,
                'trades': pd.DataFrame(trades)
            }
            
        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            return None
        
    def visualize_results(self, results, symbol):
        """Visualize backtest results"""
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot portfolio value
        ax1.plot(results['portfolio_value'], label='Portfolio Value')
        ax1.set_title(f'Portfolio Value Over Time - {symbol}')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        
        # Plot drawdown
        drawdown = (results['portfolio_value'] / results['portfolio_value'].cummax() - 1)
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        
        # Plot positions over time
        ax3.plot(results['positions'], label='Position Size')
        ax3.set_title('Position Size Over Time')
        ax3.set_ylabel('Number of Shares')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'backtest_results_{symbol}.png')
        plt.close()
        
        # Print metrics
        print("\nBacktest Results:")
        print(f"Total Return: {results['metrics']['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
        print(f"Number of Trades: {results['metrics']['num_trades']}")
    
    def validate_trade_data(self, symbol, start_date, end_date):
        """Validate data before running backtest"""
        try:
            data = yf.download(symbol, start_date, end_date, progress=False)
            if data.empty:
                print(f"No data available for {symbol}")
                return None
                
            missing_data = data.isnull().sum().sum()
            if missing_data > 0:
                print(f"Warning: {missing_data} missing values found")
                data = data.fillna(method='ffill').fillna(method='bfill')
                
            if len(data) < self.window_size:
                print(f"Insufficient data points for {symbol}")
                return None
                
            return data
            
        except Exception as e:
            print(f"Error validating data: {str(e)}")
            return None


if __name__ == "__main__":
    # Run backtest
    backtester = TradingBacktester()
    
    # Test on multiple symbols
    symbols = ['SPY', 'QQQ', 'AAPL']
    start_date = '2023-01-01'
    end_date = '2024-11-25'
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        results = backtester.backtest(symbol, start_date, end_date)
        backtester.visualize_results(results, symbol)
