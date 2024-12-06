import yfinance as yf
import pandas as pd
from data_preparation import DataPreparation
from model_training import CNNTrading
from backtesting import TradingBacktester

# main.py - Improved pipeline with data validation
def run_pipeline(symbols=['AAPL', 'MSFT'], 
                start_date='2023-01-01',
                end_date='2024-11-25',
                initial_capital=100000):
    """Run the trading pipeline with improved data validation"""
    try:
        results = {}
        
        # 1. Initial data validation
        print("\n=== Data Validation ===")
        valid_data = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, start_date, end_date, progress=False)
                if len(data) > 0:
                    valid_data[symbol] = data
                    print(f"✓ {symbol}: {len(data)} days of data loaded")
                else:
                    print(f"✗ {symbol}: No data available")
            except Exception as e:
                print(f"✗ {symbol}: Error loading data - {str(e)}")
        
        if not valid_data:
            print("Error: No valid data available for any symbols")
            return {}
            
        # 2. Data Preparation
        print("\n=== Data Preparation ===")
        data_prep = DataPreparation(
            symbols=list(valid_data.keys()),
            start_date=start_date,
            end_date=end_date
        )
        
        print("Creating dataset...")
        dataset_info = data_prep.prepare_dataset(save_path='trading_dataset')
        
        # Validate dataset creation
        total_samples = sum(len(info) for info in dataset_info.values())
        if total_samples == 0:
            print("Error: No training samples generated")
            return {}
            
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(dataset_info['train'])}")
        print(f"Validation samples: {len(dataset_info['validation'])}")
        print(f"Test samples: {len(dataset_info['test'])}")
        
        # 3. Model Training/Loading
        print("\n=== Model Preparation ===")
        trader = CNNTrading(dataset_path='trading_dataset')
        if total_samples > 0:
            print("Training new model...")
            model = trader.build_model()
            model.save('trading_model.h5')
        else:
            print("Loading existing model...")
            model = tf.keras.models.load_model('trading_model.h5')
        
        # 4. Backtesting
        print("\n=== Running Backtests ===")
        backtester = TradingBacktester(
            model_path='trading_model.h5',
            initial_capital=initial_capital
        )
        
        for symbol in valid_data.keys():
            try:
                print(f"\nProcessing {symbol}...")
                data = valid_data[symbol]
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
                    
                    # Print summary statistics
                    print(f"\n{symbol} Results:")
                    print(f"Total Return: {backtest_results['metrics']['total_return']*100:.2f}%")
                    print(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']*100:.2f}%")
                    print(f"Number of Trades: {backtest_results['metrics']['num_trades']}")
                    
                    if len(backtest_results['trades']) > 0:
                        trades_df = backtest_results['trades']
                        buys = trades_df[trades_df['action'] == 'buy']
                        sells = trades_df[trades_df['action'] == 'sell']
                        print(f"\nTrade Summary:")
                        print(f"Total Buys: {len(buys)}")
                        print(f"Total Sells: {len(sells)}")
                        if len(buys) > 0:
                            print(f"Average Buy Price: ${buys['price'].mean():.2f}")
                        if len(sells) > 0:
                            print(f"Average Sell Price: ${sells['price'].mean():.2f}")
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                results[symbol] = {
                    'metrics': None,
                    'data': None,
                    'trades': pd.DataFrame()
                }
        
        return results
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return {}

# def run_pipeline(symbols=['AAPL', 'MSFT'], 
#                 start_date='2023-01-01',
#                 end_date='2023-12-31',
#                 initial_capital=100000):
#     """Run the trading pipeline and return formatted results"""
#     try:
#         # Initialize results dictionary
#         results = {}
        
#         # Create data preparation instance
#         data_prep = DataPreparation(
#             symbols=symbols,
#             start_date=start_date,
#             end_date=end_date
#         )
        
#         # Prepare dataset
#         dataset_info = data_prep.prepare_dataset(save_path='trading_dataset')
        
#         # Create and train model
#         trader = CNNTrading(dataset_path='trading_dataset')
#         model = trader.build_model()
        
#         # Save model
#         model.save('trading_model.h5')
        
#         # Run backtest
#         backtester = TradingBacktester(
#             model_path='trading_model.h5',
#             initial_capital=initial_capital
#         )
        
#         for symbol in symbols:
#             try:
#                 # Download complete OHLCV data
#                 data = yf.download(symbol, start_date, end_date)
#                 backtest_results = backtester.backtest(symbol, start_date, end_date)
                
#                 if backtest_results is not None:
#                     results[symbol] = {
#                         'metrics': backtest_results['metrics'],
#                         'data': pd.concat([
#                             data,
#                             pd.Series(backtest_results['portfolio_value'], name='portfolio_value')
#                         ], axis=1),
#                         'trades': backtest_results['trades']
#                     }
#                     print(f"Successfully processed {symbol}")
                    
#             except Exception as e:
#                 print(f"Error processing {symbol}: {str(e)}")
#                 results[symbol] = {
#                     'metrics': None,
#                     'data': None,
#                     'trades': pd.DataFrame()
#                 }
        
#         return results
        
#     except Exception as e:
#         print(f"Error in pipeline: {str(e)}")
#         return {}

# if __name__ == "__main__":
#     # Test run
#     results = run_pipeline(['AAPL'], '2023-01-01', '2023-12-31')
#     print("Results available for symbols:", list(results.keys()))
