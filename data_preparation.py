import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os

class DataPreparation:
    def __init__(self, symbols=['SPY'], 
                 start_date='2018-01-01',
                 end_date='2024-11-24',
                 window_size=15):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        
    def create_labels(self, prices, threshold=0.02):
        """Create trading signals based on future returns"""
        labels = np.zeros(len(prices))
        future_returns = prices.pct_change(5).shift(-5).fillna(0) # 5-day future returns
        
        labels[future_returns > threshold] = 1  # Buy
        labels[future_returns < -threshold] = 2  # Sell
        # 0 remains as Hold
        
        return labels[:-5]  # Remove last 5 days where we don't have future returns
        
    def prepare_dataset(self, save_path='trading_dataset'):
        """Prepare complete dataset with images and labels"""
        os.makedirs(save_path, exist_ok=True)
        for category in ['train', 'validation', 'test']:
            os.makedirs(f"{save_path}/{category}/buy", exist_ok=True)
            os.makedirs(f"{save_path}/{category}/hold", exist_ok=True)
            os.makedirs(f"{save_path}/{category}/sell", exist_ok=True)
        
        all_data = self.fetch_data()
        dataset_info = {'train': [], 'validation': [], 'test': []}
        
        for symbol in self.symbols:
            print(f"Processing {symbol}...")
            data = all_data[symbol]
            
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            if indicators is None:
                continue
                
            # Create labels
            labels = self.create_labels(data['Close'])
            
            # Create sliding windows of data
            for i in range(len(indicators) - self.window_size + 1):
                if i + self.window_size > len(labels):
                    continue
                    
                window_data = indicators.iloc[i:i+self.window_size]
                label = labels[i + self.window_size - 1]
                
                # Skip if label is NaN
                if np.isnan(label):
                    continue
                
                # Create image
                image_data = self.create_image(window_data)
                if image_data is None:
                    continue
                
                # Determine split (80% train, 10% validation, 10% test)
                split = np.random.random()
                if split < 0.8:
                    category = 'train'
                elif split < 0.9:
                    category = 'validation'
                else:
                    category = 'test'
                
                # Save image
                label_name = ['hold', 'buy', 'sell'][int(label)]
                image_path = f"{save_path}/{category}/{label_name}/{symbol}_{i}.png"
                Image.fromarray(image_data).save(image_path)
                
                # Store metadata
                dataset_info[category].append({
                    'symbol': symbol,
                    'date': data.index[i + self.window_size - 1].strftime('%Y-%m-%d'),
                    'label': label_name,
                    'path': image_path
                })
        
        # Save dataset info
        for category, info in dataset_info.items():
            pd.DataFrame(info).to_csv(f"{save_path}/{category}_info.csv", index=False)
            print(f"{category} set size: {len(info)}")
            
        return dataset_info

    def fetch_data(self):
        """Fetch data for all symbols"""
        all_data = {}
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            stock = yf.download(symbol, self.start_date, self.end_date)
            all_data[symbol] = stock
        return all_data
    
    def calculate_indicators(self, data):
        """Calculate technical indicators with smaller window sizes"""
        df = pd.DataFrame()
        
        try:
            # Use smaller window sizes (5-10 periods)
            window_sizes = [5, 6, 7, 8, 9, 10]
            
            # Calculate indicators for different window sizes
            for window in window_sizes:
                # Price-based
                df[f'sma_{window}'] = data['Close'].rolling(window=window).mean()
                df[f'ema_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
                
                # Momentum
                df[f'roc_{window}'] = data['Close'].pct_change(periods=window)
                
                # Volatility
                df[f'std_{window}'] = data['Close'].rolling(window=window).std()
                
                # Volume
                df[f'vol_sma_{window}'] = data['Volume'].rolling(window=window).mean()
                
                # Additional
                df[f'high_low_{window}'] = (data['High'] - data['Low']).rolling(window=window).mean()
            
            # Drop any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure we have exactly 12 columns for the image
            columns_to_use = list(df.columns)[:12]
            
            return df[columns_to_use]
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None


    # def calculate_indicators(self, data):
    #     """Calculate technical indicators with proper window size validation"""
    #     if len(data) < self.window_size:
    #         print(f"Warning: Data length ({len(data)}) is smaller than window size ({self.window_size})")
    #         return None
            
    #     df = pd.DataFrame()
        
    #     try:
    #         # Use dynamic window sizes based on available data
    #         max_window = min(21, len(data) // 3)  # Ensure window isn't too large
    #         window_sizes = [w for w in [5, 8, 13, 21] if w <= max_window]
            
    #         for window in window_sizes:
    #             # Trend Indicators
    #             df[f'sma_{window}'] = data['Close'].rolling(window=window, min_periods=1).mean()
    #             df[f'ema_{window}'] = data['Close'].ewm(span=window, adjust=False, min_periods=1).mean()
                
    #             # Momentum Indicators
    #             df[f'rsi_{window}'] = ta.momentum.rsi(data['Close'], window=window, fillna=True)
    #             df[f'roc_{window}'] = data['Close'].pct_change(periods=window).fillna(0)
                
    #             # Volatility Indicators
    #             df[f'bbands_upper_{window}'] = ta.volatility.bollinger_hband(
    #                 data['Close'], window=window, fillna=True)
    #             df[f'bbands_lower_{window}'] = ta.volatility.bollinger_lband(
    #                 data['Close'], window=window, fillna=True)
                
    #             # Volume Indicators
    #             df[f'volume_sma_{window}'] = data['Volume'].rolling(
    #                 window=window, min_periods=1).mean()
            
    #         # Additional Indicators
    #         df['close_to_sma'] = data['Close'] / df['sma_5'] - 1
    #         df['volume_price_ratio'] = data['Volume'] / data['Close']
    #         df['high_low_ratio'] = data['High'] / data['Low']
    #         df['daily_return'] = data['Close'].pct_change().fillna(0)
            
    #         # Handle missing values
    #         df = df.fillna(method='ffill').fillna(0)
            
    #         # Select 12 most important features
    #         important_features = [
    #             'sma_5', 'ema_5', 'rsi_5', 'roc_5',
    #             'bbands_upper_5', 'bbands_lower_5',
    #             'volume_sma_5', 'close_to_sma',
    #             'volume_price_ratio', 'high_low_ratio',
    #             'daily_return', 'volume_sma_8'
    #         ]
            
    #         # Ensure we have exactly 12 features
    #         available_features = [col for col in important_features if col in df.columns]
    #         if len(available_features) < 12:
    #             missing_count = 12 - len(available_features)
    #             # Add duplicate features if needed
    #             for i in range(missing_count):
    #                 df[f'extra_feature_{i}'] = df[available_features[0]]
    #                 available_features.append(f'extra_feature_{i}')
            
    #         return df[available_features[:12]]  # Ensure exactly 12 features
            
    #     except Exception as e:
    #         print(f"Error calculating indicators: {str(e)}")
    #         return None
        
    def create_image(self, indicators_data):
        """Convert indicators to image format"""
        if len(indicators_data) < self.window_size:
            return None
            
        # Ensure we have exactly 12 indicators
        if len(indicators_data.columns) != 12:
            print(f"Error: Expected 12 indicators, got {len(indicators_data.columns)}")
            return None
            
        # Select last window_size rows
        window_data = indicators_data.iloc[-self.window_size:].values
        
        try:
            # Normalize data
            normalized_data = self.scaler.fit_transform(window_data)
            
            # Reshape to 12 x window_size (12 indicators)
            image_data = normalized_data.T
            
            # Verify shape
            if image_data.shape != (12, self.window_size):
                print(f"Error: Wrong shape: {image_data.shape}")
                return None
            
            # Convert to uint8 for image
            image_data = (image_data * 255).astype(np.uint8)
            return image_data
            
        except Exception as e:
            print(f"Error in create_image: {e}")
            return None
