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