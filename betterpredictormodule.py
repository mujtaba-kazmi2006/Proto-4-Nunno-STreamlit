import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingAnalyzer:
    def __init__(self):
        self.confluence_threshold = 3  # Minimum confluences for strong signals
    
    def fetch_binance_ohlcv(self, symbol="BTCUSDT", interval="15m", limit=1000):
        """Fetch OHLCV data from Binance with error handling"""
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote Asset Volume", "Number of Trades",
                "Taker Buy Base", "Taker Buy Quote", "Ignore"
            ])
            
            # Convert timestamps and prices
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].astype({
                "Open": float, "High": float, "Low": float, "Close": float, "Volume": float
            })
            df.set_index('Open Time', inplace=True)
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch data: {str(e)}")
    
    def add_comprehensive_indicators(self, df):
        """Add comprehensive technical indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Momentum Indicators
        df['RSI_14'] = RSIIndicator(close, window=14).rsi()
        df['RSI_21'] = RSIIndicator(close, window=21).rsi()
        df['Stoch_K'] = StochasticOscillator(high, low, close, window=14).stoch()
        df['Stoch_D'] = StochasticOscillator(high, low, close, window=14).stoch_signal()
        df['Williams_R'] = WilliamsRIndicator(high, low, close).williams_r()
        
        # Trend Indicators
        df['EMA_9'] = EMAIndicator(close, window=9).ema_indicator()
        df['EMA_21'] = EMAIndicator(close, window=21).ema_indicator()
        df['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        
        # MACD
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # ADX and DI
        adx = ADXIndicator(high, low, close)
        df['ADX'] = adx.adx()
        df['DI_Plus'] = adx.adx_pos()
        df['DI_Minus'] = adx.adx_neg()
        
        # Volatility Indicators
        bb = BollingerBands(close, window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Keltner Channels
        kc = KeltnerChannel(high, low, close)
        df['KC_Upper'] = kc.keltner_channel_hband()
        df['KC_Lower'] = kc.keltner_channel_lband()
        df['KC_Middle'] = kc.keltner_channel_mband()
        
        # ATR and volatility measures
        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
        df['ATR_Percent'] = (df['ATR'] / close) * 100
        
        # Volume Indicators  
        df['Volume_SMA'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_SMA']
        df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df['CMF'] = ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow()
        
        # Price Action
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open'] * 100
        df['Upper_Wick'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open'] * 100
        df['Lower_Wick'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open'] * 100
        df['Total_Range'] = (df['High'] - df['Low']) / df['Open'] * 100
        
        # Support/Resistance levels (simplified)
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        
        # Rate of Change
        df['ROC_5'] = ((close / close.shift(5)) - 1) * 100
        df['ROC_14'] = ((close / close.shift(14)) - 1) * 100
        
        df.dropna(inplace=True)
        return df
    
    def analyze_momentum_confluence(self, row):
        """Analyze momentum indicators for confluences"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # RSI Analysis
        if row['RSI_14'] < 30:
            confluences['bullish'].append({
                'indicator': 'RSI (14)',
                'condition': f"Oversold at {row['RSI_14']:.1f}",
                'implication': "Potential bounce or reversal setup. Watch for bullish divergence or break above 30.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['RSI_14'] > 70:
            confluences['bearish'].append({
                'indicator': 'RSI (14)',
                'condition': f"Overbought at {row['RSI_14']:.1f}",
                'implication': "Potential pullback or distribution. Watch for bearish divergence or break below 70.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif 45 <= row['RSI_14'] <= 55:
            confluences['neutral'].append({
                'indicator': 'RSI (14)',
                'condition': f"Neutral at {row['RSI_14']:.1f}",
                'implication': "Balanced momentum. Look for directional break above 55 or below 45.",
                'strength': 'Low',
                'timeframe': 'Short-term'
            })
        
        # Stochastic Analysis
        if row['Stoch_K'] < 20 and row['Stoch_D'] < 20:
            confluences['bullish'].append({
                'indicator': 'Stochastic',
                'condition': f"Both %K ({row['Stoch_K']:.1f}) and %D ({row['Stoch_D']:.1f}) oversold",
                'implication': "Strong oversold condition. Potential reversal when %K crosses above %D.",
                'strength': 'Strong' if row['Stoch_K'] > row['Stoch_D'] else 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['Stoch_K'] > 80 and row['Stoch_D'] > 80:
            confluences['bearish'].append({
                'indicator': 'Stochastic',
                'condition': f"Both %K ({row['Stoch_K']:.1f}) and %D ({row['Stoch_D']:.1f}) overbought",
                'implication': "Strong overbought condition. Potential reversal when %K crosses below %D.",
                'strength': 'Strong' if row['Stoch_K'] < row['Stoch_D'] else 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Williams %R Analysis
        if row['Williams_R'] < -80:
            confluences['bullish'].append({
                'indicator': 'Williams %R',
                'condition': f"Oversold at {row['Williams_R']:.1f}",
                'implication': "Potential buying opportunity. Watch for move above -80 for confirmation.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['Williams_R'] > -20:
            confluences['bearish'].append({
                'indicator': 'Williams %R',
                'condition': f"Overbought at {row['Williams_R']:.1f}",
                'implication': "Potential selling pressure. Watch for move below -20 for confirmation.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        return confluences
    
    def analyze_trend_confluence(self, row):
        """Analyze trend indicators for confluences"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # EMA Alignment
        ema_alignment = "bullish" if row['EMA_9'] > row['EMA_21'] > row['EMA_50'] else "bearish" if row['EMA_9'] < row['EMA_21'] < row['EMA_50'] else "mixed"
        
        if ema_alignment == "bullish":
            confluences['bullish'].append({
                'indicator': 'EMA Alignment',
                'condition': "EMA 9 > EMA 21 > EMA 50",
                'implication': "Strong bullish trend structure. Expect continuation with pullbacks to EMAs as support.",
                'strength': 'Strong',
                'timeframe': 'Medium-term'
            })
        elif ema_alignment == "bearish":
            confluences['bearish'].append({
                'indicator': 'EMA Alignment',
                'condition': "EMA 9 < EMA 21 < EMA 50",
                'implication': "Strong bearish trend structure. Expect continuation with rallies to EMAs as resistance.",
                'strength': 'Strong',
                'timeframe': 'Medium-term'
            })
        
        # Price vs EMAs
        if row['Close'] > row['EMA_21']:
            confluences['bullish'].append({
                'indicator': 'Price vs EMA 21',
                'condition': f"Price {((row['Close']/row['EMA_21']-1)*100):+.2f}% above EMA 21",
                'implication': "Bullish bias maintained. EMA 21 likely to act as dynamic support.",
                'strength': 'Medium',
                'timeframe': 'Short to Medium-term'
            })
        else:
            confluences['bearish'].append({
                'indicator': 'Price vs EMA 21',
                'condition': f"Price {((row['Close']/row['EMA_21']-1)*100):+.2f}% below EMA 21",
                'implication': "Bearish bias maintained. EMA 21 likely to act as dynamic resistance.",
                'strength': 'Medium',
                'timeframe': 'Short to Medium-term'
            })
        
        # MACD Analysis
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Histogram'] > 0:
            confluences['bullish'].append({
                'indicator': 'MACD',
                'condition': "MACD above signal line with positive histogram",
                'implication': "Bullish momentum building. Watch for histogram expansion for stronger moves.",
                'strength': 'Strong' if row['MACD_Histogram'] > 0 else 'Medium',
                'timeframe': 'Medium-term'
            })
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Histogram'] < 0:
            confluences['bearish'].append({
                'indicator': 'MACD',
                'condition': "MACD below signal line with negative histogram",
                'implication': "Bearish momentum building. Watch for histogram expansion for stronger moves.",
                'strength': 'Strong' if row['MACD_Histogram'] < 0 else 'Medium',
                'timeframe': 'Medium-term'
            })
        
        # ADX Trend Strength
        if row['ADX'] > 25:
            trend_direction = "bullish" if row['DI_Plus'] > row['DI_Minus'] else "bearish"
            confluences[trend_direction].append({
                'indicator': 'ADX Trend Strength',
                'condition': f"Strong trending market (ADX: {row['ADX']:.1f})",
                'implication': f"Strong {trend_direction} trend in place. Expect trend continuation with minor pullbacks.",
                'strength': 'Strong' if row['ADX'] > 40 else 'Medium',
                'timeframe': 'Medium to Long-term'
            })
        elif row['ADX'] < 20:
            confluences['neutral'].append({
                'indicator': 'ADX Trend Strength',
                'condition': f"Weak trending market (ADX: {row['ADX']:.1f})",
                'implication': "Market in consolidation/ranging phase. Look for breakout setups.",
                'strength': 'Medium',
                'timeframe': 'All timeframes'
            })
        
        return confluences
    
    def analyze_volatility_confluence(self, row):
        """Analyze volatility and mean reversion indicators"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Bollinger Bands Analysis
        bb_pos = row['BB_Position']
        if bb_pos < 0.1:  # Near lower band
            confluences['bullish'].append({
                'indicator': 'Bollinger Bands',
                'condition': f"Price near lower band (Position: {bb_pos:.2f})",
                'implication': "Potential mean reversion setup. Watch for bounce off lower band or breakdown.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif bb_pos > 0.9:  # Near upper band
            confluences['bearish'].append({
                'indicator': 'Bollinger Bands',
                'condition': f"Price near upper band (Position: {bb_pos:.2f})",
                'implication': "Potential mean reversion setup. Watch for rejection at upper band or breakout.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Bollinger Band Width
        if row['BB_Width'] < 2:  # Low volatility
            confluences['neutral'].append({
                'indicator': 'Bollinger Band Width',
                'condition': f"Low volatility environment (Width: {row['BB_Width']:.2f}%)",
                'implication': "Squeeze condition. Expect volatility expansion and potential breakout soon.",
                'strength': 'Strong',
                'timeframe': 'Short to Medium-term'
            })
        elif row['BB_Width'] > 8:  # High volatility
            confluences['neutral'].append({
                'indicator': 'Bollinger Band Width',
                'condition': f"High volatility environment (Width: {row['BB_Width']:.2f}%)",
                'implication': "Volatility expansion phase. Expect potential reversion to mean.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # ATR Analysis
        if row['ATR_Percent'] > 3:
            confluences['neutral'].append({
                'indicator': 'Average True Range',
                'condition': f"High volatility (ATR: {row['ATR_Percent']:.2f}%)",
                'implication': "Elevated volatility. Use wider stops and smaller position sizes.",
                'strength': 'Medium',
                'timeframe': 'All timeframes'
            })
        
        return confluences
    
    def analyze_volume_confluence(self, row):
        """Analyze volume-based confluences"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Volume Analysis
        if row['Volume_Ratio'] > 1.5:
            confluences['neutral'].append({
                'indicator': 'Volume',
                'condition': f"Above average volume ({row['Volume_Ratio']:.1f}x normal)",
                'implication': "Strong participation. Moves likely to be more sustainable.",
                'strength': 'Strong' if row['Volume_Ratio'] > 2 else 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['Volume_Ratio'] < 0.7:
            confluences['neutral'].append({
                'indicator': 'Volume',
                'condition': f"Below average volume ({row['Volume_Ratio']:.1f}x normal)",
                'implication': "Low participation. Moves may lack conviction and sustainability.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Chaikin Money Flow
        if row['CMF'] > 0.2:
            confluences['bullish'].append({
                'indicator': 'Chaikin Money Flow',
                'condition': f"Strong buying pressure (CMF: {row['CMF']:.2f})",
                'implication': "Money flowing into the asset. Supports bullish bias.",
                'strength': 'Strong' if row['CMF'] > 0.3 else 'Medium',
                'timeframe': 'Medium-term'
            })
        elif row['CMF'] < -0.2:
            confluences['bearish'].append({
                'indicator': 'Chaikin Money Flow',
                'condition': f"Strong selling pressure (CMF: {row['CMF']:.2f})",
                'implication': "Money flowing out of the asset. Supports bearish bias.",
                'strength': 'Strong' if row['CMF'] < -0.3 else 'Medium',
                'timeframe': 'Medium-term'
            })
        
        return confluences
    
    def analyze_price_action(self, row):
        """Analyze price action patterns"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Candle Analysis
        if row['Body_Size'] > 2:  # Large body
            candle_type = "bullish" if row['Close'] > row['Open'] else "bearish"
            confluences[candle_type].append({
                'indicator': 'Price Action',
                'condition': f"Large {candle_type} candle (Body: {row['Body_Size']:.2f}%)",
                'implication': f"Strong {candle_type} conviction. Expect follow-through in next few candles.",
                'strength': 'Strong' if row['Body_Size'] > 3 else 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Wick Analysis
        if row['Upper_Wick'] > row['Body_Size'] * 2 and row['Close'] > row['Open']:
            confluences['bearish'].append({
                'indicator': 'Price Action - Wicks',
                'condition': f"Long upper wick on bullish candle (Wick: {row['Upper_Wick']:.2f}%)",
                'implication': "Rejection at highs despite bullish close. Potential resistance area.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        if row['Lower_Wick'] > row['Body_Size'] * 2 and row['Close'] < row['Open']:
            confluences['bullish'].append({
                'indicator': 'Price Action - Wicks',
                'condition': f"Long lower wick on bearish candle (Wick: {row['Lower_Wick']:.2f}%)",
                'implication': "Support found at lows despite bearish close. Potential support area.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        return confluences
    
    def generate_comprehensive_analysis(self, df):
        """Generate comprehensive market analysis"""
        latest_row = df.iloc[-1]
        
        # Gather all confluences
        momentum_conf = self.analyze_momentum_confluence(latest_row)
        trend_conf = self.analyze_trend_confluence(latest_row)
        volatility_conf = self.analyze_volatility_confluence(latest_row)
        volume_conf = self.analyze_volume_confluence(latest_row)
        price_action_conf = self.analyze_price_action(latest_row)
        
        # Combine all confluences
        all_confluences = {
            'bullish': (momentum_conf['bullish'] + trend_conf['bullish'] + 
                       volatility_conf['bullish'] + volume_conf['bullish'] + 
                       price_action_conf['bullish']),
            'bearish': (momentum_conf['bearish'] + trend_conf['bearish'] + 
                       volatility_conf['bearish'] + volume_conf['bearish'] + 
                       price_action_conf['bearish']),
            'neutral': (momentum_conf['neutral'] + trend_conf['neutral'] + 
                       volatility_conf['neutral'] + volume_conf['neutral'] + 
                       price_action_conf['neutral'])
        }
        
        return all_confluences, latest_row
    
    def calculate_confluence_strength(self, confluences):
        """Calculate overall confluence strength"""
        strength_weights = {'Strong': 3, 'Medium': 2, 'Low': 1}
        
        bullish_score = sum(strength_weights.get(conf['strength'], 1) for conf in confluences['bullish'])
        bearish_score = sum(strength_weights.get(conf['strength'], 1) for conf in confluences['bearish'])
        neutral_score = sum(strength_weights.get(conf['strength'], 1) for conf in confluences['neutral'])
        
        total_score = bullish_score + bearish_score + neutral_score
        
        if total_score == 0:
            return "No Clear Signal", 0
        
        if bullish_score > bearish_score and bullish_score >= self.confluence_threshold:
            bias_strength = (bullish_score / total_score) * 100
            return "Bullish Bias", bias_strength
        elif bearish_score > bullish_score and bearish_score >= self.confluence_threshold:
            bias_strength = (bearish_score / total_score) * 100
            return "Bearish Bias", bias_strength
        else:
            return "Mixed/Neutral", max(bullish_score, bearish_score) / total_score * 100
    
    def display_analysis(self, symbol, timeframe, confluences, latest_row):
        """Display comprehensive analysis results"""
        print(f"\n{'='*80}")
        print(f"🔍 NUNNO'S ENHANCED TECHNICAL ANALYSIS - {symbol} ({timeframe})")
        print(f"{'='*80}")
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${latest_row['Close']:.4f}")
        print(f"📊 24h Range: ${latest_row['Low']:.4f} - ${latest_row['High']:.4f}")
        
        # Overall Market Bias
        bias, strength = self.calculate_confluence_strength(confluences)
        print(f"\n🎯 OVERALL MARKET BIAS: {bias} ({strength:.1f}% confidence)")
        
        # Bullish Confluences
        if confluences['bullish']:
            print(f"\n🟢 BULLISH CONFLUENCES ({len(confluences['bullish'])} signals):")
            print("-" * 60)
            for i, conf in enumerate(confluences['bullish'], 1):
                print(f"{i}. {conf['indicator']} [{conf['strength']}] - {conf['timeframe']}")
                print(f"   📍 Condition: {conf['condition']}")
                print(f"   💡 Implication: {conf['implication']}")
                print()
        
        # Bearish Confluences
        if confluences['bearish']:
            print(f"\n🔴 BEARISH CONFLUENCES ({len(confluences['bearish'])} signals):")
            print("-" * 60)
            for i, conf in enumerate(confluences['bearish'], 1):
                print(f"{i}. {conf['indicator']} [{conf['strength']}] - {conf['timeframe']}")
                print(f"   📍 Condition: {conf['condition']}")
                print(f"   💡 Implication: {conf['implication']}")
                print()
        
        # Neutral/Mixed Signals
        if confluences['neutral']:
            print(f"\n🟡 NEUTRAL/MIXED SIGNALS ({len(confluences['neutral'])} signals):")
            print("-" * 60)
            for i, conf in enumerate(confluences['neutral'], 1):
                print(f"{i}. {conf['indicator']} [{conf['strength']}] - {conf['timeframe']}")
                print(f"   📍 Condition: {conf['condition']}")
                print(f"   💡 Implication: {conf['implication']}")
                print()
        
        # Key Levels
        print(f"\n📊 KEY LEVELS:")
        print(f"   Pivot Point: ${latest_row['Pivot']:.4f}")
        print(f"   Resistance 1: ${latest_row['R1']:.4f}")
        print(f"   Support 1: ${latest_row['S1']:.4f}")
        print(f"   BB Upper: ${latest_row['BB_Upper']:.4f}")
        print(f"   BB Lower: ${latest_row['BB_Lower']:.4f}")
        print(f"   EMA 21: ${latest_row['EMA_21']:.4f}")
        print(f"   EMA 50: ${latest_row['EMA_50']:.4f}")
        
        # Risk Management
        atr_value = latest_row['ATR']
        print(f"\n⚠️  RISK MANAGEMENT:")
        print(f"   ATR: ${atr_value:.4f} ({latest_row['ATR_Percent']:.2f}%)")
        print(f"   Suggested Stop Distance: ${atr_value * 1.5:.4f}")
        print(f"   Volatility Level: {'High' if latest_row['ATR_Percent'] > 3 else 'Medium' if latest_row['ATR_Percent'] > 1.5 else 'Low'}")
        
        print(f"\n{'='*80}")
        print("⚡ Remember: This analysis is for educational purposes. Always use proper risk management!")
        print(f"{'='*80}")

def user_input_token():
    """Enhanced token selection with more options"""
    options = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", 
        "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT",
        "LINKUSDT", "UNIUSDT", "LTCUSDT", "BCHUSDT", "FILUSDT"
    ]
    print("\n🪙 Select a token to analyze:")
    for i, token in enumerate(options[:10], start=1):
        print(f"{i:2d}. {token}")
    print(f"11. More tokens...")
    print(f"12. Enter custom token")
    
    choice = input("\nYour choice: ").strip()
    
    if choice.isdigit():
        choice_num = int(choice)
        if 1 <= choice_num <= 10:
            return options[choice_num-1]
        elif choice_num == 11:
            print("\n📋 Additional tokens:")
            for i, token in enumerate(options[10:], start=11):
                print(f"{i:2d}. {token}")
            sub_choice = input("Select token: ").strip()
            if sub_choice.isdigit() and 11 <= int(sub_choice) <= len(options):
                return options[int(sub_choice)-1]
        elif choice_num == 12:
            custom = input("Enter custom token symbol (e.g., ATOMUSDT): ").upper().strip()
            if custom.endswith('USDT'):
                return custom
            else:
                return custom + 'USDT'
    
    print("Invalid choice. Defaulting to BTCUSDT.")
    return "BTCUSDT"

def user_input_timeframe():
    """Enhanced timeframe selection"""
    tf_options = {
        "1": ("1m", "1 Minute - Scalping"),
        "2": ("3m", "3 Minute - Short Scalping"), 
        "3": ("5m", "5 Minute - Scalping"),
        "4": ("15m", "15 Minute - Short Term"),
        "5": ("30m", "30 Minute - Short Term"),
        "6": ("1h", "1 Hour - Medium Term"),
        "7": ("2h", "2 Hour - Medium Term"),
        "8": ("4h", "4 Hour - Swing Trading"),
        "9": ("6h", "6 Hour - Swing Trading"),
        "10": ("12h", "12 Hour - Position"),
        "11": ("1d", "Daily - Position Trading")
    }
    
    print("\n⏰ Select a timeframe:")
    for key, (tf, description) in tf_options.items():
        print(f"{key:2s}. {tf:3s} - {description}")
    
    choice = input("\nYour choice: ").strip()
    selected = tf_options.get(choice, ("15m", "15 Minute - Short Term"))
    return selected[0]

def generate_trading_plan(confluences, latest_row, bias, strength):
    """Generate a structured trading plan based on confluences"""
    print(f"\n📋 TRADING PLAN SUGGESTIONS:")
    print("=" * 50)
    
    atr = latest_row['ATR']
    current_price = latest_row['Close']
    
    if bias == "Bullish Bias" and strength > 60:
        print("🎯 BULLISH SETUP IDENTIFIED")
        print(f"   Entry Strategy: Look for pullbacks to EMA 21 (${latest_row['EMA_21']:.4f}) or BB Middle")
        print(f"   Stop Loss: Below EMA 50 (${latest_row['EMA_50']:.4f}) or {atr*1.5:.4f} below entry")
        print(f"   Target 1: Pivot R1 (${latest_row['R1']:.4f})")
        print(f"   Target 2: BB Upper Band (${latest_row['BB_Upper']:.4f})")
        print(f"   Risk/Reward: Aim for 1:2 minimum ratio")
        
    elif bias == "Bearish Bias" and strength > 60:
        print("🎯 BEARISH SETUP IDENTIFIED")
        print(f"   Entry Strategy: Look for rallies to EMA 21 (${latest_row['EMA_21']:.4f}) or BB Middle")
        print(f"   Stop Loss: Above EMA 50 (${latest_row['EMA_50']:.4f}) or {atr*1.5:.4f} above entry")
        print(f"   Target 1: Pivot S1 (${latest_row['S1']:.4f})")
        print(f"   Target 2: BB Lower Band (${latest_row['BB_Lower']:.4f})")
        print(f"   Risk/Reward: Aim for 1:2 minimum ratio")
        
    else:
        print("⚖️ MIXED/RANGING MARKET")
        print(f"   Strategy: Range trading between key levels")
        print(f"   Buy Zone: Near BB Lower (${latest_row['BB_Lower']:.4f}) or Support")
        print(f"   Sell Zone: Near BB Upper (${latest_row['BB_Upper']:.4f}) or Resistance") 
        print(f"   Stop Loss: Beyond range boundaries + {atr:.4f}")
        print(f"   Wait for: Clear breakout with volume confirmation")
    
    print(f"\n⚠️  RISK MANAGEMENT RULES:")
    print(f"   • Position Size: Risk only 1-2% of capital per trade")
    print(f"   • ATR Stop: {atr:.4f} (Current volatility measure)")
    print(f"   • Volume Confirmation: Wait for volume > {latest_row['Volume_SMA']:.0f}")
    print(f"   • Time Filter: Avoid news events and low liquidity hours")

def main():
    """Enhanced main program with better error handling and user experience"""
    analyzer = TradingAnalyzer()
    
    try:
        print("🚀 Welcome to Nunno's Enhanced Trading Analysis System")
        print("=" * 60)
        
        # Get user inputs
        token = user_input_token()
        timeframe = user_input_timeframe()
        
        print(f"\n📊 Fetching data for {token} on {timeframe} timeframe...")
        print("⏳ Please wait while I analyze the market...")
        
        # Fetch and analyze data
        df = analyzer.fetch_binance_ohlcv(symbol=token, interval=timeframe, limit=1000)
        df = analyzer.add_comprehensive_indicators(df)
        
        if len(df) < 100:
            print("⚠️ Warning: Limited data available. Analysis may be less reliable.")
        
        # Generate comprehensive analysis
        confluences, latest_row = analyzer.generate_comprehensive_analysis(df)
        
        # Display results
        analyzer.display_analysis(token, timeframe, confluences, latest_row)
        
        # Calculate overall bias
        bias, strength = analyzer.calculate_confluence_strength(confluences)
        
        # Generate trading plan
        generate_trading_plan(confluences, latest_row, bias, strength)
        
        # Additional insights
        print(f"\n🔮 MARKET INSIGHTS:")
        print("-" * 30)
        
        # Momentum insight
        rsi = latest_row['RSI_14']
        if rsi > 50:
            print(f"📈 Momentum: Bullish momentum (RSI: {rsi:.1f})")
        else:
            print(f"📉 Momentum: Bearish momentum (RSI: {rsi:.1f})")
        
        # Trend insight
        if latest_row['EMA_9'] > latest_row['EMA_21']:
            print(f"📊 Short-term Trend: Bullish (EMA 9 > EMA 21)")
        else:
            print(f"📊 Short-term Trend: Bearish (EMA 9 < EMA 21)")
        
        # Volatility insight
        bb_width = latest_row['BB_Width']
        if bb_width < 2:
            print(f"🌊 Volatility: Low - Expect breakout soon (BB Width: {bb_width:.2f}%)")
        elif bb_width > 6:
            print(f"🌊 Volatility: High - Potential mean reversion (BB Width: {bb_width:.2f}%)")
        else:
            print(f"🌊 Volatility: Normal (BB Width: {bb_width:.2f}%)")
        
        # Volume insight
        vol_ratio = latest_row['Volume_Ratio']
        if vol_ratio > 1.5:
            print(f"📊 Volume: Above average ({vol_ratio:.1f}x) - Strong participation")
        elif vol_ratio < 0.7:
            print(f"📊 Volume: Below average ({vol_ratio:.1f}x) - Weak participation")
        else:
            print(f"📊 Volume: Average ({vol_ratio:.1f}x) - Normal participation")
            
        print(f"\n🎓 EDUCATIONAL TIP:")
        print("   Confluence trading means waiting for multiple indicators to align")
        print("   in the same direction. The more confluences, the higher the probability")
        print("   of a successful trade. Always combine technical analysis with proper")
        print("   risk management and market context.")
        
        print(f"\n💡 NEXT STEPS:")
        print("   1. Monitor the key levels mentioned above")
        print("   2. Wait for confluence confirmation before entering")
        print("   3. Set alerts at critical support/resistance levels") 
        print("   4. Keep an eye on volume for breakout confirmations")
        print("   5. Review higher timeframe context before trading")
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Analysis interrupted by user.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("💡 Suggestions:")
        print("   • Check your internet connection")
        print("   • Verify the token symbol is correct")
        print("   • Try a different timeframe")
        print("   • Binance API might be temporarily unavailable")
        
    finally:
        print(f"\n👋 Thank you for using Nunno's Enhanced Trading Analysis!")
        print("   Remember: Past performance doesn't guarantee future results.")
        print("   Always do your own research and trade responsibly! 🙏")

if __name__ == "__main__":
    main()