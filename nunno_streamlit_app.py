# nunno_streamlit_full.py
import streamlit as st
import requests
import base64
import re
import io
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from fuzzywuzzy import process


# Try importing local modules (optional - prediction & monte carlo features)
try:
    import betterpredictormodule
except Exception:
    betterpredictormodule = None

try:
    from montecarlo_module import simulate_trades, monte_carlo_summary
except Exception:
    simulate_trades = None
    monte_carlo_summary = None

# API keys (recommended to put into Streamlit secrets)
AI_API_KEY = st.secrets.get("AI_API_KEY", "")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

SYSTEM_PROMPT = (
    "You are Nunno, a friendly AI (Numinous Nexus AI). "
    "You teach trading and investing to complete beginners in very simple language. "
    "The user's name is {user_name}, age {user_age}. Tailor explanations for beginners. "
    "You have integrated prediction, tokenomics, monte carlo simulation and chart analysis. "
    "When giving outputs, make them neat with headings, tables, and emojis. "
    "If asked about your founder, say you were built by Mujtaba Kazmi."
)

MAX_HISTORY_MESSAGES = 20

def manage_history_length(history_list):
    if not history_list:
        return []
    system_msg = None
    if history_list and history_list[0].get("role") == "system":
        system_msg = history_list[0]
        temp = history_list[1:]
    else:
        temp = history_list
    if len(temp) > MAX_HISTORY_MESSAGES - 1:
        temp = temp[-(MAX_HISTORY_MESSAGES - 1):]
    if system_msg:
        return [system_msg] + temp
    return temp

def flatten_conversation_for_api(conv):
    msgs = []
    for m in conv:
        role = m.get("role", "user")
        if role == "system":
            msgs.append({"role":"system", "content": m.get("content", "")})
        elif role == "user":
            msgs.append({"role":"user", "content": m.get("content", "")})
        elif role == "assistant":
            kind = m.get("kind", "text")
            if kind == "tokenomics":
                data = m.get("data", {})
                text = "Tokenomics Analysis:\n" + "\n".join([f"- {k}: {v}" for k,v in data.items()])
                msgs.append({"role":"assistant", "content": text})
            elif kind == "prediction":
                data = m.get("data", {})
                text = f"Prediction for {data.get('symbol','')}: Bias {data.get('bias','')}, Strength {data.get('strength','')}\nPlan:\n{data.get('plan','')}"
                msgs.append({"role":"assistant", "content": text})
            elif kind == "news":
                headlines = m.get("data", [])
                text = "News headlines:\n" + "\n".join(headlines)
                msgs.append({"role":"assistant", "content": text})
            else:
                msgs.append({"role":"assistant", "content": m.get("content", "")})
    return msgs

# ---------------------------
# Tokenomics / Coingecko
# ---------------------------
def fetch_historical_prices(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=365"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return [p[1] for p in data.get("prices", [])]
    except Exception:
        return None

def calculate_cagr_and_volatility(prices):
    if not prices or len(prices) < 3:
        return None, None, None
    returns = [np.log(prices[i+1] / prices[i]) for i in range(len(prices)-1)]
    avg_daily = np.mean(returns)
    daily_vol = np.std(returns)
    ann_return = np.exp(avg_daily * 365) - 1
    ann_vol = daily_vol * np.sqrt(365)
    conservative = ann_return * 0.5
    return ann_return, ann_vol, conservative

def fetch_token_data(coin_id, investment_amount=1000):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower().strip()}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        market = data.get("market_data", {})
        circ = market.get("circulating_supply") or 0
        total = market.get("total_supply") or 0
        price = market.get("current_price", {}).get("usd", 0) or 0
        mcap = market.get("market_cap", {}).get("usd", 0) or 0
        fdv = total * price if total else 0
        circ_percent = (circ / total) * 100 if total else None
        fdv_mcap_ratio = (fdv / mcap) if mcap else None
        healthy = bool(circ_percent and circ_percent > 50 and fdv_mcap_ratio and fdv_mcap_ratio < 2)
        prices = fetch_historical_prices(coin_id)
        if prices and len(prices) > 10:
            cagr, vol, cons = calculate_cagr_and_volatility(prices)
        else:
            cagr, vol, cons = None, None, None
        exp_yearly = investment_amount * cons if cons else 0
        exp_monthly = exp_yearly / 12 if cons else 0

        result = {
            "Coin": f"{data.get('name','')} ({data.get('symbol','').upper()})",
            "Price": f"${price:,.6f}",
            "Market Cap": f"${mcap/1e9:,.2f}B" if mcap else "N/A",
            "Total Supply (M)": f"{total/1e6:,.2f}M" if total else "N/A",
            "Circulating Supply (M)": f"{circ/1e6:,.2f}M" if circ else "N/A",
            "Circulating %": f"{circ_percent:.2f}%" if circ_percent is not None else "N/A",
            "FDV (B)": f"${fdv/1e9:,.2f}B" if fdv else "N/A",
            "FDV/MarketCap Ratio": f"{fdv_mcap_ratio:.2f}" if fdv_mcap_ratio is not None else "N/A",
            "Historical CAGR": f"{cagr*100:.2f}%" if cagr else "N/A",
            "Annual Volatility": f"{vol*100:.2f}%" if vol else "N/A",
            "Realistic Yearly Return (50% CAGR)": f"{cons*100:.2f}%" if cons else "N/A",
            "Expected Monthly ($)": f"${exp_monthly:,.2f}",
            "Expected Yearly ($)": f"${exp_yearly:,.2f}",
            "Health": "✅ Healthy" if healthy else "⚠️ Risky or Inflated"
        }
        return result
    except Exception:
        return None

def tokenomics_df(token_data):
    if not token_data:
        return None
    df = pd.DataFrame(list(token_data.items()), columns=["Metric", "Value"])
    return df

def suggest_similar_tokens(user_input):
    try:
        res = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=10)
        res.raise_for_status()
        coin_list = res.json()
        coin_ids = [coin['id'] for coin in coin_list]
        best = process.extract(user_input.lower(), coin_ids, limit=5)
        return [b[0] for b in best if b[1] > 60]
    except Exception:
        return []

# ---------------------------
# Market news
# ---------------------------
def fetch_market_news():
    if not NEWS_API_KEY:
        return ["[Error] NEWS_API_KEY not configured in Streamlit secrets."]
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "finance OR bitcoin OR stock market OR federal reserve OR inflation",
        "from": datetime.now().strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        return [f"- {a['title']} ({a['source']['name']})" for a in data.get("articles", [])]
    except Exception as e:
        return [f"[Error fetching news] {e}"]

# ---------------------------
# AI / Chart calls
# ---------------------------
def ask_nunno(messages):
    if not AI_API_KEY:
        return "[Error] AI_API_KEY not configured."
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct",
        "messages": messages
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI Error] {e}"

def analyze_chart(image_b64):
    if not AI_API_KEY:
        return "[Error] AI_API_KEY not configured."
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type":"text", "text":"You're an expert trading analyst. Analyze this chart: identify trend, SR, patterns, and predict the next move."},
                {"type":"image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }],
        "max_tokens": 1000
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Chart API Error] {e}"

def is_tokenomics_request(text):
    """Check if request is specifically about tokenomics"""
    tokenomics_specific = [
        "tokenomics", "supply", "fdv", "market cap", "circulating", 
        "should i invest", "inflation rate", "token economics", "coin analysis"
    ]
    text_lower = text.lower()
    
    # Check for explicit tokenomics keywords
    has_tokenomics = any(keyword in text_lower for keyword in tokenomics_specific)
    
    # Check for prediction keywords that should override tokenomics
    prediction_override = any(keyword in text_lower for keyword in [
        "predict", "forecast", "next move", "price prediction", 
        "where will", "target price", "technical analysis", "trend"
    ])
    
    # If it has prediction keywords, it's NOT tokenomics
    if prediction_override:
        return False
        
    return has_tokenomics

def is_prediction_request(text):
    """Check if request is specifically about predictions/technical analysis"""
    prediction_keywords = [
        "predict", "forecast", "next move", "price prediction", 
        "where will", "target price", "technical analysis", "trend",
        "analysis", "chart", "bullish", "bearish", "support", "resistance"
    ]
    return any(keyword in text.lower() for keyword in prediction_keywords)

# ---------------------------
# Streamlit UI start
# ---------------------------
st.set_page_config(page_title="Nunno AI", page_icon="🧠", layout="wide")

# session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role":"system", "content": SYSTEM_PROMPT.format(user_name="User", user_age="N/A")}]
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"
if "user_age" not in st.session_state:
    st.session_state.user_age = "N/A"
if "uploaded_b64" not in st.session_state:
    st.session_state.uploaded_b64 = None

# sidebar
with st.sidebar:
    st.header("Profile & Controls")
    st.session_state.user_name = st.text_input("Your name", st.session_state.user_name)
    st.session_state.user_age = st.text_input("Your age (optional)", st.session_state.user_age)
    if st.button("Start New Chat"):
        st.session_state.conversation = [{"role":"system", "content": SYSTEM_PROMPT.format(user_name=st.session_state.user_name, user_age=st.session_state.user_age)}]
        st.rerun()

    st.markdown("---")
    st.subheader("Upload Chart (optional)")
    uploaded = st.file_uploader("Upload trading chart image (png/jpg)", type=["png","jpg","jpeg"])
    if uploaded is not None:
        st.session_state.uploaded_b64 = base64.b64encode(uploaded.read()).decode("utf-8")
        st.success("Chart uploaded and ready for analysis.")

    st.markdown("---")
    st.subheader("Quick Examples")
    if st.button("Analyze Bitcoin tokenomics with $1000"):
        st.session_state.conversation.append({"role":"user","content":"Analyze Bitcoin tokenomics with $1000"})
        st.rerun()
    if st.button("What's happening in the market?"):
        st.session_state.conversation.append({"role":"user","content":"What's happening in the market?"})
        st.rerun()
    if st.button("Predict BTC price movement"):
        st.session_state.conversation.append({"role":"user","content":"Predict BTC price movement 15m"})
        st.rerun()

# main layout
col1, col2 = st.columns([3,1])

with col1:
    st.header("Chat")
    # render conversation
    for msg in st.session_state.conversation:
        role = msg.get("role","user")
        if role == "system":
            continue  # Don't show system messages
        elif role == "user":
            with st.chat_message("user"):
                st.markdown(msg.get("content",""))
        elif role == "assistant":
            kind = msg.get("kind","text")
            if kind == "tokenomics":
                with st.chat_message("assistant"):
                    st.markdown("📊 **Tokenomics Analysis**")
                    df = tokenomics_df(msg.get("data",{}))
                    if df is not None:
                        st.table(df)
                    health = msg.get("data",{}).get("Health","")
                    if "✅" in health:
                        st.success(health)
                    else:
                        st.warning(health)
                    note = msg.get("content","")
                    if note:
                        st.markdown(note)
            elif kind == "prediction":
                with st.chat_message("assistant"):
                    data = msg.get("data",{})
                    bias = data.get("bias","")
                    strength = data.get("strength",0) or 0
                    symbol = data.get("symbol","")
                    tf = data.get("tf","")
                    
                    # Display prediction header
                    if isinstance(bias, str) and "bullish" in bias.lower():
                        st.success(f"🎯 {symbol} ({tf}) — Bias: {bias} ({strength:.1f}% confidence)")
                    elif isinstance(bias, str) and "bearish" in bias.lower():
                        st.error(f"🎯 {symbol} ({tf}) — Bias: {bias} ({strength:.1f}% confidence)")
                    else:
                        st.info(f"🎯 {symbol} ({tf}) — Bias: {bias} ({strength:.1f}% confidence)")

                    # Display confluences with better formatting
                    confluences = data.get("confluences", {})
                    if confluences:
                        st.markdown("### 📊 Confluence Analysis")
                        
                        # Bullish confluences
                        if confluences.get("bullish"):
                            st.markdown("#### 🟢 Bullish Signals")
                            for i, conf in enumerate(confluences["bullish"], 1):
                                with st.expander(f"{i}. {conf.get('indicator', 'Signal')} [{conf.get('strength', 'Medium')}]"):
                                    st.markdown(f"**Condition:** {conf.get('condition', 'N/A')}")
                                    st.markdown(f"**Implication:** {conf.get('implication', 'N/A')}")
                                    st.markdown(f"**Timeframe:** {conf.get('timeframe', 'N/A')}")
                        
                        # Bearish confluences
                        if confluences.get("bearish"):
                            st.markdown("#### 🔴 Bearish Signals")
                            for i, conf in enumerate(confluences["bearish"], 1):
                                with st.expander(f"{i}. {conf.get('indicator', 'Signal')} [{conf.get('strength', 'Medium')}]"):
                                    st.markdown(f"**Condition:** {conf.get('condition', 'N/A')}")
                                    st.markdown(f"**Implication:** {conf.get('implication', 'N/A')}")
                                    st.markdown(f"**Timeframe:** {conf.get('timeframe', 'N/A')}")
                        
                        # Neutral signals
                        if confluences.get("neutral"):
                            st.markdown("#### 🟡 Neutral/Mixed Signals")
                            for i, conf in enumerate(confluences["neutral"], 1):
                                with st.expander(f"{i}. {conf.get('indicator', 'Signal')} [{conf.get('strength', 'Medium')}]"):
                                    st.markdown(f"**Condition:** {conf.get('condition', 'N/A')}")
                                    st.markdown(f"**Implication:** {conf.get('implication', 'N/A')}")
                                    st.markdown(f"**Timeframe:** {conf.get('timeframe', 'N/A')}")

                    # Display trading plan
                    plan = data.get("plan","")
                    if plan:
                        st.markdown("### 📋 Trading Plan")
                        st.text(plan)
                        
                    # Display key levels if available
                    latest_data = data.get("latest_data")
                    if latest_data:
                        st.markdown("### 📊 Key Levels")
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Current Price", f"${latest_data.get('Close', 0):.4f}")
                            st.metric("EMA 21", f"${latest_data.get('EMA_21', 0):.4f}")
                            st.metric("EMA 50", f"${latest_data.get('EMA_50', 0):.4f}")
                        with cols[1]:
                            st.metric("RSI", f"{latest_data.get('RSI_14', 0):.1f}")
                            st.metric("BB Upper", f"${latest_data.get('BB_Upper', 0):.4f}")
                            st.metric("BB Lower", f"${latest_data.get('BB_Lower', 0):.4f}")
                    
                    note = msg.get("content","")
                    if note:
                        st.markdown("### 💡 Additional Notes")
                        st.markdown(note)
                        
            elif kind == "montecarlo":
                with st.chat_message("assistant"):
                    st.markdown("🧪 **Monte Carlo Simulation**")
                    st.markdown(msg.get("content",""))
            elif kind == "news":
                with st.chat_message("assistant"):
                    st.markdown("📰 **Market News**")
                    for h in msg.get("data",[]):
                        st.markdown(h)
                    if msg.get("content"):
                        st.markdown("**AI Explanation:**")
                        st.markdown(msg.get("content"))
            elif kind == "chart":
                with st.chat_message("assistant"):
                    st.markdown("📷 **Chart Analysis**")
                    st.markdown(msg.get("content",""))
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg.get("content",""))

    # Chat input
    prompt = st.chat_input("Ask Nunno about trading, tokenomics, predictions, news...")
    if prompt:
        st.session_state.conversation.append({"role":"user","content":prompt})
        lower = prompt.lower()

        assistant_entry = {"role":"assistant", "kind":"text", "content":""}

        # PREDICTION - Check this FIRST and make it more specific
        if is_prediction_request(prompt):
            if betterpredictormodule is None:
                assistant_entry["content"] = "Prediction features require the local module 'betterpredictormodule'. It's not available on this server."
            else:
                # Extract symbol
                symbol = "BTCUSDT"
                symbol_mappings = {
                    "btc": "BTCUSDT", "bitcoin": "BTCUSDT", "xbt": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT",
    "bnb": "BNBUSDT", "binance": "BNBUSDT",

    # Layer 1s
    "sol": "SOLUSDT", "solana": "SOLUSDT",
    "ada": "ADAUSDT", "cardano": "ADAUSDT",
    "avax": "AVAXUSDT", "avalanche": "AVAXUSDT",
    "dot": "DOTUSDT", "polkadot": "DOTUSDT",
    "atom": "ATOMUSDT", "cosmos": "ATOMUSDT",
    "near": "NEARUSDT", "near protocol": "NEARUSDT",
    "algo": "ALGOUSDT", "algorand": "ALGOUSDT",
    "apt": "APTUSDT", "aptos": "APTUSDT",
    "sui": "SUIUSDT", "sui network": "SUIUSDT",

    # Layer 2s / Scaling
    "matic": "MATICUSDT", "polygon": "MATICUSDT",
    "op": "OPUSDT", "optimism": "OPUSDT",
    "arb": "ARBUSDT", "arbitrum": "ARBUSDT",
    "imx": "IMXUSDT", "immutable": "IMXUSDT",

    # Meme Coins
    "doge": "DOGEUSDT", "dogecoin": "DOGEUSDT",
    "shib": "SHIBUSDT", "shiba": "SHIBUSDT", "shiba inu": "SHIBUSDT",
    "pepe": "PEPEUSDT", "pepe coin": "PEPEUSDT",
    "floki": "FLOKIUSDT", "floki inu": "FLOKIUSDT",

    # Stablecoins
    "usdt": "USDTUSDT", "tether": "USDTUSDT",   # self-pair just in case
    "usdc": "USDCUSDT", "usd coin": "USDCUSDT",
    "dai": "DAIUSDT",
    "busd": "BUSDUSDT", "binance usd": "BUSDUSDT",
    "tusd": "TUSDUSDT", "trueusd": "TUSDUSDT",

    # Other Majors & DeFi
    "xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "ltc": "LTCUSDT", "litecoin": "LTCUSDT",
    "link": "LINKUSDT", "chainlink": "LINKUSDT",
    "uni": "UNIUSDT", "uniswap": "UNIUSDT",
    "aave": "AAVEUSDT",
    "comp": "COMPUSDT", "compound": "COMPUSDT",
    "sand": "SANDUSDT", "sandbox": "SANDUSDT",
    "mana": "MANAUSDT", "decentraland": "MANAUSDT",
    "axs": "AXSUSDT", "axie": "AXSUSDT",
    "rndr": "RNDRUSDT", "render": "RNDRUSDT",
    "gala": "GALAUSDT",
    "fil": "FILUSDT", "filecoin": "FILUSDT",
    "icp": "ICPUSDT", "internet computer": "ICPUSDT",
    "hbar": "HBARUSDT", "hedera": "HBARUSDT",
                }
                
                for key, val in symbol_mappings.items():
                    if key in lower:
                        symbol = val
                        break
                
                # Extract timeframe
                tf = "15m"
                tf_mappings = {
                    "1m": "1m", "1 minute": "1m", "1min": "1m",
                    "5m": "5m", "5 minute": "5m", "5min": "5m", 
                    "15m": "15m", "15 minute": "15m", "15min": "15m",
                    "1h": "1h", "1 hour": "1h", "1hr": "1h", "hourly": "1h",
                    "4h": "4h", "4 hour": "4h", "4hr": "4h",
                    "1d": "1d", "daily": "1d", "day": "1d"
                }
                
                for key, val in tf_mappings.items():
                    if key in lower:
                        tf = val
                        break
                
                try:
                    analyzer = betterpredictormodule.TradingAnalyzer()
                    df = analyzer.fetch_binance_ohlcv(symbol=symbol, interval=tf, limit=1000)
                    df = analyzer.add_comprehensive_indicators(df)
                    confluences, latest = analyzer.generate_comprehensive_analysis(df)
                    bias, strength = analyzer.calculate_confluence_strength(confluences)
                    
                    # Capture trading plan output
                    old_stdout = io.StringIO()
                    backup = sys.stdout
                    try:
                        sys.stdout = old_stdout
                        betterpredictormodule.generate_trading_plan(confluences, latest, bias, strength)
                    finally:
                        sys.stdout = backup
                    plan_text = old_stdout.getvalue()
                    
                    assistant_entry["kind"] = "prediction"
                    assistant_entry["data"] = {
                        "symbol": symbol,
                        "tf": tf,
                        "bias": bias,
                        "strength": strength,
                        "confluences": confluences,
                        "plan": plan_text,
                        "latest_data": latest.to_dict()  # Add latest data for key levels
                    }
                    assistant_entry["content"] = f"Technical analysis complete for {symbol} on {tf} timeframe."
                except Exception as e:
                    assistant_entry["content"] = f"Prediction error: {e}"
                    
        # TOKENOMICS - Only if explicitly tokenomics
        elif is_tokenomics_request(prompt):
            detected = None
            coin_mappings = {
                'bitcoin': 'bitcoin', 'btc': 'bitcoin', 'xbt': 'bitcoin',
    'ethereum': 'ethereum', 'eth': 'ethereum',
    'binance': 'binancecoin', 'bnb': 'binancecoin',

    # Layer 1s
    'solana': 'solana', 'sol': 'solana',
    'cardano': 'cardano', 'ada': 'cardano',
    'avalanche': 'avalanche-2', 'avax': 'avalanche-2',
    'polkadot': 'polkadot', 'dot': 'polkadot',
    'cosmos': 'cosmos', 'atom': 'cosmos',
    'near': 'near', 'near protocol': 'near',
    'algorand': 'algorand', 'algo': 'algorand',
    'aptos': 'aptos', 'apt': 'aptos',
    'sui': 'sui', 'sui network': 'sui',

    # Layer 2s / Scaling
    'polygon': 'matic-network', 'matic': 'matic-network',
    'optimism': 'optimism', 'op': 'optimism',
    'arbitrum': 'arbitrum', 'arb': 'arbitrum',
    'immutable': 'immutable-x', 'imx': 'immutable-x',

    # Meme Coins
    'dogecoin': 'dogecoin', 'doge': 'dogecoin',
    'shiba': 'shiba-inu', 'shib': 'shiba-inu',
    'pepe': 'pepe', 'pepe coin': 'pepe',
    'floki': 'floki', 'floki inu': 'floki',

    # Stablecoins
    'usdt': 'tether', 'tether': 'tether',
    'usdc': 'usd-coin', 'usd coin': 'usd-coin',
    'dai': 'dai',
    'busd': 'binance-usd', 'binance usd': 'binance-usd',
    'tusd': 'true-usd', 'trueusd': 'true-usd',

    # Other Majors
    'xrp': 'ripple', 'ripple': 'ripple',
    'ltc': 'litecoin', 'litecoin': 'litecoin',
    'link': 'chainlink', 'chainlink': 'chainlink',
    'uni': 'uniswap', 'uniswap': 'uniswap',
    'aave': 'aave',
    'comp': 'compound-governance-token', 'compound': 'compound-governance-token',
    'sand': 'the-sandbox', 'sandbox': 'the-sandbox',
    'mana': 'decentraland', 'decentraland': 'decentraland',
    'axs': 'axie-infinity', 'axie': 'axie-infinity',
    'render': 'render-token', 'rndr': 'render-token',
    'gala': 'gala',
    'fil': 'filecoin', 'filecoin': 'filecoin',
    'icp': 'internet-computer', 'internet computer': 'internet-computer',
    'hbar': 'hedera-hashgraph', 'hedera': 'hedera-hashgraph',
            }
            for k,v in coin_mappings.items():
                if k in lower:
                    detected = v
                    break
            if not detected:
                words = re.findall(r"[A-Za-z\-]+", lower)
                if words:
                    candidates = suggest_similar_tokens(words[0])
                    if candidates:
                        detected = candidates[0]
            
            # parse amount
            amount = 1000
            m = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', prompt)
            if m:
                try:
                    amount = float(m.group(1).replace(',',''))
                except:
                    amount = 1000
                    
            if not detected:
                assistant_entry["content"] = "Please specify which coin to analyze (e.g., 'Analyze Bitcoin tokenomics' or 'Analyze SOL with $500')."
            else:
                token_data = fetch_token_data(detected, amount)
                if not token_data:
                    assistant_entry["content"] = f"Couldn't fetch tokenomics data for '{detected}'. Try a different coin or check your internet connection."
                else:
                    assistant_entry["kind"] = "tokenomics"
                    assistant_entry["data"] = token_data
                    assistant_entry["content"] = f"Tokenomics analysis for {token_data.get('Coin')} with ${amount:,.2f} investment."
                    
        # MONTE CARLO
        elif any(k in lower for k in ["simulate","monte carlo","simulate strategy"]):
            if simulate_trades is None:
                assistant_entry["content"] = "Monte Carlo simulation requires 'montecarlo_module' available on the server."
            else:
                win = 0.6
                rr = 1.5
                ntr = 100
                m = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*win', lower)
                if m:
                    win = float(m.group(1))/100
                m2 = re.search(r'(\d+(?:\.\d+)?)\s*(rr|r:r|risk:?reward)', lower)
                if m2:
                    rr = float(m2.group(1))
                try:
                    result = simulate_trades(win, rr, ntr, "choppy")
                    summary = monte_carlo_summary(result)
                    assistant_entry["kind"] = "montecarlo"
                    assistant_entry["content"] = summary
                except Exception as e:
                    assistant_entry["content"] = f"Monte Carlo error: {e}"
                    
        # NEWS
        elif any(k in lower for k in ["news","what's happening","market news","headlines"]):
            headlines = fetch_market_news()
            assistant_entry["kind"] = "news"
            assistant_entry["data"] = headlines
            if AI_API_KEY:
                msgs = flatten_conversation_for_api(st.session_state.conversation)
                msgs.append({"role":"user","content":"Please explain these headlines in simple language for a beginner:\n" + "\n".join(headlines)})
                ai_expl = ask_nunno(manage_history_length(msgs))
                assistant_entry["content"] = ai_expl or ""
            else:
                assistant_entry["content"] = ""
                
        # CHART
        elif any(k in lower for k in ["chart","analyze chart","analyze the chart","upload chart","analyze image"]):
            if st.session_state.uploaded_b64:
                res = analyze_chart(st.session_state.uploaded_b64)
                assistant_entry["kind"] = "chart"
                assistant_entry["content"] = res
            else:
                assistant_entry["content"] = "No chart uploaded. Please upload a chart image in the sidebar and try again."
                
        # DEFAULT CHAT
        else:
            msgs = flatten_conversation_for_api(st.session_state.conversation)
            msgs.append({"role":"user","content": prompt})
            ai_resp = ask_nunno(manage_history_length(msgs))
            assistant_entry["content"] = ai_resp

        st.session_state.conversation.append(assistant_entry)
        st.rerun()

with col2:
    st.header("Tools & Quick Actions")
    st.subheader("Tokenomics quick test")
    coin = st.text_input("Coin id or name (e.g., bitcoin, ethereum)", value="")
    invest = st.number_input("Investment amount ($)", value=1000)
    if st.button("Analyze coin (sidebar)"):
        if coin.strip():
            td = fetch_token_data(coin.strip(), invest)
            if td:
                st.table(tokenomics_df(td))
                st.success(td.get("Health",""))
            else:
                st.error("Could not fetch tokenomics for that coin.")
    st.markdown("---")
    st.subheader("Market News")
    if st.button("Fetch headlines"):
        news = fetch_market_news()
        for h in news:
            st.markdown(h)
    st.markdown("---")
    st.subheader("Uploaded Chart")
    if st.session_state.uploaded_b64:
        st.image("data:image/png;base64," + st.session_state.uploaded_b64, use_column_width=True)
        if st.button("Analyze uploaded chart"):
            res = analyze_chart(st.session_state.uploaded_b64)
            st.text(res)

st.caption("Notes: Put AI_API_KEY and NEWS_API_KEY in Streamlit secrets for full functionality. Local modules (betterpredictormodule, montecarlo_module) must be available for prediction/monte carlo features.")
