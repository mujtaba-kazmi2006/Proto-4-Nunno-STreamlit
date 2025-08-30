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
try:
    AI_API_KEY = st.secrets.get("AI_API_KEY", "")
    NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
except:
    AI_API_KEY = ""
    NEWS_API_KEY = ""

SYSTEM_PROMPT = (
    "You are Nunno, a friendly AI (Numinous Nexus AI). "
    "You teach trading and investing to complete beginners in very simple language. "
    "The user's name is {user_name}, age {user_age}. Tailor explanations for beginners. "
    "You have integrated prediction, tokenomics, monte carlo simulation and chart analysis. "
    "When giving outputs, make them neat with headings, tables, and emojis. "
    "If asked about your founder, say you were built by Mujtaba Kazmi."
)

MAX_HISTORY_MESSAGES = 20

def get_theme_css(theme):
    """Generate CSS for professional dark/light mode themes"""
    if theme == "dark":
        return """
        <style>
        /* Global transition for smooth theme switching */
        * {
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
        }
        
        .stApp {
            background: #1a1a1a;
            color: #ffffff !important;
            transition: background-color 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* Fix Streamlit header - the white bar at the top */
        header[data-testid="stHeader"],
        .stAppHeader,
        .st-emotion-cache-1eyfjps {
            background: #1a1a1a !important;
            background-color: #1a1a1a !important;
            border-bottom: 1px solid #30363d !important;
        }
        
        /* Fix the decoration bar inside header */
        .stDecoration,
        .st-emotion-cache-1dp5vir {
            background: #1a1a1a !important;
            background-color: #1a1a1a !important;
        }
        
        /* Fix toolbar area */
        .stAppToolbar,
        .st-emotion-cache-15ecox0 {
            background: #1a1a1a !important;
            background-color: #1a1a1a !important;
        }
        
        /* Fix toolbar actions */
        .stToolbarActions,
        .st-emotion-cache-1p1m4ay {
            background: #1a1a1a !important;
        }
        
        /* Style the Deploy button and menu in header */
        button[data-testid="stBaseButton-header"],
        button[data-testid="stBaseButton-headerNoPadding"],
        .st-emotion-cache-usvq0g,
        .st-emotion-cache-1w7bu1y {
            background: #30363d !important;
            color: #ffffff !important;
            border: 1px solid #484f58 !important;
        }
        
        /* Hide or style the main menu */
        .stMainMenu {
            background: #1a1a1a !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
            background: #161b22 !important;
            border-right: 1px solid #30363d;
        }
        
        .css-1d391kg *, .css-1lcbmhc *, section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Nuclear approach for white bar above chat */
        .stApp > div {
            background: #1a1a1a !important;
        }
        
        .stApp > div > div {
            background: #1a1a1a !important;
        }
        
        .stApp > div > div > div {
            background: #1a1a1a !important;
        }
        
        .stApp > div > div > div > div {
            background: #1a1a1a !important;
        }
        
        /* Target all possible main content wrappers */
        .main, 
        .main > div,
        .main > div > div,
        .main > div > div > div,
        div[data-testid="block-container"],
        div[data-testid="stVerticalBlock"],
        div[data-testid="column"] {
            background: #1a1a1a !important;
            background-color: #1a1a1a !important;
        }
        
        /* Override any element that might be white */
        .stApp div:not([data-testid="stChatMessage"]):not(.stFileUploader):not(.stButton) {
            background: transparent !important;
        }
        
        /* Specific fix for main container */
        .css-1y4p8pa, .css-12oz5g7, .css-1629p8f {
            background: #1a1a1a !important;
        }
        
        /* Chat messages */
        .stChatMessage {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            margin: 1rem 0;
            padding: 1rem;
        }
        
        .stChatMessage * {
            color: #ffffff !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            background: #238636 !important;
            color: white !important;
            border: 1px solid #2ea043 !important;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-size: 14px;
        }
        
        .stButton > button:hover {
            background: #2ea043 !important;
        }
        
        /* Theme toggle button - Exact selector targeting */
        button[data-testid="stBaseButton-secondary"][kind="secondary"] {
            background: #30363d !important;
            color: #ffffff !important;
            border: 1px solid #484f58 !important;
            border-radius: 6px !important;
            padding: 0.4rem 0.6rem !important;
            min-width: 42px !important;
            height: 38px !important;
        }
        
        /* Target the specific emotion-cache class if needed */
        .st-emotion-cache-qm7g72 {
            background: #30363d !important;
            color: #ffffff !important;
            border: 1px solid #484f58 !important;
            border-radius: 6px !important;
        }
        
        /* Target the inner markdown container and paragraph */
        button[data-testid="stBaseButton-secondary"] .st-emotion-cache-p7i6r9,
        button[data-testid="stBaseButton-secondary"] .st-emotion-cache-p7i6r9 p {
            color: #ffffff !important;
        }
        
        /* Backup selector using the button structure */
        button[kind="secondary"]:has(div[data-testid="stMarkdownContainer"]) {
            background: #30363d !important;
            color: #ffffff !important;
            border: 1px solid #484f58 !important;
        }
        
        /* Input fields - FIXED for dark mode visibility */
        .stTextInput > div > div > input, 
        .stTextArea > div > div > textarea, 
        .stChatInput > div > div > input {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            color: #ffffff !important;
            border-radius: 6px;
        }
        
        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder,
        .stChatInput > div > div > input::placeholder {
            color: #8b949e !important;
            opacity: 1 !important;
            transition: color 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus, 
        .stTextArea > div > div > textarea:focus, 
        .stChatInput > div > div > input:focus {
            border-color: #388bfd !important;
            box-shadow: 0 0 0 2px rgba(56, 139, 253, 0.3) !important;
        }
        
        /* Chat input styling - Nuclear approach for stubborn Streamlit styling */
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div,
        [data-testid="stChatInput"] > div > div > div {
            background: #21262d !important;
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
        }
        
        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] > div input,
        [data-testid="stChatInput"] > div textarea,
        [data-testid="stChatInput"] > div > div input,
        [data-testid="stChatInput"] > div > div textarea,
        [data-testid="stChatInput"] > div > div > div input,
        [data-testid="stChatInput"] > div > div > div textarea {
            background: #21262d !important;
            background-color: #21262d !important;
            color: #ffffff !important;
            border: none !important;
            border-color: transparent !important;
        }
        
        [data-testid="stChatInput"] input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {
            color: #8b949e !important;
        }
        
        /* Catch-all for any remaining chat input elements */
        .stApp [data-testid="stChatInput"] * {
            background: #21262d !important;
            color: #ffffff !important;
        }
        
        /* Override any emotion-cache classes that might be interfering */
        div[class*="st-emotion-cache"] input,
        div[class*="st-emotion-cache"] textarea {
            background: #21262d !important;
            color: #ffffff !important;
        }
        
        /* File uploader - FIXED for dark mode */
        .stFileUploader > div {
            background: #21262d !important;
            border: 1px dashed #30363d !important;
            border-radius: 6px;
            color: #ffffff !important;
        }
        
        .stFileUploader * {
            color: #ffffff !important;
        }
        
        /* File uploader button and text */
        .stFileUploader label {
            color: #ffffff !important;
        }
        
        .stFileUploader button {
            background: #30363d !important;
            color: #ffffff !important;
            border: 1px solid #484f58 !important;
        }
        
        /* File uploader drag area */
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: #21262d !important;
            border: 2px dashed #30363d !important;
            color: #ffffff !important;
        }
        
        .stFileUploader [data-testid="stFileUploaderDropzone"] * {
            color: #ffffff !important;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: #238636 !important;
            border: 1px solid #2ea043 !important;
            color: white !important;
            border-radius: 6px;
        }
        
        .stError, .stWarning {
            background: #da3633 !important;
            border: 1px solid #f85149 !important;
            color: white !important;
            border-radius: 6px;
        }
        
        .stInfo {
            background: #1f6feb !important;
            border: 1px solid #388bfd !important;
            color: white !important;
            border-radius: 6px;
        }
        
        /* Tables */
        .stDataFrame {
            background: #21262d !important;
            border-radius: 6px;
            border: 1px solid #30363d;
        }
        
        .stDataFrame * {
            color: #ffffff !important;
            background: #21262d !important;
        }
        
        /* Metrics */
        div[data-testid="metric-container"] {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            padding: 1rem;
        }
        
        div[data-testid="metric-container"] * {
            color: #ffffff !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background: #21262d !important;
            color: #ffffff !important;
        }
        
        .streamlit-expanderContent {
            background: #161b22 !important;
            border: 1px solid #30363d !important;
        }
        
        /* Select boxes */
        .stSelectbox * {
            color: #ffffff !important;
            background: #21262d !important;
        }
        
        /* Select box dropdown */
        .stSelectbox > div > div {
            background: #21262d !important;
            border: 1px solid #30363d !important;
        }
        
        .stSelectbox option {
            background: #21262d !important;
            color: #ffffff !important;
        }
        
        /* Markdown text */
        .stMarkdown * {
            color: #ffffff !important;
        }
        
        /* Labels for inputs */
        label {
            color: #ffffff !important;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background: #238636 !important;
        }
        
        /* Sliders */
        .stSlider * {
            color: #ffffff !important;
        }
        
        /* Number input */
        .stNumberInput > div > div > input {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            color: #ffffff !important;
        }
        
        /* Date input */
        .stDateInput > div > div > input {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            color: #ffffff !important;
        }
        
        /* Time input */
        .stTimeInput > div > div > input {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            color: #ffffff !important;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background: #0d1117 !important;
            border: 1px solid #30363d !important;
        }
        
        .stCodeBlock * {
            color: #e6edf3 !important;
        }
        </style>
        """
    else:  # light theme
        return """
        <style>
        /* Global transition for smooth theme switching */
        * {
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
        }
        
        .stApp {
            background: #ffffff;
            color: #1a202c !important;
            transition: background-color 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* Global text color */
        .stApp * {
            color: #1a202c !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
            background: #f7fafc !important;
            border-right: 1px solid #e2e8f0;
        }
        
        .css-1d391kg *, .css-1lcbmhc *, section[data-testid="stSidebar"] * {
            color: #1a202c !important;
        }
        
        /* Main content area */
        .main .block-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
        }
        
        /* Chat messages */
        .stChatMessage {
            background: rgba(247, 250, 252, 0.8) !important;
            border: 1px solid #cbd5e0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
            margin: 1rem 0;
            padding: 1rem;
        }
        
        /* User messages */
        div[data-testid="stChatMessageContainer"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, #3182ce 0%, #2b77cb 100%) !important;
            border: 1px solid #4299e1 !important;
            color: white !important;
        }
        
        /* Assistant messages */
        div[data-testid="stChatMessageContainer"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(135deg, #38a169 0%, #48bb78 100%) !important;
            border: 1px solid #68d391 !important;
            color: white !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2d3748 !important;
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            color: #3182ce !important;
            background: linear-gradient(90deg, #3182ce, #4299e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3182ce 0%, #2b77cb 100%) !important;
            color: white !important;
            border: 1px solid #4299e1 !important;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2c5aa0 0%, #2a69ac 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(49, 130, 206, 0.4);
        }
        
        /* Input fields */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stChatInput > div > div > input {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px solid #e2e8f0 !important;
            color: #2d3748 !important;
            border-radius: 8px;
            padding: 0.75rem;
            font-size: 16px;
        }
        
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus, .stChatInput > div > div > input:focus {
            border-color: #3182ce !important;
            box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.2) !important;
        }
        
        /* File uploader */
        .stFileUploader > div {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px dashed #e2e8f0 !important;
            border-radius: 8px;
            padding: 2rem;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: linear-gradient(135deg, #38a169 0%, #48bb78 100%) !important;
            border: 1px solid #68d391 !important;
            color: white !important;
            border-radius: 8px;
        }
        
        .stError {
            background: linear-gradient(135deg, #e53e3e 0%, #fc8181 100%) !important;
            border: 1px solid #feb2b2 !important;
            color: white !important;
            border-radius: 8px;
        }
        
        /* Tables */
        .stDataFrame {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        /* Metrics */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 1rem;
        }
        
        /* Toggle switch */
        .stToggle > label {
            color: #2d3748 !important;
            font-weight: 500;
        }
        </style>
        """

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

# Theme state management
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role":"system", "content": SYSTEM_PROMPT.format(user_name="User", user_age="N/A")}]
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"
if "user_age" not in st.session_state:
    st.session_state.user_age = "N/A"
if "uploaded_b64" not in st.session_state:
    st.session_state.uploaded_b64 = None
if "chart_analysis" not in st.session_state:
    st.session_state.chart_analysis = None

# sidebar
with st.sidebar:
    # Header with simple theme toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Profile & Controls")
    with col2:
        # Simple toggle button
        if st.button("◐", help="Toggle theme", key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    
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
        
        # Add analyze button when chart is uploaded
        if st.button("🔍 Analyze Chart", key="analyze_chart_btn"):
            with st.spinner("Analyzing chart..."):
                result = analyze_chart(st.session_state.uploaded_b64)
                st.session_state.chart_analysis = result
            st.rerun()

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
                        st.success(f"🎯 {symbol} ({tf}) – Bias: {bias} ({strength:.1f}% confidence)")
                    elif isinstance(bias, str) and "bearish" in bias.lower():
                        st.error(f"🎯 {symbol} ({tf}) – Bias: {bias} ({strength:.1f}% confidence)")
                    else:
                        st.info(f"🎯 {symbol} ({tf}) – Bias: {bias} ({strength:.1f}% confidence)")

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
                        "latest_data": latest.to_dict() if latest is not None else None
                    }
                    assistant_entry["content"] = f"Completed technical analysis for {symbol} on {tf} timeframe."
                    
                except Exception as e:
                    assistant_entry["content"] = f"Prediction error: {e}"

        # CHART ANALYSIS - Now handled in sidebar, remove from main chat flow
        # elif ("chart" in lower or "analyze my uploaded chart" in lower) and st.session_state.uploaded_b64:
        #     result = analyze_chart(st.session_state.uploaded_b64)
        #     assistant_entry["kind"] = "chart"
        #     assistant_entry["content"] = result
        #     # Clear uploaded chart after analysis
        #     st.session_state.uploaded_b64 = None

        # TOKENOMICS - now comes AFTER prediction check
        elif is_tokenomics_request(prompt):
            # Extract investment amount
            investment = 1000
            match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', prompt)
            if match:
                investment = float(match.group(1).replace(',', ''))
            
            # Extract coin/token
            coin = "bitcoin"
            common_coins = {
                "btc": "bitcoin", "bitcoin": "bitcoin",
                "eth": "ethereum", "ethereum": "ethereum",
                "ada": "cardano", "cardano": "cardano",
                "sol": "solana", "solana": "solana",
                "doge": "dogecoin", "dogecoin": "dogecoin",
                "shib": "shiba-inu", "shiba": "shiba-inu",
                "matic": "polygon", "polygon": "polygon",
                "avax": "avalanche-2", "avalanche": "avalanche-2",
                "dot": "polkadot", "polkadot": "polkadot",
                "link": "chainlink", "chainlink": "chainlink",
                "uni": "uniswap", "uniswap": "uniswap",
                "xrp": "ripple", "ripple": "ripple",
                "ltc": "litecoin", "litecoin": "litecoin"
            }
            
            for key, val in common_coins.items():
                if key in lower:
                    coin = val
                    break
            
            # If no common coin found, try fuzzy matching
            if coin == "bitcoin" and not any(k in lower for k in common_coins.keys()):
                tokens = re.findall(r'\b([a-z]{2,10})\b', lower)
                if tokens:
                    suggestions = suggest_similar_tokens(tokens[0])
                    if suggestions:
                        coin = suggestions[0]
            
            token_data = fetch_token_data(coin, investment)
            if token_data:
                assistant_entry["kind"] = "tokenomics"
                assistant_entry["data"] = token_data
                assistant_entry["content"] = f"Based on the analysis, here's what your ${investment:,} investment could look like."
            else:
                assistant_entry["content"] = f"Sorry, couldn't find tokenomics data for '{coin}'. Try a different coin name or symbol."

        # NEWS
        elif "news" in lower or "market" in lower or "happening" in lower:
            headlines = fetch_market_news()
            assistant_entry["kind"] = "news"
            assistant_entry["data"] = headlines
            if not any("Error" in h for h in headlines):
                # Get AI summary of news
                news_text = "\n".join(headlines)
                ai_messages = flatten_conversation_for_api(st.session_state.conversation)
                ai_messages.append({"role": "user", "content": f"Explain these news headlines in simple terms for a beginner trader:\n{news_text}"})
                ai_response = ask_nunno(ai_messages)
                assistant_entry["content"] = ai_response

        # MONTE CARLO
        elif ("monte carlo" in lower or "simulation" in lower) and simulate_trades:
            try:
                results = simulate_trades(num_simulations=1000)
                summary = monte_carlo_summary(results)
                assistant_entry["kind"] = "montecarlo"
                assistant_entry["content"] = summary
            except Exception as e:
                assistant_entry["content"] = f"Monte Carlo simulation error: {e}"

        # DEFAULT AI CHAT
        else:
            ai_messages = flatten_conversation_for_api(st.session_state.conversation)
            ai_messages.append({"role": "user", "content": prompt})
            ai_response = ask_nunno(ai_messages)
            assistant_entry["content"] = ai_response

        st.session_state.conversation.append(assistant_entry)
        st.session_state.conversation = manage_history_length(st.session_state.conversation)
        st.rerun()

with col2:
    st.subheader("Quick Info")
    
    # Module Status
    st.markdown("**🧩 Features**")
    if betterpredictormodule:
        st.success("✅ Predictions Available")
    else:
        st.warning("⚠️ Predictions Module Missing")
        
    if simulate_trades:
        st.success("✅ Monte Carlo Available")
    else:
        st.warning("⚠️ Monte Carlo Module Missing")
    
    # Upload status
    if st.session_state.uploaded_b64:
        st.success("📷 Chart Ready for Analysis")
    else:
        st.info("📷 No Chart Uploaded")
        
    st.markdown("---")
    st.markdown("**💡 Tips**")
    st.markdown("- Use specific coin names (BTC, ETH, ADA)")
    st.markdown("- Include timeframes (15m, 1h, 4h, 1d)")
    st.markdown("- Ask for predictions, tokenomics, or news")
    st.markdown("- Upload charts for technical analysis")
    
    # Chart Analysis Results Section
    if st.session_state.chart_analysis:
        st.markdown("---")
        st.markdown("**📈 Chart Analysis**")
        with st.expander("View Analysis", expanded=True):
            st.markdown(st.session_state.chart_analysis)
            if st.button("Clear Analysis", key="clear_analysis"):
                st.session_state.chart_analysis = None
                st.rerun()
