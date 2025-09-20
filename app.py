import os import re import json import hashlib import requests import pandas as pd import streamlit as st from bs4 import BeautifulSoup import google.generativeai as genai import yfinance as yf from datetime import datetime, timedelta

--- Configuration & Setup ---

Configure AI

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") if not api_key: st.error("❌ Missing Gemini API key. Please set it in your Streamlit secrets.") st.stop()

Configure the Gemini client

try: genai.configure(api_key=api_key) except Exception as e: st.error(f"❌ Failed to configure Gemini API: {e}") st.stop()

--- Utility Functions ---

def safe_get(data, key, default='N/A'): return data.get(key, default)

def format_currency(value): if isinstance(value, (int, float)): if value >= 1e12: return f"${value / 1e12:.2f}T" elif value >= 1e9: return f"${value / 1e9:.2f}B" elif value >= 1e6: return f"${value / 1e6:.2f}M" return f"${value:,.2f}" return value

--- Data Fetching Functions ---

@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...") def fetch_sec_earnings(ticker, quarters=4): filings_data = [] api_key = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY") if not api_key: st.error("❌ Missing SEC API key. Please set it in your Streamlit secrets.") return []

query_url = "https://api.sec-api.io"
payload = {
    "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
    "from": "0",
    "size": quarters,
    "sort": [{"filedAt": {"order": "desc"}}]
}
headers = {"Content-Type": "application/json", "Authorization": api_key}

try:
    query_response = requests.post(query_url, json=payload, headers=headers)
    query_response.raise_for_status()
    filings = query_response.json().get("filings", [])
except requests.exceptions.RequestException as e:
    st.error(f"❌ Failed to fetch filing URLs from sec-api.io: {e}")
    return []

if not filings:
    st.info(f"No 10-Q or 10-K filings found for {ticker}.")
    return []

for filing in filings:
    filing_url = filing.get("linkToFilingDetails")
    if not filing_url:
        continue

    converter_url = "https://api.sec-api.io/xbrl-to-json"

    # Debug log
    st.write(f"🔎 DEBUG: Trying filing URL → {filing_url}")

    try:
        xbrl_response = requests.get(
            converter_url, params={"url": filing_url}, headers={"Authorization": api_key}
        )
        st.write(f"🔎 DEBUG: Converter status = {xbrl_response.status_code}")
        xbrl_response.raise_for_status()
        data = xbrl_response.json()
    except Exception as e:
        st.warning(f"⚠️ Primary filing URL failed: {e}. Trying fallback...")
        # --- fallback: construct EDGAR XML URL ---
        try:
            accession_number = filing.get("accessionNo", "").replace("-", "")
            cik = filing.get("cik", "")
            period = filing.get("periodOfReport", "").replace("-", "")
            xbrl_fallback = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{ticker.lower()}-{period}_htm.xml"
            st.write(f"🔎 DEBUG: Trying fallback URL → {xbrl_fallback}")

            xbrl_response = requests.get(
                converter_url, params={"url": xbrl_fallback}, headers={"Authorization": api_key}
            )
            st.write(f"🔎 DEBUG: Fallback status = {xbrl_response.status_code}")
            xbrl_response.raise_for_status()
            data = xbrl_response.json()
        except Exception as e2:
            st.error(f"❌ Both primary and fallback XBRL fetch failed: {e2}")
            continue

    income_statements = data.get("incomeStatement", [])
    if not income_statements:
        st.write("⚠️ DEBUG: No income statements found in XBRL data.")
        continue

    last_statement = income_statements[-1]
    revenue = net_income = eps = None

    for fact in last_statement:
        if fact.get("concept") == "Revenues":
            revenue = fact.get("value")
        elif fact.get("concept") == "NetIncomeLoss":
            net_income = fact.get("value")
        elif fact.get("concept") == "EarningsPerShareDiluted":
            eps = fact.get("value")

    filings_data.append({
        'date': filing.get("filedAt"),
        'type': filing.get("formType"),
        'revenue': revenue,
        'eps': eps,
        'net_income': net_income
    })

return filings_data

2️⃣ Earnings Call Transcript Scraper (Simulated)

@st.cache_data(ttl=3600, show_spinner="🎙️ Fetching earnings call transcripts...") def fetch_earnings_transcripts(ticker, quarters=2): transcripts = [] sample_transcript = { 'date': '2024-10-30', 'quarter': 'Q3 2024', 'source': 'SeekingAlpha', 'ceo_comments': [ "We delivered strong results this quarter with revenue growth of 8% year-over-year", "Our new product line is gaining significant traction in the market", "We remain optimistic about our growth prospects for the remainder of the year" ], 'analyst_questions': [ "What are your expectations for margin expansion next quarter?", "How is the competitive landscape affecting your market share?", "Can you provide more details on your capital allocation strategy?" ], 'key_metrics_discussed': [ "User growth rate increased 12% quarter-over-quarter", "Gross margins improved to 68%, up from 65% last quarter", "Free cash flow generation remains strong at $8.2B" ] } transcripts.append(sample_transcript) if quarters > 1: old_transcript = sample_transcript.copy() old_transcript['date'] = '2024-07-31' old_transcript['quarter'] = 'Q2 2024' old_transcript['source'] = 'MotleyFool' transcripts.append(old_transcript) return transcripts[:quarters]

3️⃣ Analyst Reports & News Scraper (Simulated)

@st.cache_data(ttl=1800, show_spinner="📰 Collecting analyst reports...") def fetch_analyst_sentiment(ticker, days=30): return [ { 'date': '2024-11-01', 'firm': 'Goldman Sachs', 'rating': 'Buy', 'price_target': 180, 'headline': 'Strong Q3 results support positive outlook', 'key_points': [ 'Revenue beat expectations by 3%', 'Margin expansion ahead of schedule', 'Management guidance raised for FY2024' ] }, { 'date': '2024-10-31', 'firm': 'Morgan Stanley', 'rating': 'Overweight', 'price_target': 175, 'headline': 'Solid execution on strategic initiatives', 'key_points': [ 'Market share gains in key segments', 'Strong balance sheet provides flexibility', 'Well-positioned for economic uncertainty' ] } ]

4️⃣ Stock Price & Market Data

@st.cache_data(ttl=300, show_spinner="📈 Fetching market data...") def fetch_market_data(ticker, days=90): try: stock = yf.Ticker(ticker) info = stock.info end_date = datetime.now() start_date = end_date - timedelta(days=days) hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

if hist.empty:
        st.warning(f"⚠️ No market data found for {ticker} in the last {days} days.")
        return {}

    current_price = hist['Close'][-1]
    price_change_30d = ((current_price - hist['Close'][-30]) / hist['Close'][-30]) * 100
    avg_volume = hist['Volume'].mean()

    return {
        'current_price': current_price,
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('trailingPE', 0),
        'price_change_30d': price_change_30d,
        'avg_volume': avg_volume,
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'price_history': hist['Close'].tolist()[-30:],
        'volume_history': hist['Volume'].tolist()[-30:]
    }
except Exception as e:
    st.warning(f"⚠️ Market data fetch failed for {ticker}: {e}")
    return {}

5️⃣ AI-Powered Earnings Analysis

@st.cache_data(ttl=7200, show_spinner="🤖 Generating AI analysis...") def analyze_earnings_with_ai(ticker, sec_filings, transcripts, analyst_reports, market_data): try: filings_context = "\n".join([ f"  - {f.get('date', 'N/A')} ({f.get('type', 'N/A')}): Revenue: {format_currency(f.get('revenue'))}, EPS: ${f.get('eps', 'N/A')}, Net Income: {format_currency(f.get('net_income'))}" for f in sec_filings ]) if sec_filings else "No SEC filing data available."

transcript_context = "\n".join([
        f"  - {t.get('quarter', 'N/A')} ({t.get('date', 'N/A')}): CEO comments: {'; '.join(t.get('ceo_comments', []))}"
        for t in transcripts
    ]) if transcripts else "No transcript data available."

    analyst_context = "\n".join([
        f"  - {r.get('firm', 'N/A')}: Rating: {r.get('rating', 'N/A')}, Price Target: {format_currency(r.get('price_target'))}, Headline: {r.get('headline', 'N/A')}"
        for r in analyst_reports
    ]) if analyst_reports else "No analyst report data available."

    market_context = f"""
    Current Price: {format_currency(market_data.get('current_price', 0))}
    30-day Price Change: {market_data.get('price_change_30d', 0):.1f}%
    P/E Ratio: {market_data.get('pe_ratio', 0):.1f}
    Market Cap: {format_currency(market_data.get('market_cap', 0))}
    Sector: {market_data.get('sector', 'Unknown')}
    Industry: {market_data.get('industry', 'Unknown')}
    """

    prompt = f"""
    You are a highly experienced and professional financial analyst. Your task is to provide a comprehensive, structured investment analysis of {ticker}'s recent earnings.
    Output must be a valid JSON object with specific keys (overall_grade, investment_thesis, financial_health, key_strengths, key_risks, analyst_consensus, earnings_surprises, competitive_position, valuation_assessment, price_catalysts, recommendation, risk_level).

    RECENT EARNINGS DATA:
    {filings_context}

    MANAGEMENT COMMENTARY:
    {transcript_context}

    ANALYST COVERAGE:
    {analyst_context}

    MARKET DATA:
    {market_context}
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
    return json.loads(response.text)
except Exception as e:
    st.error(f"⚠️ AI analysis failed: {e}")
    return {}

6️⃣ Streamlit Dashboard

def render_dashboard(): st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide") st.title("📊 AI-Powered Earnings Intelligence Platform") st.markdown("A comprehensive analysis combining SEC filings, earnings calls, analyst reports, and market data.")

st.subheader("Enter Ticker Symbol")
col1, col2 = st.columns([0.7, 0.3])
with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").strip().upper()
with col2:
    quarters = st.selectbox("Quarters to analyze", [1, 2, 3, 4], index=1)

analyze_button = st.button("🔍 Analyze Earnings", type="primary", use_container_width=True)
st.markdown("---")

if analyze_button and ticker:
    with st.spinner("🔄 Collecting earnings intelligence... This may take a moment."):
        sec_data = fetch_sec_earnings(ticker, quarters)
        transcripts = fetch_earnings_transcripts(ticker, quarters)
        analyst_data = fetch_analyst_sentiment(ticker)
        market_data = fetch_market_data(ticker)

    st.subheader("Source Data Overview")
    tab1, tab2, tab3, tab4 = st.tabs(["📋 SEC Filings", "🎙️ Earnings Calls", "📰 Analyst Reports", "📈 Market Data"])

    with tab1:
        st.markdown("#### SEC Filing Summary")
        if sec_data:
            df = pd.DataFrame(sec_data)
            df['revenue'] = df['revenue'].apply(lambda x: format_currency(x) if x is not None else 'N/A')
            df['net_income'] = df['net_income'].apply(lambda x: format_currency(x) if x is not None else 'N/A')
            st.dataframe(df, use_container_width=True)
        else:
            st.info(f"No SEC filings found for {ticker}.")

    with tab2:
        st.markdown("#### Earnings Call Highlights")
        if transcripts:
            for transcript in transcripts:
                with st.expander(f"**{safe_get(transcript, 'quarter')}** - {safe_get(transcript, 'date')}"):
                    st.write("**CEO Key Comments:**")
                    for comment in safe_get(transcript, 'ceo_comments', []):
                        st.write(f"• {comment}")
                    st.write("**Key Metrics Discussed:**")
                    for metric in safe_get(transcript, 'key_metrics_discussed', []):
                        st.write(f"• {metric}")
        else:
            st.info(f"No earnings call transcripts found for {ticker}.")

    with tab3:
        st.markdown("#### Analyst Coverage")
        if analyst_data:
            for report in analyst_data:
                with st.expander(f"**{safe_get(report, 'firm')}** - {safe_get(report, 'rating')} | Price Target: {format_currency(safe_get(report, 'price_target'))}"):
                    st.write(f"**{safe_get(report, 'headline')}**")
                    for point in safe_get(report, 'key_points', []):
                        st.write(f"• {point}")
        else:
            st.info(f"No analyst reports found for {ticker}.")

    with tab4:
        st.markdown("#### Key Market Performance Metrics")
        if market_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", format_currency(safe_get(market_data, 'current_price', 0)))
            with col2:
                change_30d = safe_get(market_data, 'price_change_30d', 0)
                st.metric("30-Day Change", f"{change_30d:.1f}%", delta=f"{change_30d:.1f}%")
            with col3:
                st.metric("P/E Ratio", f"{safe_get(market_data, 'pe_ratio', 0):.1f}")
            with

