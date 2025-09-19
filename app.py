# earnings_analyzer.py
import os
import re
import json
import hashlib
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai
import yfinance as yf
from datetime import datetime, timedelta

# --- Configuration & Setup ---
# Configure AI
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set it in your Streamlit secrets.")
    st.stop()

# Configure the Gemini client
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()

# --- Utility Functions ---
def safe_get(data, key, default='N/A'):
    """Safely get a value from a dictionary with a default."""
    return data.get(key, default)

def format_currency(value):
    """Formats a number into a readable currency string."""
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        elif value >= 1e9:
            return f"${value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.2f}"
    return value

# --- Data Fetching Functions ---
# 1️⃣ SEC EDGAR Earnings Scraper
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4):
    """
    Fetch recent financial data directly from SEC XBRL APIs.
    This is more reliable than scraping raw filings.
    """
    filings_data = []

    # 1. Look up CIK using a Ticker-to-CIK mapping
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': 'Financial Analyzer App/1.0 (info@example.com)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        ticker_data = response.json()
        
        if not isinstance(ticker_data, list):
            st.error("❌ SEC API returned unexpected data format for ticker lookup.")
            return []

        cik = None
        for company in ticker_data:
            if isinstance(company, dict) and company.get('ticker') == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            st.warning(f"⚠️ CIK for {ticker} not found.")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch CIK for {ticker}: {e}")
        return []
    except json.JSONDecodeError:
        st.error("❌ Failed to decode JSON from SEC ticker lookup API.")
        return []

    # 2. Get company facts using the XBRL API
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {'User-Agent': 'Financial Analyzer App/1.0 (info@example.com)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        company_facts = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch company facts for CIK {cik}: {e}")
        return []
    except json.JSONDecodeError:
        st.error("❌ Failed to decode JSON from SEC company facts API.")
        return []

    # 3. Extract and parse relevant financial data
    try:
        us_gaap_data = company_facts.get('facts', {}).get('us-gaap', {})
        
        revenue_facts = us_gaap_data.get('Revenues', {}).get('units', {}).get('USD', [])
        eps_facts = us_gaap_data.get('EarningsPerShareDiluted', {}).get('units', {}).get('USD-per-share', [])
        net_income_facts = us_gaap_data.get('NetIncomeLoss', {}).get('units', {}).get('USD', [])
        
        # Sort facts by end date to get the most recent ones
        all_facts = sorted(
            [f for f in revenue_facts + eps_facts + net_income_facts if f.get('form') in ['10-Q', '10-K']],
            key=lambda x: datetime.strptime(x['end'], '%Y-%m-%d'),
            reverse=True
        )

        seen_periods = set()
        for fact in all_facts:
            period_end = fact.get('end')
            form_type = fact.get('form')
            
            if period_end not in seen_periods and len(filings_data) < quarters:
                filing = {
                    'date': period_end,
                    'type': form_type,
                    'revenue': None,
                    'eps': None,
                    'net_income': None
                }
                
                # Find matching revenue, EPS, and net income for this period
                for rev_fact in revenue_facts:
                    if rev_fact.get('end') == period_end:
                        filing['revenue'] = rev_fact.get('val')
                        break
                for eps_fact in eps_facts:
                    if eps_fact.get('end') == period_end:
                        filing['eps'] = eps_fact.get('val')
                        break
                for ni_fact in net_income_facts:
                    if ni_fact.get('end') == period_end:
                        filing['net_income'] = ni_fact.get('val')
                        break
                
                filings_data.append(filing)
                seen_periods.add(period_end)
    
    except Exception as e:
        st.error(f"⚠️ Error parsing financial data for {ticker}: {e}")
        return []

    return filings_data

# 2️⃣ Earnings Call Transcript Scraper (Simulated)
@st.cache_data(ttl=3600, show_spinner="🎙️ Fetching earnings call transcripts...")
def fetch_earnings_transcripts(ticker, quarters=2):
    """
    Scrape earnings call transcripts from SeekingAlpha, MotleyFool, etc.
    NOTE: This is a simulated function due to the difficulty of scraping these sites reliably.
    """
    transcripts = []
    
    # For demo purposes, return sample transcript data
    sample_transcript = {
        'date': '2024-10-30',
        'quarter': 'Q3 2024',
        'source': 'SeekingAlpha',
        'ceo_comments': [
            "We delivered strong results this quarter with revenue growth of 8% year-over-year",
            "Our new product line is gaining significant traction in the market",
            "We remain optimistic about our growth prospects for the remainder of the year"
        ],
        'analyst_questions': [
            "What are your expectations for margin expansion next quarter?",
            "How is the competitive landscape affecting your market share?",
            "Can you provide more details on your capital allocation strategy?"
        ],
        'key_metrics_discussed': [
            "User growth rate increased 12% quarter-over-quarter",
            "Gross margins improved to 68%, up from 65% last quarter",
            "Free cash flow generation remains strong at $8.2B"
        ]
    }
    
    transcripts.append(sample_transcript)
    if quarters > 1:
        # Add a second, older sample for variety
        old_transcript = sample_transcript.copy()
        old_transcript['date'] = '2024-07-31'
        old_transcript['quarter'] = 'Q2 2024'
        old_transcript['source'] = 'MotleyFool'
        transcripts.append(old_transcript)
    
    return transcripts[:quarters]

# 3️⃣ Analyst Reports & News Scraper (Simulated)
@st.cache_data(ttl=1800, show_spinner="📰 Collecting analyst reports...")
def fetch_analyst_sentiment(ticker, days=30):
    """Fetch recent analyst reports and news sentiment (simulated)"""
    # Sample analyst data
    sample_reports = [
        {
            'date': '2024-11-01',
            'firm': 'Goldman Sachs',
            'rating': 'Buy',
            'price_target': 180,
            'headline': 'Strong Q3 results support positive outlook',
            'key_points': [
                'Revenue beat expectations by 3%',
                'Margin expansion ahead of schedule',
                'Management guidance raised for FY2024'
            ]
        },
        {
            'date': '2024-10-31',
            'firm': 'Morgan Stanley',
            'rating': 'Overweight',
            'price_target': 175,
            'headline': 'Solid execution on strategic initiatives',
            'key_points': [
                'Market share gains in key segments',
                'Strong balance sheet provides flexibility',
                'Well-positioned for economic uncertainty'
            ]
        }
    ]
    return sample_reports

# 4️⃣ Stock Price & Market Data
@st.cache_data(ttl=300, show_spinner="📈 Fetching market data...")
def fetch_market_data(ticker, days=90):
    """
    Get stock price, volume, and market reaction data using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist.empty:
            st.warning(f"⚠️ No market data found for {ticker} in the last {days} days.")
            return {}

        current_price = hist['Close'][-1]
        price_change_30d = ((current_price - hist['Close'][-30]) / hist['Close'][-30]) * 100
        avg_volume = hist['Volume'].mean()

        market_data = {
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
        return market_data
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed for {ticker}: {e}")
        return {}

# 5️⃣ AI-Powered Earnings Analysis
@st.cache_data(ttl=7200, show_spinner="🤖 Generating AI analysis...")
def analyze_earnings_with_ai(ticker, sec_filings, transcripts, analyst_reports, market_data):
    """
    Comprehensive AI analysis of all earnings data using a structured prompt.
    """
    try:
        # Prepare data contexts with rich detail
        filings_context = "\n".join([
            f"  - {f.get('period', 'N/A')} ({f.get('type', 'N/A')}): Revenue: {format_currency(f.get('revenue'))}, EPS: ${f.get('eps', 'N/A')}, Net Income: {format_currency(f.get('net_income'))}"
            for f in sec_filings
        ]) if sec_filings else "No SEC filing data available."

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
        You are a highly experienced and professional financial analyst. Your task is to provide a comprehensive, structured investment analysis of {ticker}'s recent earnings, based on the provided data. Synthesize the information from all sources to create a coherent narrative.

        The analysis must be provided in a single JSON object.

        **Instructions for the JSON structure:**
        - **overall_grade**: A letter grade (A-F) reflecting the company's overall performance.
        - **investment_thesis**: A clear, concise 2-3 sentence summary of the core investment argument (e.g., bull or bear case).
        - **financial_health**: A nested object with specific, detailed analysis of key financial metrics.
            - **revenue_trend**: A short paragraph analyzing the revenue trajectory and key drivers.
            - **profitability**: A short paragraph assessing margin trends, net income, and efficiency.
            - **balance_sheet**: A short paragraph evaluating the company's financial strength, liquidity, and debt levels.
        - **key_strengths**: A list of 3-5 specific, bulleted strengths.
        - **key_risks**: A list of 3-5 specific, bulleted risks.
        - **analyst_consensus**: A nested object summarizing analyst sentiment.
            - **avg_rating**: The overall consensus (e.g., "Strong Buy", "Hold", "Underperform").
            - **price_target_range**: A price target range based on the analyst data (e.g., "$XXX - $XXX").
            - **sentiment_shift**: An analysis of recent changes in analyst sentiment (e.g., "Upgraded", "Downgraded", "Stable").
        - **earnings_surprises**: A nested object detailing how performance compared to expectations.
            - **revenue_beat_miss**: "Beat" or "Miss" with a percentage figure.
            - **eps_beat_miss**: "Beat" or "Miss" with the dollar amount.
            - **guidance_reaction**: "Raised", "Lowered", or "Maintained" with context.
        - **competitive_position**: A short paragraph on the company's market standing and competitive advantages.
        - **valuation_assessment**: A short paragraph explaining whether the stock is "Overvalued", "Fairly Valued", or "Undervalued" with supporting reasoning (e.g., P/E ratio, growth).
        - **price_catalysts**: A list of 2-3 upcoming events that could affect the stock price.
        - **recommendation**: A clear recommendation: "Buy", "Hold", or "Sell", with a brief justification.
        - **risk_level**: "Low", "Medium", or "High" risk assessment.
        
        **Source Data for Analysis:**

        RECENT EARNINGS DATA:
        {filings_context}

        MANAGEMENT COMMENTARY:
        {transcript_context}

        ANALYST COVERAGE:
        {analyst_context}

        MARKET DATA:
        {market_context}
        
        Generate the full, valid JSON object now. Do not include any text before or after the JSON.
        """

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ AI response was not valid JSON: {e}. Raw response: {response.text}")
        return {}
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}

# 6️⃣ Streamlit Dashboard
def render_dashboard():
    """Renders the Streamlit UI."""
    st.set_page_config(
        page_title="Earnings Intelligence",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 AI-Powered Earnings Intelligence Platform")
    st.markdown("A comprehensive analysis combining SEC filings, earnings calls, analyst reports, and market data.")

    # --- Input Section ---
    st.subheader("Enter Ticker Symbol")
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").strip().upper()
    with col2:
        quarters = st.selectbox("Quarters to analyze", [1, 2, 3, 4], index=1)

    analyze_button = st.button("🔍 Analyze Earnings", type="primary", use_container_width=True)
    st.markdown("---")

    if analyze_button and ticker:
        # --- Data Fetching ---
        with st.spinner("🔄 Collecting earnings intelligence... This may take a moment."):
            sec_data = fetch_sec_earnings(ticker, quarters)
            transcripts = fetch_earnings_transcripts(ticker, quarters)
            analyst_data = fetch_analyst_sentiment(ticker)
            market_data = fetch_market_data(ticker)

        # --- Display Raw Data in Tabs ---
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
                with col4:
                    st.metric("Market Cap", format_currency(safe_get(market_data, 'market_cap', 0)))
            else:
                st.info(f"No market data found for {ticker}.")

        # --- AI Analysis Section ---
        st.markdown("---")
        st.header("🤖 AI Investment Analysis")
        
        with st.spinner("🧠 Generating comprehensive analysis..."):
            analysis = analyze_earnings_with_ai(ticker, sec_data, transcripts, analyst_data, market_data)

        if analysis:
            display_ai_analysis(analysis)
        else:
            st.warning("Could not generate a full AI analysis. Please check the data sources or try a different ticker.")

    st.markdown("---")
    st.caption("Data Sources: SEC EDGAR, Earnings Call Transcripts, Analyst Reports, Yahoo Finance | AI Analysis: Google Gemini")

def display_ai_analysis(analysis):
    """Helper function to display the AI analysis in a structured way."""
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade = safe_get(analysis, 'overall_grade')
        st.markdown(f"""
        <div style="background-color: #444444; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h3 style="margin: 0; color: #90ee90;">📊 Overall Grade</h3>
            <h1 style="margin: 0.5rem 0 0 0; color: #90ee90; font-size: 3rem;">{grade}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        recommendation = safe_get(analysis, 'recommendation')
        color = "#a3d900" if "Buy" in recommendation else "#fdd835" if "Hold" in recommendation else "#f44336"
        st.markdown(f"""
        <div style="background-color: #444444; padding: 1rem; border-radius: 0.5rem; border-
