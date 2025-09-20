import os
import json
import requests
import pandas as pd
import streamlit as st
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
def safe_get(data, key, default="N/A"):
    """Safely get a value from a dictionary with a default."""
    return data.get(key, default) if isinstance(data, dict) else default


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
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4, debug_logs=True):
    """
    Fetch recent financial data using sec-api.io Query API + XBRL-to-JSON converter.
    If converter fails on the filing URL, try an EDGAR XML fallback.
    Returns a list of dicts with date, period/type, revenue, eps, net_income.
    """
    filings_data = []

    sec_api_key = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")
    if not sec_api_key:
        st.error("❌ Missing SEC API key. Please set it in your Streamlit secrets.")
        return []

    query_url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    headers = {"Content-Type": "application/json", "Authorization": sec_api_key}

    try:
        query_response = requests.post(query_url, json=payload, headers=headers, timeout=30)
        query_response.raise_for_status()
        filings = query_response.json().get("filings", [])
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch filing URLs from sec-api.io: {e}")
        return []

    if not filings:
        if debug_logs:
            st.info(f"No 10-Q or 10-K filings found for {ticker}.")
        return []

    converter_url = "https://api.sec-api.io/xbrl-to-json"

    for filing in filings:
        filing_url = filing.get("linkToFilingDetails")
        if not filing_url:
            if debug_logs:
                st.write("⚠️ DEBUG: Filing has no 'linkToFilingDetails', skipping.")
            continue

        data = None

        # Try primary converter call with the filing details URL
        if debug_logs:
            st.write(f"🔎 DEBUG: Trying converter for: {filing_url}")
        try:
            r = requests.get(converter_url, params={"url": filing_url}, headers={"Authorization": sec_api_key}, timeout=30)
            if debug_logs:
                st.write(f"🔎 DEBUG: Converter status {r.status_code} (primary)")
            r.raise_for_status()
            data = r.json()
        except Exception as e_primary:
            if debug_logs:
                st.warning(f"⚠️ Primary converter call failed for filing URL: {e_primary}")
            # Attempt EDGAR fallback
            accession_no = filing.get("accessionNo", "").replace("-", "")
            cik = filing.get("cik", "")
            period = filing.get("periodOfReport", "")  # e.g., YYYY-MM-DD
            # Build a couple of reasonable fallback patterns
            fallback_attempts = []
            if cik and accession_no:
                # Common fallback pattern: <accession>-xbrl.xml (some filings)
                fallback_attempts.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{accession_no}-xbrl.xml")
                # Another common pattern includes ticker + period
                if period and ticker:
                    p = period.replace("-", "")
                    fallback_attempts.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{ticker.lower()}-{p}_htm.xml")
                    fallback_attempts.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{ticker.lower()}-{p}.xml")
            else:
                if debug_logs:
                    st.write("⚠️ DEBUG: Missing cik/accession_no for fallback.")

            # Try fallbacks in order
            for fb in fallback_attempts:
                try:
                    if debug_logs:
                        st.write(f"🔎 DEBUG: Trying fallback URL: {fb}")
                    r2 = requests.get(converter_url, params={"url": fb}, headers={"Authorization": sec_api_key}, timeout=30)
                    if debug_logs:
                        st.write(f"🔎 DEBUG: Converter status {r2.status_code} (fallback)")
                    r2.raise_for_status()
                    data = r2.json()
                    break
                except Exception as e_fb:
                    if debug_logs:
                        st.write(f"⚠️ DEBUG: Fallback failed: {e_fb}")
                    data = None

        if not data:
            if debug_logs:
                st.warning("⚠️ DEBUG: No XBRL JSON obtained for this filing; skipping.")
            continue

        # Extract financials from returned JSON
        # Many sec-api xbrl-to-json responses include 'incomeStatement' or 'IncomeStatements'
        income_statements = data.get("incomeStatement") or data.get("IncomeStatements") or data.get("IncomeStatement")
        if not income_statements:
            # Try 'facts' approach
            facts = data.get("facts", {})
            revenue = None
            net_income = None
            eps = None
            if facts:
                for concept, entry in facts.items():
                    lower = concept.lower()
                    try:
                        # entry can be dict with 'value' or list of items
                        value = entry.get("value") if isinstance(entry, dict) else (entry[0].get("value") if entry else None)
                    except Exception:
                        value = None
                    if "revenue" in lower and revenue is None:
                        revenue = value
                    if "netincome" in lower and net_income is None:
                        net_income = value
                    if ("earningspershare" in lower or "eps" in lower) and eps is None:
                        eps = value
            else:
                if debug_logs:
                    st.write("⚠️ DEBUG: No 'incomeStatement' or relevant 'facts' in returned data.")
            filing_data = {
                "date": filing.get("filedAt"),
                "period": filing.get("periodOfReport"),
                "type": filing.get("formType"),
                "revenue": revenue,
                "eps": eps,
                "net_income": net_income,
                "accession_number": filing.get("accessionNo"),
            }
            filings_data.append(filing_data)
            continue

        # income_statements may be a list of periods or nested
        try:
            # If it's a list of lists/dicts, try to pick the most recent; fallback to last
            if isinstance(income_statements, list) and len(income_statements) > 0:
                latest = income_statements[0] if isinstance(income_statements[0], dict) else income_statements[-1]
            else:
                latest = income_statements
        except Exception:
            latest = income_statements

        revenue = None
        net_income = None
        eps = None

        # latest might be a list of facts or dict of named fields
        if isinstance(latest, dict):
            # look for common keys
            for key in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomer", "TotalRevenue"]:
                if key in latest:
                    revenue = latest.get(key)
                    break
            for key in ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]:
                if key in latest:
                    net_income = latest.get(key)
                    break
            for key in ["EarningsPerShareDiluted", "EarningsPerShareBasic", "EarningsPerShare"]:
                if key in latest:
                    eps = latest.get(key)
                    break
        elif isinstance(latest, list):
            # list of fact dicts
            for fact in reversed(latest):  # search from last entries
                concept = fact.get("concept", "") if isinstance(fact, dict) else ""
                if concept in ("Revenues",) and revenue is None:
                    revenue = fact.get("value")
                if concept in ("NetIncomeLoss",) and net_income is None:
                    net_income = fact.get("value")
                if concept in ("EarningsPerShareDiluted",) and eps is None:
                    eps = fact.get("value")

        filings_data.append(
            {
                "date": filing.get("filedAt"),
                "period": filing.get("periodOfReport"),
                "type": filing.get("formType"),
                "revenue": revenue,
                "eps": eps,
                "net_income": net_income,
                "accession_number": filing.get("accessionNo"),
            }
        )

    return filings_data


# 2️⃣ Earnings Call Transcript Scraper (Simulated)
@st.cache_data(ttl=3600, show_spinner="🎙️ Fetching earnings call transcripts...")
def fetch_earnings_transcripts(ticker, quarters=2):
    """
    Simulated transcripts (real scraping omitted). Returns sample structured transcripts.
    """
    transcripts = []
    sample_transcript = {
        "date": "2024-10-30",
        "quarter": "Q3 2024",
        "source": "SeekingAlpha",
        "ceo_comments": [
            "We delivered strong results this quarter with revenue growth of 8% year-over-year",
            "Our new product line is gaining significant traction in the market",
            "We remain optimistic about our growth prospects for the remainder of the year",
        ],
        "analyst_questions": [
            "What are your expectations for margin expansion next quarter?",
            "How is the competitive landscape affecting your market share?",
            "Can you provide more details on your capital allocation strategy?",
        ],
        "key_metrics_discussed": [
            "User growth rate increased 12% quarter-over-quarter",
            "Gross margins improved to 68%, up from 65% last quarter",
            "Free cash flow generation remains strong at $8.2B",
        ],
    }
    transcripts.append(sample_transcript)
    if quarters > 1:
        old = sample_transcript.copy()
        old["date"] = "2024-07-31"
        old["quarter"] = "Q2 2024"
        old["source"] = "MotleyFool"
        transcripts.append(old)
    return transcripts[:quarters]


# 3️⃣ Analyst Reports & News Scraper (Simulated)
@st.cache_data(ttl=1800, show_spinner="📰 Collecting analyst reports...")
def fetch_analyst_sentiment(ticker, days=30):
    """
    Simulated analyst report list for demo purposes.
    """
    return [
        {
            "date": "2024-11-01",
            "firm": "Goldman Sachs",
            "rating": "Buy",
            "price_target": 180,
            "headline": "Strong Q3 results support positive outlook",
            "key_points": [
                "Revenue beat expectations by 3%",
                "Margin expansion ahead of schedule",
                "Management guidance raised for FY2024",
            ],
        },
        {
            "date": "2024-10-31",
            "firm": "Morgan Stanley",
            "rating": "Overweight",
            "price_target": 175,
            "headline": "Solid execution on strategic initiatives",
            "key_points": [
                "Market share gains in key segments",
                "Strong balance sheet provides flexibility",
                "Well-positioned for economic uncertainty",
            ],
        },
    ]


# 4️⃣ Stock Price & Market Data
@st.cache_data(ttl=300, show_spinner="📈 Fetching market data...")
def fetch_market_data(ticker, days=90):
    """
    Use yfinance to get price history and simple metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info if hasattr(stock, "info") else {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if hist.empty:
            st.warning(f"⚠️ No market data found for {ticker} in the last {days} days.")
            return {}

        current_price = float(hist["Close"].iloc[-1])
        price_change_30d = (
            (current_price - float(hist["Close"].iloc[-30])) / float(hist["Close"].iloc[-30]) * 100
            if len(hist) >= 30 else 0.0
        )
        avg_volume = float(hist["Volume"].mean()) if "Volume" in hist.columns else 0.0

        return {
            "current_price": current_price,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "price_change_30d": price_change_30d,
            "avg_volume": avg_volume,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price_history": hist["Close"].tolist()[-30:],
            "volume_history": hist["Volume"].tolist()[-30:] if "Volume" in hist.columns else [],
        }
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed for {ticker}: {e}")
        return {}


# 5️⃣ AI-Powered Earnings Analysis
@st.cache_data(ttl=7200, show_spinner="🤖 Generating AI analysis...")
def analyze_earnings_with_ai(ticker, sec_filings, transcripts, analyst_reports, market_data):
    """
    Ask Gemini to synthesize a JSON investment analysis using the collected data.
    """
    try:
        filings_context = "\n".join(
            [
                f"  - {f.get('period', f.get('date','N/A'))} ({f.get('type', 'N/A')}): Revenue: {format_currency(f.get('revenue'))}, EPS: {f.get('eps', 'N/A')}, Net Income: {format_currency(f.get('net_income'))}"
                for f in sec_filings
            ]
        ) if sec_filings else "No SEC filing data available."

        transcript_context = "\n".join(
            [
                f"  - {t.get('quarter','N/A')} ({t.get('date','N/A')}): CEO comments: {'; '.join(t.get('ceo_comments', []))}"
                for t in transcripts
            ]
        ) if transcripts else "No transcript data available."

        analyst_context = "\n".join(
            [
                f"  - {r.get('firm', 'N/A')}: Rating: {r.get('rating', 'N/A')}, Price Target: {format_currency(r.get('price_target'))}, Headline: {r.get('headline', 'N/A')}"
                for r in analyst_reports
            ]
        ) if analyst_reports else "No analyst report data available."

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

The analysis must be provided in a single JSON object with these keys:
overall_grade, investment_thesis, financial_health (with revenue_trend, profitability, balance_sheet),
key_strengths (list), key_risks (list), analyst_consensus (avg_rating, price_target_range, sentiment_shift),
earnings_surprises (revenue_beat_miss, eps_beat_miss, guidance_reaction),
competitive_position, valuation_assessment, price_catalysts (list), recommendation, risk_level.

RECENT EARNINGS DATA:
{filings_context}

MANAGEMENT COMMENTARY:
{transcript_context}

ANALYST COVERAGE:
{analyst_context}

MARKET DATA:
{market_context}

Return only a valid JSON object — no extra prose.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
        )

        # response.text should contain JSON
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ AI response was not valid JSON: {e}. Raw response: {getattr(response,'text', 'NO RESPONSE')}")
        return {}
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}


# 6️⃣ Streamlit Dashboard
def render_dashboard():
    st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Earnings Intelligence Platform")
    st.markdown("A comprehensive analysis combining SEC filings, earnings calls, analyst reports, and market data.")
    st.markdown("Use this tool to fetch XBRL-based financials, simulated transcripts/analyst notes, market data, and an AI synthesis.")

    # Debug toggle for logs
    debug_logs = st.checkbox("Show SEC debug logs", value=True)

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
        with st.spinner("🔄 Collecting earnings intelligence... This may take a moment."):
            sec_data = fetch_sec_earnings(ticker, quarters, debug_logs=debug_logs)
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
                # Ensure columns exist
                if "revenue" in df.columns:
                    df["revenue"] = df["revenue"].apply(lambda x: format_currency(x) if x is not None else "N/A")
                if "net_income" in df.columns:
                    df["net_income"] = df["net_income"].apply(lambda x: format_currency(x) if x is not None else "N/A")
                st.dataframe(df, use_container_width=True)
            else:
                st.info(f"No SEC filings found for {ticker}.")

        with tab2:
            st.markdown("#### Earnings Call Highlights")
            if transcripts:
                for transcript in transcripts:
                    with st.expander(f"**{safe_get(transcript,'quarter')}** - {safe_get(transcript,'date')}"):
                        st.write("**CEO Key Comments:**")
                        for comment in safe_get(transcript, "ceo_comments", []):
                            st.write(f"• {comment}")
                        st.write("**Key Metrics Discussed:**")
                        for metric in safe_get(transcript, "key_metrics_discussed", []):
                            st.write(f"• {metric}")
            else:
                st.info(f"No earnings call transcripts found for {ticker}.")

        with tab3:
            st.markdown("#### Analyst Coverage")
            if analyst_data:
                for report in analyst_data:
                    with st.expander(f"**{safe_get(report,'firm')}** - {safe_get(report,'rating')} | Price Target: {format_currency(safe_get(report,'price_target'))}"):
                        st.write(f"**{safe_get(report,'headline')}**")
                        for point in safe_get(report, "key_points", []):
                            st.write(f"• {point}")
            else:
                st.info(f"No analyst reports found for {ticker}.")

        with tab4:
            st.markdown("#### Key Market Performance Metrics")
            if market_data:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Current Price", format_currency(safe_get(market_data, "current_price", 0)))
                with c2:
                    change_30d = safe_get(market_data, "price_change_30d", 0)
                    st.metric("30-Day Change", f"{change_30d:.1f}%", delta=f"{change_30d:.1f}%")
                with c3:
                    st.metric("P/E Ratio", f"{safe_get(market_data, 'pe_ratio', 0):.1f}")
                with c4:
                    st.metric("Market Cap", format_currency(safe_get(market_data, "market_cap", 0)))
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
    st.caption("Data Sources: SEC EDGAR / sec-api.io, Earnings Call Transcripts (sample), Analyst Reports (sample), Yahoo Finance | AI: Google Gemini")


def display_ai_analysis(analysis):
    """Helper function to display the AI analysis in a structured way."""
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade = safe_get(analysis, "overall_grade", "N/A")
        st.markdown(
            f"""
        <div style="background-color: #444444; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h3 style="margin: 0; color: #90ee90;">📊 Overall Grade</h3>
            <h1 style="margin: 0.5rem 0 0 0; color: #90ee90; font-size: 3rem;">{grade}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        recommendation = safe_get(analysis, "recommendation", "N/A")
        color = "#a3d900" if "Buy" in recommendation else "#fdd835" if "Hold" in recommendation else "#f44336"
        st.markdown(
            f"""
        <div style="background-color: #444444; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
            <h4 style="margin: 0; color: {color};">💡 Recommendation</h4>
            <p style="margin: 0.5rem 0 0 0; font-weight: 600; color: #ffffff;">{recommendation}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        risk_level = safe_get(analysis, "risk_level", "N/A")
        risk_colors = {"Low": "#a3d900", "Medium": "#fdd835", "High": "#f44336"}
        risk_color = risk_colors.get(risk_level, "#bbbbbb")
        st.markdown(
            f"""
        <div style="background-color: #444444; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {risk_color};">
            <h4 style="margin: 0; color: {risk_color};">⚠️ Risk Level</h4>
            <p style="margin: 0.5rem 0 0 0; font-weight: 600; color: #ffffff;">{risk_level}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        valuation = safe_get(analysis, "valuation_assessment", "N/A")
        display_valuation = (valuation[:120] + "...") if isinstance(valuation, str) and len(valuation) > 120 else valuation
        st.markdown(
            f"""
        <div style="background-color: #444444; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #00bcd4;">
            <h4 style="margin: 0; color: #00bcd4;">💰 Valuation</h4>
            <p style="margin: 0.5rem 0 0 0; font-weight: 600; color: #ffffff;">{display_valuation}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Investment thesis
    st.markdown("### 🎯 Investment Thesis")
    st.info(safe_get(analysis, "investment_thesis", "N/A"))

    # Detailed analysis
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Key Strengths")
        for strength in safe_get(analysis, "key_strengths", []):
            st.success(f"✓ {strength}")
        st.markdown("### 📈 Earnings Performance")
        earnings = safe_get(analysis, "earnings_surprises", {})
        st.write(f"**Revenue:** {safe_get(earnings, 'revenue_beat_miss', 'N/A')}")
        st.write(f"**EPS:** {safe_get(earnings, 'eps_beat_miss', 'N/A')}")
        st.write(f"**Guidance:** {safe_get(earnings, 'guidance_reaction', 'N/A')}")
    with c2:
        st.markdown("### ⚠️ Key Risks")
        for risk in safe_get(analysis, "key_risks", []):
            st.error(f"✗ {risk}")
        st.markdown("### 🔮 Price Catalysts")
        for catalyst in safe_get(analysis, "price_catalysts", []):
            st.info(f"• {catalyst}")

    # Financial health and competitive position
    st.markdown("### 💼 Financial Health & Competitive Position")
    financial_health = safe_get(analysis, "financial_health", {})
    st.write(f"**Revenue Trend:** {safe_get(financial_health, 'revenue_trend', 'N/A')}")
    st.write(f"**Profitability:** {safe_get(financial_health, 'profitability', 'N/A')}")
    st.write(f"**Balance Sheet:** {safe_get(financial_health, 'balance_sheet', 'N/A')}")
    st.write(f"**Competitive Position:** {safe_get(analysis, 'competitive_position', 'N/A')}")
    st.write(f"**Valuation:** {safe_get(analysis, 'valuation_assessment', 'N/A')}")

    # Analyst consensus
    st.markdown("### 👥 Analyst Consensus")
    consensus = safe_get(analysis, "analyst_consensus", {})
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric("Average Rating", safe_get(consensus, "avg_rating", "N/A"))
    with cc2:
        st.metric("Price Target Range", safe_get(consensus, "price_target_range", "N/A"))
    with cc3:
        st.write("**Sentiment Shift**")
        st.write(safe_get(consensus, "sentiment_shift", "N/A"))


if __name__ == "__main__":
    render_dashboard()