import os
import json
import requests
import pandas as pd
import streamlit as st
import google.generativeai as genai
import yfinance as yf
from datetime import datetime, timedelta

# --- Configuration & Setup ---
# Gemini API key (for AI)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Missing Gemini API key. Please set it in your Streamlit secrets or env.")
    st.stop()

# SEC API key (for sec-api.io)
SEC_API_KEY = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")
# SEC key is optional: function will warn when missing.

# Configure Gemini client
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()


# --- Utility Functions ---
def safe_get(data, key, default="N/A"):
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def format_currency(value):
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        if value >= 1e9:
            return f"${value / 1e9:.2f}B"
        if value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.2f}"
    return value


# --- SEC filings fetcher (robust, uses linkToXbrl when available) ---
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker: str, quarters: int = 4, debug: bool = False):
    """
    Fetch recent 10-Q/10-K filings for `ticker` via sec-api.io query,
    then use linkToXbrl when present to call sec-api.io/xbrl-to-json.
    Returns list of filings with basic metrics (revenue, eps, net_income).
    """
    filings_data = []

    sec_key = SEC_API_KEY
    if not sec_key:
        if debug:
            st.warning("⚠️ SEC_API_KEY not found. SEC fetching will likely fail.")
        return []

    # Query sec-api.io for filings
    query_url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    headers = {"Content-Type": "application/json", "Authorization": sec_key}

    try:
        r = requests.post(query_url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        filings = r.json().get("filings", [])
    except Exception as e:
        if debug:
            st.error(f"❌ Failed to query sec-api.io: {e}")
        return []

    if not filings:
        if debug:
            st.info(f"No 10-Q/10-K filings found for {ticker}.")
        return []

    converter_url = "https://api.sec-api.io/xbrl-to-json"

    for filing in filings:
        filed_at = filing.get("filedAt")
        form_type = filing.get("formType")
        link_to_xbrl = filing.get("linkToXbrl")  # preferred
        link_to_filing = filing.get("linkToFilingDetails")  # fallback

        # Also try scanning documentFormatFiles list if present for an xml
        doc_files = filing.get("documentFormatFiles", [])

        candidate_urls = []
        if link_to_xbrl:
            candidate_urls.append(link_to_xbrl)
        if doc_files and isinstance(doc_files, list):
            for doc in doc_files:
                url = doc.get("documentUrl") or doc.get("downloadUrl")
                if url and url.endswith(".xml"):
                    candidate_urls.append(url)
        if link_to_filing:
            candidate_urls.append(link_to_filing)

        # Also build some common EDGAR fallback patterns using accessionNo & cik
        accession = filing.get("accessionNo", "").replace("-", "")
        cik = filing.get("cik", "")
        period = filing.get("periodOfReport", "")
        if cik and accession:
            # common pattern: <accession>-xbrl.xml
            candidate_urls.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession}-xbrl.xml")
            # patterns including ticker+period
            if ticker and period:
                p = period.replace("-", "")
                candidate_urls.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{ticker.lower()}-{p}_htm.xml")
                candidate_urls.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{ticker.lower()}-{p}.xml")

        if debug:
            st.write(f"🔎 DEBUG: Filing {form_type} filedAt={filed_at} candidate URLs (first wins):")
            for u in candidate_urls:
                st.write("  •", u)

        data = None
        used_url = None

        # Try each candidate URL with the converter until one succeeds
        for target in candidate_urls:
            if not target:
                continue
            try:
                resp = requests.get(converter_url, params={"url": target}, headers={"Authorization": sec_key}, timeout=30)
                if debug:
                    st.write(f"🔎 DEBUG: Converter status for {target}: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                used_url = target
                break
            except Exception as e:
                if debug:
                    st.write(f"⚠️ DEBUG: Converter failed for {target}: {e}")
                data = None
                used_url = None
                continue

        if not data:
            if debug:
                st.warning(f"⚠️ No XBRL JSON available for {ticker} {form_type} filed {filed_at}. Skipping.")
            continue

        # Extract financial metrics robustly
        revenue = None
        net_income = None
        eps = None

        # Try common fields/structures
        # 1) incomeStatement key (list or dict)
        income_statements = None
        for cand in ("incomeStatement", "IncomeStatements", "IncomeStatement", "incomeStatements"):
            if cand in data:
                income_statements = data.get(cand)
                break

        if income_statements:
            # If it's a list of period objects or lists of facts
            latest = None
            try:
                if isinstance(income_statements, list) and len(income_statements) > 0:
                    # Sometimes it's a list of dicts, sometimes list-of-lists
                    latest = income_statements[0] if isinstance(income_statements[0], dict) else income_statements[-1]
                else:
                    latest = income_statements
            except Exception:
                latest = income_statements

            # If latest is dict with named fields:
            if isinstance(latest, dict):
                # check keys
                for key in ["Revenues", "SalesRevenueNet", "TotalRevenue", "RevenueFromContractWithCustomer"]:
                    if key in latest and revenue is None:
                        revenue = latest.get(key)
                for key in ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]:
                    if key in latest and net_income is None:
                        net_income = latest.get(key)
                for key in ["EarningsPerShareDiluted", "EarningsPerShareBasic", "EarningsPerShare"]:
                    if key in latest and eps is None:
                        eps = latest.get(key)
            elif isinstance(latest, list):
                # list of fact dicts
                for fact in reversed(latest):
                    concept = fact.get("concept") or ""
                    val = fact.get("value")
                    if ("revenu" in concept.lower() or "salesrevenue" in concept.lower()) and revenue is None:
                        revenue = val
                    if ("netincome" in concept.lower() or "profitloss" in concept.lower()) and net_income is None:
                        net_income = val
                    if ("earningspershare" in concept.lower() or "eps" in concept.lower()) and eps is None:
                        eps = val
        else:
            # 2) fallback to 'facts' structure
            facts = data.get("facts", {}) or {}
            if facts:
                for concept, entry in facts.items():
                    low = concept.lower()
                    # entry may be dict with 'value' or list
                    value = None
                    try:
                        if isinstance(entry, dict) and "value" in entry:
                            value = entry.get("value")
                        elif isinstance(entry, list) and entry:
                            # entry[0] may contain 'value'
                            first = entry[0]
                            if isinstance(first, dict) and "value" in first:
                                value = first.get("value")
                    except Exception:
                        value = None

                    if value is None:
                        continue

                    if "revenue" in low and revenue is None:
                        revenue = value
                    if "netincome" in low and net_income is None:
                        net_income = value
                    if ("earningspershare" in low or "eps" in low) and eps is None:
                        eps = value

        filings_data.append({
            "filed_at": filed_at,
            "period": filing.get("periodOfReport"),
            "type": form_type,
            "xbrl_source_used": used_url,
            "revenue": revenue,
            "eps": eps,
            "net_income": net_income,
            "accession_number": filing.get("accessionNo"),
        })

    return filings_data


# --- Simulated Earnings Call Transcripts ---
@st.cache_data(ttl=3600, show_spinner="🎙️ Fetching earnings call transcripts...")
def fetch_earnings_transcripts(ticker: str, quarters: int = 2):
    # Simulated sample transcripts for demo
    transcripts = []
    sample = {
        "date": "2024-10-30",
        "quarter": "Q3 2024",
        "source": "SeekingAlpha",
        "ceo_comments": [
            "We delivered strong results this quarter with revenue growth of 8% year-over-year",
            "Our new product line is gaining significant traction in the market",
            "We remain optimistic about our growth prospects for the remainder of the year"
        ],
        "analyst_questions": [
            "What are your expectations for margin expansion next quarter?",
            "How is the competitive landscape affecting your market share?",
            "Can you provide more details on your capital allocation strategy?"
        ],
        "key_metrics_discussed": [
            "User growth rate increased 12% quarter-over-quarter",
            "Gross margins improved to 68%, up from 65% last quarter",
            "Free cash flow generation remains strong at $8.2B"
        ],
    }
    transcripts.append(sample)
    if quarters > 1:
        t2 = sample.copy()
        t2["date"] = "2024-07-31"
        t2["quarter"] = "Q2 2024"
        t2["source"] = "MotleyFool"
        transcripts.append(t2)
    return transcripts[:quarters]


# --- Simulated Analyst Reports ---
@st.cache_data(ttl=1800, show_spinner="📰 Collecting analyst reports...")
def fetch_analyst_sentiment(ticker: str, days: int = 30):
    sample_reports = [
        {
            "date": "2024-11-01",
            "firm": "Goldman Sachs",
            "rating": "Buy",
            "price_target": 180,
            "headline": "Strong Q3 results support positive outlook",
            "key_points": [
                "Revenue beat expectations by 3%",
                "Margin expansion ahead of schedule",
                "Management guidance raised for FY2024"
            ]
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
                "Well-positioned for economic uncertainty"
            ]
        }
    ]
    return sample_reports


# --- Market Data via yfinance ---
@st.cache_data(ttl=300, show_spinner="📈 Fetching market data...")
def fetch_market_data(ticker: str, days: int = 90):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info if hasattr(stock, "info") else {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if hist.empty:
            return {}

        current_price = float(hist["Close"].iloc[-1])
        price_change_30d = ((current_price - float(hist["Close"].iloc[-30])) / float(hist["Close"].iloc[-30]) * 100) if len(hist) >= 30 else 0.0
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


# --- AI Analysis using Gemini ---
@st.cache_data(ttl=7200, show_spinner="🤖 Generating AI analysis...")
def analyze_earnings_with_ai(ticker: str, sec_filings: list, transcripts: list, analyst_reports: list, market_data: dict):
    """
    Uses Gemini to produce a structured JSON investment analysis.
    """
    try:
        filings_context = "\n".join([
            f"  - {f.get('period', f.get('filed_at','N/A'))} ({f.get('type','N/A')}): Revenue: {format_currency(f.get('revenue'))}, EPS: {f.get('eps','N/A')}, Net Income: {format_currency(f.get('net_income'))}"
            for f in sec_filings
        ]) if sec_filings else "No SEC filing data available."

        transcript_context = "\n".join([
            f"  - {t.get('quarter','N/A')} ({t.get('date','N/A')}): CEO comments: {'; '.join(t.get('ceo_comments', []))}"
            for t in transcripts
        ]) if transcripts else "No transcript data available."

        analyst_context = "\n".join([
            f"  - {r.get('firm','N/A')}: Rating: {r.get('rating','N/A')}, Price Target: {format_currency(r.get('price_target'))}, Headline: {r.get('headline','N/A')}"
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

The analysis must be provided in a single JSON object with keys:
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

        return json.loads(response.text)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ AI did not return valid JSON: {e}. Raw: {getattr(response, 'text', '')}")
        return {}
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}


# --- Streamlit Dashboard ---
def render_dashboard():
    st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Earnings Intelligence Platform")
    st.markdown("Combines SEC filings (XBRL), earnings call highlights, analyst reports, market data and AI synthesis.")

    # Debug toggle
    debug = st.checkbox("Show SEC debug logs", value=True)

    # Input
    st.subheader("Enter Ticker Symbol")
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").strip().upper()
    with col2:
        quarters = st.selectbox("Quarters to analyze", [1, 2, 3, 4], index=1)

    analyze_button = st.button("🔍 Analyze Earnings", type="primary", use_container_width=True)
    st.markdown("---")

    if analyze_button and ticker:
        with st.spinner("🔄 Collecting earnings intelligence..."):
            sec_data = fetch_sec_earnings(ticker, quarters, debug=debug)
            transcripts = fetch_earnings_transcripts(ticker, quarters)
            analyst_data = fetch_analyst_sentiment(ticker)
            market_data = fetch_market_data(ticker)

        # Tabs for raw data
        st.subheader("Source Data Overview")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 SEC Filings", "🎙️ Earnings Calls", "📰 Analyst Reports", "📈 Market Data"])

        with tab1:
            st.markdown("#### SEC Filing Summary")
            if sec_data:
                df = pd.DataFrame(sec_data)
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
                for t in transcripts:
                    with st.expander(f"**{safe_get(t,'quarter')}** - {safe_get(t,'date')}"):
                        st.write("**CEO Comments:**")
                        for c in safe_get(t, "ceo_comments", []):
                            st.write(f"• {c}")
                        st.write("**Key Metrics:**")
                        for m in safe_get(t, "key_metrics_discussed", []):
                            st.write(f"• {m}")
            else:
                st.info(f"No transcripts found for {ticker}.")

        with tab3:
            st.markdown("#### Analyst Coverage")
            if analyst_data:
                for rep in analyst_data:
                    with st.expander(f"**{safe_get(rep,'firm')}** - {safe_get(rep,'rating')}"):
                        st.write(f"**{safe_get(rep,'headline')}**")
                        for p in safe_get(rep, "key_points", []):
                            st.write(f"• {p}")
            else:
                st.info(f"No analyst coverage found for {ticker}.")

        with tab4:
            st.markdown("#### Market Data Snapshot")
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
                st.info(f"No market data for {ticker}.")

        # AI analysis
        st.markdown("---")
        st.header("🤖 AI Investment Analysis")
        with st.spinner("🧠 Generating comprehensive analysis..."):
            analysis = analyze_earnings_with_ai(ticker, sec_data, transcripts, analyst_data, market_data)

        if analysis:
            display_ai_analysis(analysis)
        else:
            st.warning("Could not generate AI analysis. Check logs and inputs.")

    st.markdown("---")
    st.caption("Data Sources: sec-api.io / SEC EDGAR, simulated transcripts & analyst notes, Yahoo Finance; AI: Google Gemini")


# --- Display AI Analysis helper ---
def display_ai_analysis(analysis: dict):
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade = safe_get(analysis, "overall_grade", "N/A")
        st.markdown(f"""
        <div style="background-color:#444;padding:1rem;border-radius:.5rem;text-align:center;">
            <h3 style="margin:0;color:#90ee90">📊 Overall Grade</h3>
            <h1 style="margin:.5rem 0 0 0;color:#90ee90;font-size:2.5rem;">{grade}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        recommendation = safe_get(analysis, "recommendation", "N/A")
        color = "#a3d900" if "Buy" in recommendation else "#fdd835" if "Hold" in recommendation else "#f44336"
        st.markdown(f"""
        <div style="background-color:#444;padding:1rem;border-radius:.5rem;border-left:4px solid {color};">
            <h4 style="margin:0;color:{color}">💡 Recommendation</h4>
            <p style="margin:.5rem 0 0 0;font-weight:600;color:#fff;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        risk_level = safe_get(analysis, "risk_level", "N/A")
        risk_colors = {"Low": "#a3d900", "Medium": "#fdd835", "High": "#f44336"}
        risk_color = risk_colors.get(risk_level, "#bbbbbb")
        st.markdown(f"""
        <div style="background-color:#444;padding:1rem;border-radius:.5rem;border-left:4px solid {risk_color};">
            <h4 style="margin:0;color:{risk_color}">⚠️ Risk Level</h4>
            <p style="margin:.5rem 0 0 0;font-weight:600;color:#fff;">{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        valuation = safe_get(analysis, "valuation_assessment", "N/A")
        display_val = valuation if isinstance(valuation, str) and len(valuation) <= 140 else (valuation[:140] + "..." if isinstance(valuation, str) else valuation)
        st.markdown(f"""
        <div style="background-color:#444;padding:1rem;border-radius:.5rem;border-left:4px solid #00bcd4;">
            <h4 style="margin:0;color:#00bcd4">💰 Valuation</h4>
            <p style="margin:.5rem 0 0 0;font-weight:600;color:#fff;">{display_val}</p>
        </div>
        """, unsafe_allow_html=True)

    # Investment thesis
    st.markdown("### 🎯 Investment Thesis")
    st.info(safe_get(analysis, "investment_thesis", "N/A"))

    # Strengths & Risks
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Key Strengths")
        for s in safe_get(analysis, "key_strengths", []):
            st.success(f"✓ {s}")
        st.markdown("### 📈 Earnings Performance")
        earnings = safe_get(analysis, "earnings_surprises", {})
        st.write(f"**Revenue:** {safe_get(earnings, 'revenue_beat_miss', 'N/A')}")
        st.write(f"**EPS:** {safe_get(earnings, 'eps_beat_miss', 'N/A')}")
        st.write(f"**Guidance:** {safe_get(earnings, 'guidance_reaction', 'N/A')}")
    with c2:
        st.markdown("### ⚠️ Key Risks")
        for r in safe_get(analysis, "key_risks", []):
            st.error(f"✗ {r}")
        st.markdown("### 🔮 Price Catalysts")
        for p in safe_get(analysis, "price_catalysts", []):
            st.info(f"• {p}")

    # Financial health & position
    st.markdown("### 💼 Financial Health & Competitive Position")
    fh = safe_get(analysis, "financial_health", {})
    st.write(f"**Revenue Trend:** {safe_get(fh, 'revenue_trend', 'N/A')}")
    st.write(f"**Profitability:** {safe_get(fh, 'profitability', 'N/A')}")
    st.write(f"**Balance Sheet:** {safe_get(fh, 'balance_sheet', 'N/A')}")
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