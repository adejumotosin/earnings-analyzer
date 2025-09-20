import os
import json
import requests
import pandas as pd
import streamlit as st
import google.generativeai as genai
import yfinance as yf
from datetime import datetime, timedelta

# ==============================
# --- Configuration & Setup ---
# ==============================
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set it in your Streamlit secrets.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()


# ==========================
# --- Utility Functions ---
# ==========================
def safe_get(data, key, default="N/A"):
    return data.get(key, default)


def format_currency(value):
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        elif value >= 1e9:
            return f"${value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.2f}"
    return value


# ====================================
# --- SEC API: XBRL-to-JSON fetch ---
# ====================================
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4):
    filings_data = []
    raw_responses = {}

    sec_api_key = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")
    if not sec_api_key:
        st.error("❌ Missing SEC API key. Please set it in your Streamlit secrets.")
        return [], {}

    query_url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    headers = {"Content-Type": "application/json", "Authorization": sec_api_key}

    try:
        query_response = requests.post(query_url, json=payload, headers=headers)
        query_response.raise_for_status()
        filings = query_response.json().get("filings", [])
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch filing metadata: {e}")
        return [], {}

    if not filings:
        st.info(f"No 10-Q or 10-K filings found for {ticker}.")
        return [], {}

    for filing in filings:
        filing_date = filing.get("filedAt")
        form_type = filing.get("formType")
        accession_no = filing.get("accessionNo")

        filing_xbrl = filing.get("linkToXbrl")
        filing_html = filing.get("linkToFilingDetails")

        candidates = []
        if filing_xbrl:
            candidates.append(("xbrl-url", filing_xbrl))
        if filing_html:
            candidates.append(("htm-url", filing_html))
        if accession_no:
            candidates.append(("accession-no", accession_no))

        st.write(f"🔎 DEBUG: Filing {form_type} filedAt={filing_date} candidates:")
        for k, v in candidates:
            st.write(f"• {k}: {v}")

        xbrl_data = None
        for param_name, param_value in candidates:
            try:
                xbrl_url = "https://api.sec-api.io/xbrl-to-json"
                resp = requests.get(
                    xbrl_url,
                    params={param_name: param_value},
                    headers={"Authorization": sec_api_key},
                )
                st.write(
                    f"🔎 DEBUG: Converter status for {param_name}={param_value}: {resp.status_code}"
                )
                resp.raise_for_status()
                xbrl_data = resp.json()
                raw_responses[filing_date] = xbrl_data
                break
            except requests.exceptions.RequestException as e:
                st.warning(
                    f"⚠️ DEBUG: Converter failed for {param_name}={param_value}: {e}"
                )

        if not xbrl_data:
            st.warning(
                f"⚠️ No XBRL JSON available for {ticker} {form_type} filed {filing_date}. Skipping."
            )
            continue

        # Extract income statement safely
        possible_keys = [
            "StatementsOfIncome",
            "incomeStatement",
            "ComprehensiveIncomeStatement",
            "StatementOfIncome",
        ]
        income_statements = None
        for key in possible_keys:
            if key in xbrl_data:
                income_statements = xbrl_data[key]
                break

        if not income_statements:
            st.warning(
                f"⚠️ No income statement found in filing {form_type} {filing_date}. Keys available: {list(xbrl_data.keys())[:10]}"
            )
            continue

        if isinstance(income_statements, list) and income_statements:
            last_statement = income_statements[-1]
        elif isinstance(income_statements, dict):
            last_statement = income_statements
        else:
            st.warning(f"⚠️ Unexpected format for income statement in {form_type} {filing_date}")
            continue

        revenue = (
            last_statement.get("Revenues")
            or last_statement.get("RevenueFromContractWithCustomerExcludingAssessedTax")
        )
        net_income = last_statement.get("NetIncomeLoss")
        eps = last_statement.get("EarningsPerShareDiluted")

        filings_data.append(
            {
                "date": filing_date,
                "type": form_type,
                "revenue": revenue,
                "eps": eps,
                "net_income": net_income,
            }
        )

    return filings_data, raw_responses


# =====================================================
# --- Simulated Transcripts, Analyst, Market Data ---
# =====================================================
@st.cache_data(ttl=3600, show_spinner="🎙️ Fetching earnings call transcripts...")
def fetch_earnings_transcripts(ticker, quarters=2):
    transcripts = [
        {
            "date": "2024-10-30",
            "quarter": "Q3 2024",
            "source": "SeekingAlpha",
            "ceo_comments": [
                "We delivered strong results this quarter with revenue growth of 8% year-over-year",
                "Our new product line is gaining significant traction in the market",
            ],
            "key_metrics_discussed": [
                "User growth +12% QoQ",
                "Gross margins improved to 68%",
            ],
        }
    ]
    return transcripts[:quarters]


@st.cache_data(ttl=1800, show_spinner="📰 Collecting analyst reports...")
def fetch_analyst_sentiment(ticker):
    return [
        {
            "date": "2024-11-01",
            "firm": "Goldman Sachs",
            "rating": "Buy",
            "price_target": 180,
            "headline": "Strong Q3 results support positive outlook",
        }
    ]


@st.cache_data(ttl=300, show_spinner="📈 Fetching market data...")
def fetch_market_data(ticker, days=90):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist = stock.history(
            start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )

        if hist.empty:
            st.warning(f"⚠️ No market data found for {ticker}")
            return {}

        current_price = hist["Close"][-1]
        price_change_30d = (
            (current_price - hist["Close"][-30]) / hist["Close"][-30]
        ) * 100
        avg_volume = hist["Volume"].mean()

        return {
            "current_price": current_price,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "price_change_30d": price_change_30d,
            "avg_volume": avg_volume,
        }
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed: {e}")
        return {}


# =======================================
# --- AI-Powered Earnings Analysis ---
# =======================================
@st.cache_data(ttl=7200, show_spinner="🤖 Generating AI analysis...")
def analyze_earnings_with_ai(ticker, sec_filings, transcripts, analyst_reports, market_data):
    try:
        filings_context = (
            "\n".join(
                [
                    f"- {f.get('date')} ({f.get('type')}): Revenue: {format_currency(f.get('revenue'))}, EPS: {f.get('eps')}, Net Income: {format_currency(f.get('net_income'))}"
                    for f in sec_filings
                ]
            )
            if sec_filings
            else "No SEC data"
        )

        transcript_context = (
            "\n".join(
                [
                    f"- {t.get('quarter')} ({t.get('date')}): CEO: {'; '.join(t.get('ceo_comments', []))}"
                    for t in transcripts
                ]
            )
            if transcripts
            else "No transcript data"
        )

        analyst_context = (
            "\n".join(
                [
                    f"- {r.get('firm')}: {r.get('rating')} PT: {format_currency(r.get('price_target'))}"
                    for r in analyst_reports
                ]
            )
            if analyst_reports
            else "No analyst reports"
        )

        market_context = f"""
        Price: {format_currency(market_data.get('current_price',0))}
        30d Change: {market_data.get('price_change_30d',0):.1f}%
        P/E: {market_data.get('pe_ratio',0)}
        Market Cap: {format_currency(market_data.get('market_cap',0))}
        """

        prompt = f"""
        Provide a JSON analysis of {ticker} combining SEC filings, transcripts, analyst reports, and market data.
        Data:
        {filings_context}
        {transcript_context}
        {analyst_context}
        {market_context}
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}


# ==========================
# --- Dashboard Render ---
# ==========================
def render_dashboard():
    st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Earnings Intelligence Platform")

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        ticker = (
            st.text_input("Stock Ticker", value="AAPL", placeholder="AAPL, MSFT")
            .strip()
            .upper()
        )
    with col2:
        quarters = st.selectbox("Quarters to analyze", [1, 2, 3, 4], index=1)

    if st.button("🔍 Analyze Earnings", type="primary", use_container_width=True):
        with st.spinner("🔄 Collecting data..."):
            sec_data, raw_responses = fetch_sec_earnings(ticker, quarters)
            transcripts = fetch_earnings_transcripts(ticker, quarters)
            analyst_data = fetch_analyst_sentiment(ticker)
            market_data = fetch_market_data(ticker)

        st.subheader("Source Data Overview")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 SEC Filings", "🎙️ Transcripts", "📰 Analysts", "📈 Market"]
        )

        with tab1:
            if sec_data:
                df = pd.DataFrame(sec_data)
                df["revenue"] = df["revenue"].apply(
                    lambda x: format_currency(x) if x else "N/A"
                )
                st.dataframe(df, use_container_width=True)

                with st.expander("🔎 Show Raw SEC JSON (Debug)"):
                    st.json(raw_responses)
            else:
                st.info("No SEC filings found")

        with tab2:
            for t in transcripts:
                st.write(f"**{t['quarter']}** ({t['date']}) - {t['source']}")
                for c in t["ceo_comments"]:
                    st.write(f"- {c}")

        with tab3:
            for r in analyst_data:
                st.write(f"**{r['firm']}** - {r['rating']} (PT: {format_currency(r['price_target'])})")
                st.write(r["headline"])

        with tab4:
            if market_data:
                st.metric("Price", format_currency(market_data["current_price"]))
                st.metric("P/E", market_data["pe_ratio"])
                st.metric("Market Cap", format_currency(market_data["market_cap"]))

        st.header("🤖 AI Investment Analysis")
        analysis = analyze_earnings_with_ai(ticker, sec_data, transcripts, analyst_data, market_data)
        if analysis:
            st.json(analysis)


if __name__ == "__main__":
    render_dashboard()