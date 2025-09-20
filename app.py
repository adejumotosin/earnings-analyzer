# app.py
import os
import json
import math
import requests
import pandas as pd
import streamlit as st
import google.generativeai as genai
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

# -------------------------
# Configuration & Setup
# -------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
SEC_API_KEY = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in Streamlit secrets or env.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()

# -------------------------
# Utility helpers
# -------------------------
def safe_get(d: Dict, k: str, default="N/A"):
    return d.get(k, default) if isinstance(d, dict) else default

def parse_number(value: Any) -> Optional[float]:
    """
    Try to coerce various representations into a float.
    Returns None if unable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        sign = -1 if ("(" in s and ")" in s) else 1
        s = s.replace("(", "").replace(")", "").replace("$", "")
        token = s.split()[0] if s.split() else s
        try:
            return sign * float(token)
        except Exception:
            return None
    return None

def extract_fact_value(obj: Any) -> Optional[float]:
    """
    Extract numeric value from sec-api xbrl-to-json fact entry.
    Handles dicts with 'value', lists, and scalars.
    """
    if obj is None:
        return None
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        if "value" in obj:
            return parse_number(obj.get("value"))
        for k in ("val", "amount", "value"):
            if k in obj:
                return parse_number(obj.get(k))
        # try scanning nested dict values
        for v in obj.values():
            vnum = extract_fact_value(v)
            if vnum is not None:
                return vnum
        return None
    if isinstance(obj, list) and obj:
        for item in reversed(obj):
            if isinstance(item, dict) and "value" in item:
                return parse_number(item.get("value"))
            if isinstance(item, (int, float)):
                return float(item)
            if isinstance(item, str):
                v = parse_number(item)
                if v is not None:
                    return v
        # fallback: parse first element
        return extract_fact_value(obj[0])
    if isinstance(obj, str):
        return parse_number(obj)
    return None

def format_currency(value: Optional[float]) -> str:
    """Formats a number into a readable currency string or returns 'N/A'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    try:
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        if value >= 1e9:
            return f"${value / 1e9:.2f}B"
        if value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.2f}"
    except Exception:
        return "N/A"

def safe_format(val: Any) -> str:
    """Used for DataFrame display: converts to currency or 'N/A'."""
    try:
        num = None
        if isinstance(val, (int, float)):
            num = float(val)
        else:
            num = parse_number(val)
        return format_currency(num) if num is not None else "N/A"
    except Exception:
        return "N/A"

# -------------------------
# SEC XBRL -> JSON fetcher
# -------------------------
@st.cache_data(ttl=3600)
def fetch_sec_earnings(ticker: str, quarters: int = 4, debug: bool = False) -> Tuple[List[Dict], Dict]:
    # ... (No changes here, the original function is robust)
    filings_data: List[Dict] = []
    raw_responses: Dict[str, Any] = {}

    if not SEC_API_KEY:
        if debug:
            st.warning("⚠️ Missing SEC_API_KEY; cannot fetch SEC filings.")
        return [], {}

    # Query sec-api.io for filings
    query_url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    headers = {"Content-Type": "application/json", "Authorization": SEC_API_KEY}

    try:
        resp = requests.post(query_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        filings = resp.json().get("filings", [])
    except Exception as e:
        if debug:
            st.error(f"❌ Could not query sec-api.io: {e}")
        return [], {}

    if not filings:
        if debug:
            st.info("No filings returned from sec-api.io for that ticker.")
        return [], {}

    converter_endpoint = "https://api.sec-api.io/xbrl-to-json"

    for filing in filings:
        filed_at = filing.get("filedAt")
        form_type = filing.get("formType")
        accession = filing.get("accessionNo")
        link_xbrl = filing.get("linkToXbrl")
        link_html = filing.get("linkToFilingDetails")
        doc_files = filing.get("documentFormatFiles") or []

        # Build candidate param list following docs: xbrl-url -> htm-url -> accession-no
        candidates = []
        if link_xbrl:
            candidates.append(("xbrl-url", link_xbrl))
        # Extract any .xml in documentFormatFiles
        for doc in doc_files:
            url = doc.get("documentUrl") or doc.get("downloadUrl")
            if url and isinstance(url, str) and url.lower().endswith(".xml"):
                if ("xbrl-url", url) not in candidates:
                    candidates.append(("xbrl-url", url))
        if link_html:
            candidates.append(("htm-url", link_html))
        if accession:
            candidates.append(("accession-no", accession))

        if debug:
            st.write(f"🔎 DEBUG: Filing {form_type} filedAt={filed_at} candidates:")
            for k, v in candidates:
                st.write(f"  • {k}: {v}")

        xbrl_json = None
        used_param = None
        for param_name, param_value in candidates:
            try:
                resp_conv = requests.get(converter_endpoint, params={param_name: param_value}, headers={"Authorization": SEC_API_KEY}, timeout=30)
                if debug:
                    st.write(f"🔎 DEBUG: Converter status for {param_name}={param_value}: {resp_conv.status_code}")
                resp_conv.raise_for_status()
                xbrl_json = resp_conv.json()
                used_param = (param_name, param_value)
                raw_responses[str(filed_at)] = xbrl_json
                break
            except Exception as e:
                if debug:
                    st.write(f"⚠️ DEBUG: Converter failed for {param_name}={param_value}: {e}")
                xbrl_json = None
                used_param = None
                continue

        if not xbrl_json:
            if debug:
                st.warning(f"⚠️ No XBRL JSON for filing {form_type} at {filed_at}.")
            continue

        # Look for income statement data under a number of possible keys
        possible_keys = ["StatementsOfIncome", "StatementsOfOperations", "incomeStatement", "IncomeStatements", "StatementOfIncome", "ComprehensiveIncomeStatement"]
        income_block = None
        for key in possible_keys:
            if key in xbrl_json:
                income_block = xbrl_json[key]
                break

        # If not found, try 'facts' as last resort
        if income_block is None:
            facts = xbrl_json.get("facts") or {}
            rev = None
            ni = None
            eps_val = None
            for concept_key, concept_value in facts.items():
                low = concept_key.lower()
                if "revenue" in low and rev is None:
                    rev = extract_fact_value(concept_value)
                if "netincome" in low and ni is None:
                    ni = extract_fact_value(concept_value)
                if ("earningspershare" in low or "eps" in low) and eps_val is None:
                    eps_val = extract_fact_value(concept_value)
            filings_data.append({
                "filed_at": filed_at,
                "period": filing.get("periodOfReport"),
                "type": form_type,
                "revenue": rev,
                "eps": eps_val,
                "net_income": ni,
                "accession_number": accession,
                "xbrl_source_used": used_param
            })
            continue

        # Normalize income_block
        chosen_statement = None
        if isinstance(income_block, list) and len(income_block) > 0:
            for item in reversed(income_block):
                if isinstance(item, dict) and any(k.lower() in map(str.lower, item.keys()) for k in ("Revenues", "NetIncomeLoss", "EarningsPerShareDiluted")):
                    chosen_statement = item
                    break
            if chosen_statement is None:
                chosen_statement = income_block[-1]
        elif isinstance(income_block, dict):
            chosen_statement = income_block
        else:
            if debug:
                st.warning(f"⚠️ Unexpected format for income block in filing {filed_at}: {type(income_block)}")
            continue

        revenue_raw = None
        net_income_raw = None
        eps_raw = None

        if isinstance(chosen_statement, dict):
            for key in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "TotalRevenue"):
                if key in chosen_statement and revenue_raw is None:
                    revenue_raw = chosen_statement.get(key)
            for key in ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"):
                if key in chosen_statement and net_income_raw is None:
                    net_income_raw = chosen_statement.get(key)
            for key in ("EarningsPerShareDiluted", "EarningsPerShareBasic", "EarningsPerShare"):
                if key in chosen_statement and eps_raw is None:
                    eps_raw = chosen_statement.get(key)

            if revenue_raw is None or net_income_raw is None or eps_raw is None:
                for k, v in chosen_statement.items():
                    low = k.lower()
                    if revenue_raw is None and ("revenue" in low or "sales" in low):
                        revenue_raw = revenue_raw or v
                    if net_income_raw is None and ("netincome" in low or "profit" in low):
                        net_income_raw = net_income_raw or v
                    if eps_raw is None and ("earningspershare" in low or "eps" in low):
                        eps_raw = eps_raw or v

        if isinstance(chosen_statement, list):
            for fact in reversed(chosen_statement):
                concept = fact.get("concept") or ""
                val = fact.get("value") if isinstance(fact, dict) else fact
                low = concept.lower()
                if revenue_raw is None and ("revenue" in low or "sales" in low):
                    revenue_raw = val
                if net_income_raw is None and ("netincome" in low or "profitloss" in low or "profit" in low):
                    net_income_raw = val
                if eps_raw is None and ("earningspershare" in low or "eps" in low):
                    eps_raw = val

        revenue = extract_fact_value(revenue_raw)
        net_income = extract_fact_value(net_income_raw)
        eps_val = extract_fact_value(eps_raw)

        filings_data.append({
            "filed_at": filed_at,
            "period": filing.get("periodOfReport"),
            "type": form_type,
            "revenue": revenue,
            "eps": eps_val,
            "net_income": net_income,
            "accession_number": accession,
            "xbrl_source_used": used_param
        })

    return filings_data, raw_responses

# -------------------------
# Simulated transcripts & analysts - IMPROVEMENT: Add example data for multiple tickers
# -------------------------
MOCK_DATA = {
    "AAPL": {
        "transcripts": [
            {
                "date": "2024-10-30", "quarter": "Q3 2024", "source": "SeekingAlpha",
                "ceo_comments": [
                    "We delivered strong results this quarter with revenue growth of 8% year-over-year.",
                    "Our new product line is gaining significant traction in the market.",
                ],
                "key_metrics_discussed": ["User growth +12% QoQ", "Gross margins improved to 68%."]
            },
        ],
        "analysts": [
            {
                "date": "2024-11-01", "firm": "Goldman Sachs", "rating": "Buy", "price_target": 180,
                "headline": "Strong Q3 results support positive outlook",
                "key_points": ["Revenue beat expectations by 3%", "New product line success."]
            }
        ]
    },
    "MSFT": {
        "transcripts": [
            {
                "date": "2024-10-25", "quarter": "Q1 2025", "source": "Microsoft Investor Relations",
                "ceo_comments": [
                    "Strong demand for our cloud services drove significant top-line growth.",
                    "AI integration across our product suite is a key differentiator."
                ],
                "key_metrics_discussed": ["Azure revenue growth +29% YoY", "Productivity & Business Processes up 15%."]
            }
        ],
        "analysts": [
            {
                "date": "2024-10-26", "firm": "Morgan Stanley", "rating": "Overweight", "price_target": 450,
                "headline": "Cloud strength remains a core driver",
                "key_points": ["Solid earnings beat on both revenue and EPS.", "Guidance indicates continued cloud momentum."]
            }
        ]
    }
}

@st.cache_data(ttl=3600)
def fetch_earnings_transcripts(ticker: str, quarters: int = 2):
    return MOCK_DATA.get(ticker.upper(), {}).get("transcripts", [])[:quarters]

@st.cache_data(ttl=1800)
def fetch_analyst_sentiment(ticker: str):
    return MOCK_DATA.get(ticker.upper(), {}).get("analysts", [])

# -------------------------
# Market data (yfinance)
# -------------------------
@st.cache_data(ttl=300)
def fetch_market_data(ticker: str, days: int = 90):
    # ... (No changes here, the original function is robust)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info if hasattr(stock, "info") else {}
        end = datetime.now()
        start = end - timedelta(days=days)
        hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return {}
        current_price = float(hist["Close"].iloc[-1])
        price_change_30d = ((current_price - float(hist["Close"].iloc[-30])) / float(hist["Close"].iloc[-30]) * 100) if len(hist) >= 30 else 0.0
        avg_vol = float(hist["Volume"].mean()) if "Volume" in hist.columns else 0.0
        return {
            "current_price": current_price,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "price_change_30d": price_change_30d,
            "avg_volume": avg_vol,
            "price_history": hist["Close"].tolist()[-90:],
            "dates": [d.strftime("%Y-%m-%d") for d in hist.index.tolist()[-90:]]
        }
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed: {e}")
        return {}

# -------------------------
# AI Analysis (Gemini) - IMPROVEMENT: Refine Prompt
# -------------------------
@st.cache_data(ttl=7200)
def analyze_earnings_with_ai(ticker: str, sec_filings: List[Dict], transcripts: List[Dict], analyst_reports: List[Dict], market_data: Dict):
    try:
        filings_summary = []
        for f in sec_filings:
            filings_summary.append({
                "filed_at": f.get("filed_at"),
                "period": f.get("period"),
                "type": f.get("type"),
                "revenue": f.get("revenue"),
                "eps": f.get("eps"),
                "net_income": f.get("net_income")
            })

        prompt = f"""
You are a professional financial analyst. Based on the data below, produce a single, valid JSON object with the following keys.
Ensure all values are populated based on the data provided.

1.  overall_grade: A single letter grade from A to F.
2.  recommendation: A single word: "Buy", "Hold", or "Sell".
3.  investment_thesis: A concise paragraph summarizing the core investment case.
4.  financial_health: An object with keys 'revenue_trend' (describes growth), 'profitability' (discusses margins/net income), and 'balance_sheet' (discusses cash/debt).
5.  key_strengths: A list of 2-3 key positive factors.
6.  key_risks: A list of 2-3 key negative factors or risks.
7.  analyst_consensus: A summary of analyst sentiment.
8.  earnings_surprises: A summary of how the company performed against expectations.

SEC_FILINGS_SUMMARY = {json.dumps(filings_summary)}
TRANSCRIPTS = {json.dumps(transcripts)}
ANALYSTS = {json.dumps(analyst_reports)}
MARKET = {json.dumps({'current_price': market_data.get('current_price'), 'pe_ratio': market_data.get('pe_ratio'), 'market_cap': market_data.get('market_cap')})}

Return ONLY the JSON object, nothing else.
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )

        text = response.text
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ AI analysis failed to return valid JSON: {e}. Raw (truncated): {text[:2000] if 'text' in locals() else 'NO RESPONSE'}")
        return {}
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}

# -------------------------
# Dashboard Rendering & UI
# -------------------------
def display_analysis(analysis: Dict):
    """IMPROVEMENT: Function to display the AI analysis in a clean format."""
    if not analysis:
        st.warning("AI analysis data is missing or incomplete.")
        return

    st.markdown("### 🤖 AI Investment Analysis")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        grade = analysis.get("overall_grade", "N/A")
        rec = analysis.get("recommendation", "N/A")
        st.metric("Overall Grade", grade)
        st.metric("Recommendation", rec)

    with col2:
        if "investment_thesis" in analysis:
            st.markdown(f"**Investment Thesis:**\n> {analysis['investment_thesis']}")

    if "financial_health" in analysis:
        st.markdown("#### Financial Health")
        health = analysis["financial_health"]
        if isinstance(health, dict):
            if "revenue_trend" in health:
                st.markdown(f"- **Revenue Trend:** {health['revenue_trend']}")
            if "profitability" in health:
                st.markdown(f"- **Profitability:** {health['profitability']}")
            if "balance_sheet" in health:
                st.markdown(f"- **Balance Sheet:** {health['balance_sheet']}")

    if "key_strengths" in analysis:
        with st.expander("Key Strengths"):
            for strength in analysis.get("key_strengths", []):
                st.markdown(f"- {strength}")
    
    if "key_risks" in analysis:
        with st.expander("Key Risks"):
            for risk in analysis.get("key_risks", []):
                st.markdown(f"- {risk}")

    # Add other sections from the JSON if they exist
    with st.expander("Show Detailed AI Output"):
        st.json(analysis)

def render_dashboard():
    st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Earnings Intelligence Platform")
    st.markdown("---")
    st.info("Enter a stock ticker to get a comprehensive AI-generated earnings analysis.")

    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="AAPL, MSFT, GOOGL").strip().upper()
    with c2:
        quarters = st.selectbox("Quarters to analyze", [1,2,3,4], index=1)

    debug = st.checkbox("Show SEC debug logs", value=True)

    if st.button("🔍 Analyze Earnings", type="primary", use_container_width=True):
        if not ticker:
            st.warning("Please enter ticker symbol.")
            st.stop()

        # IMPROVEMENT: Add more granular spinner messages
        with st.spinner("🔄 Collecting SEC filings..."):
            sec_data, raw_responses = fetch_sec_earnings(ticker, quarters, debug=debug)
        
        with st.spinner("🎙️ Fetching earnings call transcripts..."):
            transcripts = fetch_earnings_transcripts(ticker, quarters)

        with st.spinner("📰 Collecting analyst reports..."):
            analyst_data = fetch_analyst_sentiment(ticker)
        
        with st.spinner("📈 Fetching market data..."):
            market_data = fetch_market_data(ticker, days=90)

        # Tabs
        st.subheader("Source Data Overview")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 SEC Filings", "🎙️ Transcripts", "📰 Analysts", "📈 Market"])

        with tab1:
            if sec_data:
                df = pd.DataFrame(sec_data)
                for col in ("revenue","net_income","eps"):
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: safe_format(x))
                st.dataframe(df, use_container_width=True)

                with st.expander("🔎 Show Raw SEC JSON (Debug)"):
                    st.write("Raw converter responses (by filing date):")
                    st.json(raw_responses)
            else:
                st.info("No SEC filing data available.")

        with tab2:
            if transcripts:
                for t in transcripts:
                    with st.expander(f"{t.get('quarter')} — {t.get('date')} ({t.get('source')})"):
                        st.write("CEO comments:")
                        for c in t.get("ceo_comments", []):
                            st.write(f"- {c}")
                        st.write("Key metrics:")
                        for m in t.get("key_metrics_discussed", []):
                            st.write(f"- {m}")
            else:
                st.info("No transcripts.")

        with tab3:
            if analyst_data:
                for a in analyst_data:
                    with st.expander(f"{a.get('firm')} — {a.get('rating')}"):
                        st.write(a.get("headline"))
                        for p in a.get("key_points", a.get("key_points", [])):
                            st.write(f"- {p}")
            else:
                st.info("No analyst reports.")

        with tab4:
            if market_data:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Current Price", format_currency(market_data.get("current_price", 0)))
                col_m2.metric("30-Day Change", f"{market_data.get('price_change_30d', 0):.1f}%")
                col_m3.metric("P/E Ratio", f"{market_data.get('pe_ratio', 'N/A')}")
                col_m4.metric("Market Cap", format_currency(market_data.get("market_cap", 0)))

                ph = market_data.get("price_history", [])
                dates = market_data.get("dates", [])
                if ph and dates and len(ph) == len(dates):
                    df_price = pd.DataFrame({"date": pd.to_datetime(dates), "close": ph})
                    fig_price = px.line(df_price, x="date", y="close", title=f"{ticker} - Last {len(ph)} Days Close")
                    fig_price.update_yaxes(tickprefix="$")
                    st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.info("No market data.")

        # Plotly charts for filings
        numeric_rows = []
        for f in sec_data:
            numeric_rows.append({
                "filed_at": f.get("filed_at"),
                "period": f.get("period"),
                "revenue": f.get("revenue"),
                "net_income": f.get("net_income"),
                "eps": f.get("eps")
            })

        if numeric_rows:
            chart_df = pd.DataFrame(numeric_rows)
            if "period" in chart_df.columns and chart_df["period"].notna().any():
                chart_df["x"] = pd.to_datetime(chart_df["period"], errors="coerce")
            else:
                chart_df["x"] = pd.to_datetime(chart_df["filed_at"], errors="coerce")
            chart_df = chart_df.sort_values("x")

            rev_df = chart_df.dropna(subset=["revenue","x"])[["x","revenue"]]
            ni_df = chart_df.dropna(subset=["net_income","x"])[["x","net_income"]]
            eps_df = chart_df.dropna(subset=["eps","x"])[["x","eps"]]

            if not rev_df.empty or not ni_df.empty or not eps_df.empty:
                fig = go.Figure()
                if not rev_df.empty:
                    fig.add_trace(go.Bar(
                        x=rev_df["x"], y=rev_df["revenue"], name="Revenue", yaxis="y1",
                        hovertemplate='Revenue: %{y:$.2s}<extra></extra>'
                    ))
                if not ni_df.empty:
                    fig.add_trace(go.Bar(
                        x=ni_df["x"], y=ni_df["net_income"], name="Net Income", yaxis="y1",
                        hovertemplate='Net Income: %{y:$.2s}<extra></extra>'
                    ))
                if not eps_df.empty:
                    fig.add_trace(go.Line(
                        x=eps_df["x"], y=eps_df["eps"], name="EPS", yaxis="y2", marker=dict(symbol="diamond"),
                        hovertemplate='EPS: %{y:.2f}<extra></extra>'
                    ))
                # axes
                fig.update_layout(
                    title="Revenue & Net Income (bars) and EPS (line)",
                    xaxis=dict(title="Period"),
                    yaxis=dict(title="Amount (USD)", tickprefix="$"),
                    yaxis2=dict(title="EPS", overlaying="y", side="right")
                )
                st.plotly_chart(fig, use_container_width=True)

        # AI analysis
        st.markdown("---")
        with st.spinner("🧠 Generating comprehensive analysis..."):
            analysis = analyze_earnings_with_ai(ticker, sec_data, transcripts, analyst_data, market_data)

        # IMPROVEMENT: Use the new display function
        display_analysis(analysis)

if __name__ == "__main__":
    render_dashboard()
