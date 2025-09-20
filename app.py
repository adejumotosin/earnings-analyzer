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
        # Remove commas, parentheses (negatives), currency symbols
        s = value.strip().replace(",", "")
        sign = -1 if ("(" in s and ")" in s) else 1
        s = s.replace("(", "").replace(")", "").replace("$", "")
        # Sometimes the string contains text like "1000 (thousands)" — try to keep numeric prefix
        # Split by space and take first numeric-like token
        token = s.split()[0] if s.split() else s
        try:
            return sign * float(token)
        except Exception:
            return None
    # Unknown type
    return None

def extract_fact_value(obj: Any) -> Optional[float]:
    """
    Extract value from a sec-api xbrl-to-json fact entry.
    Handles:
      - dicts with 'value'
      - lists of dicts (take first or last with 'value')
      - simple scalars
    Returns numeric float or None.
    """
    if obj is None:
        return None
    # If already numeric
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    # If dict with 'value'
    if isinstance(obj, dict):
        # Sometimes value is nested under 'value'
        if "value" in obj:
            return parse_number(obj.get("value"))
        # Some objects are concept->period->value structures; try to locate any numeric field
        for k in ("val", "amount", "value"):
            if k in obj:
                return parse_number(obj.get(k))
        # fallback: attempt to stringify
        return parse_number(json.dumps(obj))
    # If list, inspect items for 'value'
    if isinstance(obj, list) and obj:
        # prefer last (most recent) with value
        for item in reversed(obj):
            if isinstance(item, dict) and "value" in item:
                return parse_number(item.get("value"))
            if isinstance(item, dict):
                # try common numeric keys
                for k in ("value", "val", "amount"):
                    if k in item:
                        return parse_number(item.get(k))
            else:
                v = parse_number(item)
                if v is not None:
                    return v
        # fallback to parse first element
        return parse_number(obj[0])
    # If scalar string
    return parse_number(obj)

# -------------------------
# SEC XBRL -> JSON fetcher
# -------------------------
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker: str, quarters: int = 4, debug: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Returns (filings_list, raw_responses).
    filings_list: list of dicts with keys: filed_at, period, type, revenue (float/None), eps (float/None), net_income (float/None), accession_number, xbrl_source_used
    raw_responses: mapping filed_at -> raw JSON returned by the converter (for debugging)
    """
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
        # If documentFormatFiles includes an xml extracted instance, prefer that (some responses include it)
        for doc in doc_files:
            url = doc.get("documentUrl") or doc.get("downloadUrl")
            if url and isinstance(url, str) and url.lower().endswith(".xml"):
                # Add as xbrl-url if not duplicates
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
        # Try candidates until one returns 200
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
            # Many sec-api responses include 'facts' with concepts
            facts = xbrl_json.get("facts") or {}
            # Try to extract common concepts from facts
            rev = None
            ni = None
            eps_val = None
            for concept_key, concept_value in facts.items():
                lower = concept_key.lower()
                if "revenue" in lower and rev is None:
                    rev = extract_fact_value(concept_value)
                if "netincome" in lower and ni is None:
                    ni = extract_fact_value(concept_value)
                if ("earningspershare" in lower or "eps" in lower) and eps_val is None:
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

        # income_block may be list or dict. Normalize to dict (most recent)
        chosen_statement = None
        if isinstance(income_block, list) and len(income_block) > 0:
            # Some lists are lists of period dicts OR lists of fact dicts.
            # If list of dicts with named keys use last item; if list of facts, try to build mapping.
            # Prefer the last element that looks like a dict with named financial fields
            for item in reversed(income_block):
                if isinstance(item, dict) and any(k.lower() in map(str.lower, item.keys()) for k in ("Revenues", "NetIncomeLoss", "EarningsPerShareDiluted")):
                    chosen_statement = item
                    break
            if chosen_statement is None:
                # fallback to last element
                chosen_statement = income_block[-1]
        elif isinstance(income_block, dict):
            chosen_statement = income_block
        else:
            if debug:
                st.warning(f"⚠️ Unexpected format for income block in filing {filed_at}: {type(income_block)}")
            continue

        # Extract fields using robust extractor
        revenue_raw = None
        net_income_raw = None
        eps_raw = None

        # If chosen_statement is dict of named fields
        if isinstance(chosen_statement, dict):
            # Try common field names
            for key in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "TotalRevenue"):
                if key in chosen_statement and revenue_raw is None:
                    revenue_raw = chosen_statement.get(key)
            for key in ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"):
                if key in chosen_statement and net_income_raw is None:
                    net_income_raw = chosen_statement.get(key)
            for key in ("EarningsPerShareDiluted", "EarningsPerShareBasic", "EarningsPerShare"):
                if key in chosen_statement and eps_raw is None:
                    eps_raw = chosen_statement.get(key)

            # If fields are still None, inspect all values for likely candidates
            if revenue_raw is None or net_income_raw is None or eps_raw is None:
                for k, v in chosen_statement.items():
                    low = k.lower()
                    if revenue_raw is None and ("revenue" in low or "sales" in low):
                        revenue_raw = revenue_raw or v
                    if net_income_raw is None and ("netincome" in low or "profit" in low):
                        net_income_raw = net_income_raw or v
                    if eps_raw is None and ("earningspershare" in low or "eps" in low):
                        eps_raw = eps_raw or v

        # If chosen_statement is list of facts (concept/value dicts)
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
# Simulated transcripts & analysts (unchanged)
# -------------------------
@st.cache_data(ttl=3600, show_spinner="🎙️ Fetching earnings call transcripts...")
def fetch_earnings_transcripts(ticker: str, quarters: int = 2):
    sample = {
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
    return [sample][:quarters]

@st.cache_data(ttl=1800, show_spinner="📰 Collecting analyst reports...")
def fetch_analyst_sentiment(ticker: str):
    return [
        {
            "date": "2024-11-01",
            "firm": "Goldman Sachs",
            "rating": "Buy",
            "price_target": 180,
            "headline": "Strong Q3 results support positive outlook",
            "key_points": ["Revenue beat expectations by 3%"]
        }
    ]

# -------------------------
# Market data (yfinance)
# -------------------------
@st.cache_data(ttl=300, show_spinner="📈 Fetching market data...")
def fetch_market_data(ticker: str, days: int = 90):
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
            "price_history": hist["Close"].tolist()[-30:],
            "volume_history": hist["Volume"].tolist()[-30:] if "Volume" in hist.columns else []
        }
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed: {e}")
        return {}

# -------------------------
# AI Analysis (Gemini) - send summarized context only
# -------------------------
@st.cache_data(ttl=7200, show_spinner="🤖 Generating AI analysis...")
def analyze_earnings_with_ai(ticker: str, sec_filings: List[Dict], transcripts: List[Dict], analyst_reports: List[Dict], market_data: Dict):
    """
    Build a compact summary of SEC filings (dates + numeric metrics) and send it to Gemini.
    Expect JSON back; handle parse errors gracefully.
    """
    try:
        # Build compact filings summary
        filings_summary = []
        for f in sec_filings:
            filings_summary.append({
                "date": f.get("filed_at"),
                "period": f.get("period"),
                "type": f.get("type"),
                "revenue": f.get("revenue"),
                "eps": f.get("eps"),
                "net_income": f.get("net_income")
            })

        prompt = f"""
You are a professional financial analyst. Based on the data below produce a single valid JSON object with keys:
overall_grade, investment_thesis, financial_health (revenue_trend, profitability, balance_sheet),
key_strengths (list), key_risks (list), analyst_consensus (avg_rating, price_target_range, sentiment_shift),
earnings_surprises (revenue_beat_miss, eps_beat_miss, guidance_reaction), competitive_position, valuation_assessment,
price_catalysts (list), recommendation, risk_level.

Keep each field short. Use the numeric SEC metrics provided (date, revenue, eps, net_income) and analyst summaries.

SEC_FILINGS_SUMMARY = {json.dumps(filings_summary)}
TRANSCRIPTS = {json.dumps(transcripts)}
ANALYSTS = {json.dumps(analyst_reports)}
MARKET = {json.dumps({'current_price': market_data.get('current_price'), 'pe_ratio': market_data.get('pe_ratio'), 'market_cap': market_data.get('market_cap')})}

Return ONLY the JSON object, no extra commentary.
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )

        # Parse JSON safely
        text = response.text
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ AI analysis failed to return valid JSON: {e}. Raw response truncated:\n{text[:2000]}")
        return {}
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}

# -------------------------
# Dashboard
# -------------------------
def render_dashboard():
    st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Earnings Intelligence Platform")

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

        with st.spinner("🔄 Collecting earnings intelligence..."):
            sec_data, raw_responses = fetch_sec_earnings(ticker, quarters, debug=debug)
            transcripts = fetch_earnings_transcripts(ticker, quarters)
            analyst_data = fetch_analyst_sentiment(ticker)
            market_data = fetch_market_data(ticker)

        # Show raw tabs
        st.subheader("Source Data Overview")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 SEC Filings", "🎙️ Transcripts", "📰 Analysts", "📈 Market"])

        with tab1:
            if sec_data:
                df = pd.DataFrame(sec_data)
                # Format numbers nicely
                for col in ("revenue","net_income","eps"):
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: format_currency(x) if x is not None else "N/A")
                st.dataframe(df, use_container_width=True)

                with st.expander("🔎 Show Raw SEC JSON (Debug)"):
                    # Show raw responses (may be large)
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
                        for p in a.get("key_points", []):
                            st.write(f"- {p}")
            else:
                st.info("No analyst reports.")

        with tab4:
            if market_data:
                st.metric("Current Price", format_currency(market_data.get("current_price",0)))
                st.metric("30-Day Change", f"{market_data.get('price_change_30d',0):.1f}%")
                st.metric("P/E Ratio", f"{market_data.get('pe_ratio',0)}")
                st.metric("Market Cap", format_currency(market_data.get("market_cap",0)))
            else:
                st.info("No market data.")

        # Plotly charts: Revenue, Net Income, EPS over filings
        # Build a DataFrame of numeric series for charting
        numeric_rows = []
        for f in sec_data:
            d = {
                "filed_at": f.get("filed_at"),
                "period": f.get("period"),
                "revenue": f.get("revenue"),
                "net_income": f.get("net_income"),
                "eps": f.get("eps")
            }
            numeric_rows.append(d)

        if numeric_rows:
            chart_df = pd.DataFrame(numeric_rows)
            # Convert filed_at or period to datetime
            if "period" in chart_df.columns and chart_df["period"].notna().any():
                # prefer period if available
                chart_df["x"] = pd.to_datetime(chart_df["period"], errors="coerce")
            else:
                chart_df["x"] = pd.to_datetime(chart_df["filed_at"], errors="coerce")

            chart_df = chart_df.sort_values("x")
            # Revenue chart
            revenue_df = chart_df.dropna(subset=["revenue", "x"])[["x","revenue"]]
            if not revenue_df.empty:
                fig_rev = px.line(revenue_df, x="x", y="revenue", markers=True, title="Revenue Trend")
                fig_rev.update_yaxes(tickprefix="$")
                st.plotly_chart(fig_rev, use_container_width=True)
            # Net income
            ni_df = chart_df.dropna(subset=["net_income","x"])[["x","net_income"]]
            if not ni_df.empty:
                fig_ni = px.line(ni_df, x="x", y="net_income", markers=True, title="Net Income Trend")
                fig_ni.update_yaxes(tickprefix="$")
                st.plotly_chart(fig_ni, use_container_width=True)
            # EPS
            eps_df = chart_df.dropna(subset=["eps","x"])[["x","eps"]]
            if not eps_df.empty:
                fig_eps = px.line(eps_df, x="x", y="eps", markers=True, title="EPS Trend")
                st.plotly_chart(fig_eps, use_container_width=True)

        # AI analysis (summarized)
        st.markdown("---")
        st.header("🤖 AI Investment Analysis")
        with st.spinner("🧠 Generating comprehensive analysis..."):
            analysis = analyze_earnings_with_ai(ticker, sec_data, transcripts, analyst_data, market_data)

        if analysis:
            # If the model returned a dict, pretty display; otherwise show raw
            if isinstance(analysis, dict):
                # Top-level highlights
                # Show grade / recommendation if present
                grade = analysis.get("overall_grade", "N/A")
                rec = analysis.get("recommendation", "N/A")
                st.metric("Overall Grade", grade)
                st.metric("Recommendation", rec)
                st.json(analysis)
            else:
                st.write("AI output (raw):")
                st.write(analysis)
        else:
            st.warning("AI analysis could not be generated. Check debug logs and raw SEC JSON.")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    render_dashboard()