import os
import re
import json
import requests
import pandas as pd
import streamlit as st
import google.generativeai as genai
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="📊 AI-Powered Earnings Intelligence Platform", layout="wide")

# API Keys
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
sec_key = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")

if not api_key:
    st.error("❌ Missing Gemini API key.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Gemini config failed: {e}")
    st.stop()

# --- Helpers ---
def safe_get(d, k, default=None):
    return d[k] if isinstance(d, dict) and k in d else default

def safe_format(val):
    try:
        if val is None:
            return "N/A"
        val = float(val)
        if abs(val) >= 1e12:
            return f"${val/1e12:.2f}T"
        if abs(val) >= 1e9:
            return f"${val/1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"${val/1e6:.2f}M"
        return f"${val:,.2f}"
    except Exception:
        return str(val)

# --- SEC Fetcher ---
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4):
    if not sec_key:
        st.error("❌ Missing SEC API key.")
        return []

    url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(10-Q OR 10-K)",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    headers = {"Authorization": sec_key, "Content-Type": "application/json"}

    try:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        filings = r.json().get("filings", [])
    except Exception as e:
        st.error(f"❌ Filing query failed: {e}")
        return []

    filings_data = []
    for filing in filings:
        f_date = filing.get("filedAt")
        form = filing.get("formType")
        accession = filing.get("accessionNo")
        htm_url = filing.get("linkToHtml") or filing.get("linkToFilingDetails")

        st.write(f"🔎 DEBUG: Filing {form} filedAt={f_date} candidates:\n\n• htm-url: {htm_url}\n\n• accession-no: {accession}\n")

        if not htm_url:
            continue

        try:
            conv_url = "https://api.sec-api.io/xbrl-to-json"
            resp = requests.get(conv_url, params={"url": htm_url}, headers={"Authorization": sec_key})
            st.write(f"🔎 DEBUG: Converter status for htm-url={htm_url}: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.warning(f"⚠️ Converter failed for {htm_url}: {e}")
            continue

        revenue, net_income, eps = None, None, None

        # Try multiple possible locations
        candidates = [
            safe_get(data, "incomeStatement"),
            safe_get(data, "StatementsOfIncome"),
            safe_get(data, "StatementsOfOperations"),
        ]
        found = False
        for block in candidates:
            if isinstance(block, list) and block:
                last = block[-1]
                for fact in last:
                    c, v = fact.get("concept"), fact.get("value")
                    if c == "Revenues":
                        revenue = v
                    if c == "NetIncomeLoss":
                        net_income = v
                    if c == "EarningsPerShareDiluted":
                        eps = v
                found = True
                break

        # Fallback: check facts
        if not found and "facts" in data:
            facts = data["facts"]
            revenue = revenue or safe_get(facts, "Revenues")
            net_income = net_income or safe_get(facts, "NetIncomeLoss")
            eps = eps or safe_get(facts, "EarningsPerShareDiluted")

        if not any([revenue, net_income, eps]):
            st.warning(f"⚠️ No income statement data for filing {htm_url}")
            continue

        filings_data.append({
            "date": f_date,
            "type": form,
            "revenue": revenue,
            "net_income": net_income,
            "eps": eps,
        })

    return filings_data

# --- Market Data ---
@st.cache_data(ttl=900)
def fetch_market_data(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed: {e}")
        return pd.DataFrame()

# --- AI Analysis ---
def analyze_with_ai(sec_data, market_data):
    prompt = f"""
You are an equity analyst. Analyze the company based on this data:

SEC Filings (latest):
{json.dumps(sec_data[:3], indent=2)}

Market Data (last 5 days):
{market_data.tail().to_dict() if not market_data.empty else "N/A"}

Return JSON with keys:
- overall_grade (string)
- recommendation (Buy/Sell/Hold/Neutral)
- investment_thesis (string)
- financial_health (string or dict)
- key_strengths (list or string)
- key_risks (list or string)
"""
    try:
        model = genai.GenerativeModel("gemini-pro")
        res = model.generate_content(prompt)
        raw = res.text.strip()
        cleaned = raw.strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except Exception:
        return {"investment_thesis": res.text if "res" in locals() else "No analysis"}

# --- Dashboard ---
def render_dashboard():
    st.title("📊 AI-Powered Earnings Intelligence Platform")

    ticker = st.text_input("Stock Ticker", "AAPL")
    quarters = st.slider("Quarters to analyze", 1, 12, 4)

    sec_data = fetch_sec_earnings(ticker, quarters)
    market_data = fetch_market_data(ticker, "6mo")

    tabs = st.tabs(["SEC Filings", "Market", "AI Analysis", "Debug"])

    with tabs[0]:
        st.subheader("SEC Filings Summary")
        if sec_data:
            df = pd.DataFrame(sec_data)
            df["revenue"] = df["revenue"].apply(safe_format)
            df["net_income"] = df["net_income"].apply(safe_format)
            df["eps"] = df["eps"].apply(safe_format)
            st.dataframe(df)

            chart_df = pd.DataFrame(sec_data)
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            for col in ["revenue", "net_income", "eps"]:
                chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
            fig = px.line(chart_df, x="date", y=["revenue", "net_income", "eps"], markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SEC filings available.")

    with tabs[1]:
        st.subheader("Market Data")
        if not market_data.empty:
            st.line_chart(market_data["Close"])
        else:
            st.info("No market data available.")

    with tabs[2]:
        st.subheader("AI Investment Analysis")
        if sec_data or not market_data.empty:
            ai = analyze_with_ai(sec_data, market_data)

            st.write(f"**Overall Grade:** {safe_get(ai, 'overall_grade', 'N/A')}")
            st.write(f"**Recommendation:** {safe_get(ai, 'recommendation', 'N/A')}")
            st.write("\n**Investment Thesis**")
            st.write(safe_get(ai, "investment_thesis", "N/A"))

            st.write("\n**Financial Health**")
            fh = safe_get(ai, "financial_health", "N/A")
            if isinstance(fh, dict):
                for k, v in fh.items():
                    st.write(f"- {k}: {v}")
            else:
                st.write(fh)

            st.write("\n**Key Strengths**")
            ks = safe_get(ai, "key_strengths", [])
            if isinstance(ks, list) and ks:
                for s in ks:
                    st.write(f"- {s}")
            else:
                st.write(ks if ks else "N/A")

            st.write("\n**Key Risks**")
            kr = safe_get(ai, "key_risks", [])
            if isinstance(kr, list) and kr:
                for r in kr:
                    st.write(f"- {r}")
            else:
                st.write(kr if kr else "N/A")
        else:
            st.info("No data available for AI analysis.")

    with tabs[3]:
        st.subheader("Debug Info")
        st.json(sec_data)

# --- Entry ---
if __name__ == "__main__":
    render_dashboard()