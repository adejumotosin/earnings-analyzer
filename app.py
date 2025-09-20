import os
import re
import json
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Configuration & Setup ---
# Configure AI
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set it in your Streamlit secrets.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()

# --- Utility Functions ---
def safe_get(data, key, default="N/A"):
    return data.get(key, default)

def safe_format(value):
    try:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            if value >= 1e12:
                return f"${value / 1e12:.2f}T"
            elif value >= 1e9:
                return f"${value / 1e9:.2f}B"
            elif value >= 1e6:
                return f"${value / 1e6:.2f}M"
            return f"${value:,.2f}"
        return str(value)
    except Exception:
        return str(value)

# --- Data Fetching Functions ---
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4):
    """
    Fetch financial data from sec-api.io using linkToXbrl or accession number.
    """
    filings_data = []

    api_key = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")
    if not api_key:
        st.error("❌ Missing SEC API key. Please set it in your Streamlit secrets.")
        return []

    query_url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    headers = {"Content-Type": "application/json", "Authorization": api_key}

    try:
        query_response = requests.post(query_url, json=payload, headers=headers)
        query_response.raise_for_status()
        filings = query_response.json().get("filings", [])
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch filing metadata: {e}")
        return []

    if not filings:
        st.info(f"No 10-Q or 10-K filings found for {ticker}.")
        return []

    for filing in filings:
        filing_date = filing.get("filedAt")
        form_type = filing.get("formType")
        filing_html = filing.get("linkToFilingDetails")
        accession_no = filing.get("accessionNo")

        st.write(f"🔎 DEBUG: Filing {form_type} filedAt={filing_date} candidates:")
        if filing_html:
            st.write(f"• htm-url: {filing_html}")
        if accession_no:
            st.write(f"• accession-no: {accession_no}")

        target_url = None
        params = {}
        if filing_html:
            target_url = filing_html
            params = {"htm-url": target_url}
        elif accession_no:
            target_url = accession_no
            params = {"accession-no": accession_no}
        else:
            continue

        try:
            xbrl_url = "https://api.sec-api.io/xbrl-to-json"
            xbrl_response = requests.get(
                xbrl_url, params=params, headers={"Authorization": api_key}
            )
            st.write(
                f"🔎 DEBUG: Converter status for {list(params.keys())[0]}={target_url}: {xbrl_response.status_code}"
            )
            xbrl_response.raise_for_status()
            data = xbrl_response.json()

            income_statements = data.get("incomeStatement", [])
            if not income_statements:
                st.warning(f"⚠️ No income statement data for filing {target_url}")
                continue

            last_statement = income_statements[-1]
            revenue, net_income, eps = None, None, None

            for fact in last_statement:
                if fact.get("concept") == "Revenues":
                    revenue = fact.get("value")
                elif fact.get("concept") == "NetIncomeLoss":
                    net_income = fact.get("value")
                elif fact.get("concept") == "EarningsPerShareDiluted":
                    eps = fact.get("value")

            filings_data.append(
                {
                    "filed_at": filing_date,
                    "period": filing.get("periodOfReport"),
                    "type": form_type,
                    "revenue": revenue,
                    "eps": eps,
                    "net_income": net_income,
                    "accession_number": accession_no,
                    "xbrl_source_used": list(params.keys())[0],
                    "xbrl_url": target_url,
                }
            )

        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Failed to parse filing {target_url}: {e}")
            continue

    return filings_data

def fetch_market_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=90)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch market data for {ticker}: {e}")
        return pd.DataFrame()

# --- AI Analysis Function ---
def analyze_with_ai(context):
    """
    Send context to Gemini AI and get structured analysis.
    """
    prompt = f"""
    Analyze the following company earnings and market context.
    Provide JSON with fields:
    overall_grade, investment_thesis, financial_health, key_strengths, key_risks, recommendation.

    Context:
    {context}
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis failed: {e}"

def render_ai_analysis(ai_text):
    try:
        data = json.loads(ai_text)
    except Exception:
        st.error("⚠️ AI returned unstructured response. Showing raw output:")
        st.write(ai_text)
        return

    st.subheader("AI Investment Analysis")

    st.markdown(f"**Overall Grade:** {safe_get(data, 'overall_grade')}")
    st.markdown(f"**Recommendation:** {safe_get(data, 'recommendation')}")

    st.markdown("**Investment Thesis**")
    st.write(safe_get(data, "investment_thesis"))

    st.markdown("**Financial Health**")
    fh = safe_get(data, "financial_health", {})
    if isinstance(fh, dict):
        for k, v in fh.items():
            st.write(f"- {k.replace('_',' ').title()}: {v}")

    strengths = safe_get(data, "key_strengths", [])
    if strengths:
        st.markdown("**Key Strengths**")
        for s in strengths:
            st.write(f"- {s}")

    risks = safe_get(data, "key_risks", [])
    if risks:
        st.markdown("**Key Risks**")
        for r in risks:
            st.write(f"- {r}")

# --- Dashboard Rendering ---
def render_dashboard():
    st.title("📊 AI-Powered Earnings Intelligence Platform")

    ticker = st.text_input("Stock Ticker", "AAPL")
    quarters = st.selectbox("Quarters to analyze", [1, 2, 4, 8], index=0)
    if not ticker:
        st.stop()

    if st.button("🔍 Analyze Earnings"):
        sec_data = fetch_sec_earnings(ticker, quarters)
        transcripts = []  # placeholder
        analysts = []     # placeholder
        market_data = fetch_market_data(ticker)

        st.header("Source Data Overview")
        tabs = st.tabs(["SEC Filings", "Transcripts", "Analysts", "Market"])

        with tabs[0]:
            if sec_data:
                df = pd.DataFrame(sec_data)
                for col in ["revenue", "net_income", "eps"]:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: safe_format(x) if x is not None else "N/A"
                        )
                st.dataframe(df, use_container_width=True)

                # Plot revenue, net income, EPS
                df_plot = pd.DataFrame(sec_data)
                if not df_plot.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_plot["period"], y=df_plot["revenue"], name="Revenue"
                    ))
                    fig.add_trace(go.Bar(
                        x=df_plot["period"], y=df_plot["net_income"], name="Net Income"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_plot["period"], y=df_plot["eps"], name="EPS", yaxis="y2"
                    ))
                    fig.update_layout(
                        title="Revenue & Net Income (bars) and EPS (line)",
                        yaxis=dict(title="Amount (USD)"),
                        yaxis2=dict(title="EPS", overlaying="y", side="right"),
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No SEC filings available.")

        with tabs[3]:
            if not market_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=market_data.index, y=market_data["Close"], mode="lines", name="Close"
                ))
                fig.update_layout(title="Stock Price (Last 90 Days)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No market data available.")

        # --- AI Section ---
        context = {
            "sec_filings": sec_data,
            "analyst_reports": analysts,
            "transcripts": transcripts,
        }
        ai_text = analyze_with_ai(json.dumps(context))
        render_ai_analysis(ai_text)

if __name__ == "__main__":
    render_dashboard()