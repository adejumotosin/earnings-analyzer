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
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4):
    """
    Fetch recent financial data by combining CIK lookup and filing search
    with a dedicated financial statements API from sec-api.io.
    """
    filings_data = []

    api_key = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")
    if not api_key:
        st.error("❌ Missing SEC API key. Please set it in your Streamlit secrets.")
        return []
    
    # Step 1: Use the Query API to find the most recent 10-Q and 10-K filings
    query_url = "https://api.sec-api.io"
    payload = {
        "query": f"ticker:{ticker} AND formType:(\"10-Q\" OR \"10-K\")",
        "from": "0",
        "size": quarters,
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    
    try:
        query_response = requests.post(
            query_url, json=payload, headers=headers
        )
        query_response.raise_for_status()
        filings = query_response.json().get("filings", [])
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch filing URLs from sec-api.io: {e}")
        return []
    
    if not filings:
        st.info(f"No 10-Q or 10-K filings found for {ticker}.")
        return []

    # Step 2: For each filing, extract financial data
    for filing in filings:
        try:
            # Method 1: Try to get the XBRL document URL directly
            xbrl_url = None
            for file in filing.get("documentFormatFiles", []):
                if file.get("documentUrl", "").endswith(".xml"):
                    xbrl_url = file["documentUrl"]
                    break

            # If no direct XBRL URL, construct a fallback
            if not xbrl_url:
                accession_number = filing.get("accessionNo", "").replace("-", "")
                cik = filing.get("cik", "")
                xbrl_filename = f"{ticker.lower()}-{filing.get('periodOfReport', '').replace('-', '')}.htm"
                xbrl_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{xbrl_filename}"
            
            # 🔎 DEBUG LOG
            st.write(f"🔎 DEBUG: Trying XBRL URL → {xbrl_url}")

            # Use the XBRL-to-JSON Converter API
            converter_url = "https://api.sec-api.io/xbrl-to-json"
            xbrl_response = requests.get(
                converter_url, 
                params={"url": xbrl_url}, 
                headers={"Authorization": api_key},
                timeout=30
            )

            # 🔎 DEBUG LOG for response
            st.write(f"🔎 DEBUG: Converter API status → {xbrl_response.status_code}, Content-Type: {xbrl_response.headers.get('Content-Type')}")

            xbrl_response.raise_for_status()
            data = xbrl_response.json()
            
            # Extract financial metrics
            revenue = None
            net_income = None
            eps = None
            
            income_statements = data.get("IncomeStatements", data.get("incomeStatement", []))
            
            if income_statements:
                latest_statement = income_statements[0] if isinstance(income_statements, list) else income_statements
                
                revenue_fields = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 
                                'SalesRevenueNet', 'RevenueFromContractWithCustomer']
                for field in revenue_fields:
                    if field in latest_statement:
                        revenue = latest_statement[field]
                        break
                
                income_fields = ['NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic',
                               'ProfitLoss']
                for field in income_fields:
                    if field in latest_statement:
                        net_income = latest_statement[field]
                        break
                
                eps_fields = ['EarningsPerShareDiluted', 'EarningsPerShareBasic']
                for field in eps_fields:
                    if field in latest_statement:
                        eps = latest_statement[field]
                        break
            
            if not revenue and not net_income:
                facts = data.get("facts", {})
                if facts:
                    for concept, values in facts.items():
                        if 'revenue' in concept.lower() and not revenue:
                            revenue = values.get("value") if isinstance(values, dict) else values[0].get("value") if values else None
                        elif 'netincome' in concept.lower() and not net_income:
                            net_income = values.get("value") if isinstance(values, dict) else values[0].get("value") if values else None
            
            filing_data = {
                'date': filing.get("filedAt"),
                'period': filing.get("periodOfReport"),
                'type': filing.get("formType"),
                'revenue': revenue,
                'eps': eps,
                'net_income': net_income,
                'accession_number': filing.get("accessionNo")
            }
            
            filings_data.append(filing_data)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                st.warning(f"⚠️ Could not process {filing.get('formType', 'filing')} from {filing.get('filedAt', 'unknown date')}: Invalid XBRL format or URL")
            else:
                st.warning(f"⚠️ HTTP error processing filing: {e}")
            continue
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Network error processing filing: {e}")
            continue
        except (KeyError, IndexError, TypeError) as e:
            st.warning(f"⚠️ Data parsing error for filing: {e}")
            continue

    return filings_data

# The rest of your code remains unchanged...
# fetch_earnings_transcripts, fetch_analyst_sentiment, fetch_market_data,
# analyze_earnings_with_ai, render_dashboard, display_ai_analysis

if __name__ == "__main__":
    render_dashboard()