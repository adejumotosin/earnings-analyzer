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
from bs4 import BeautifulSoup
import pypdf
import io
import tempfile

# RAG Libraries
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

# OCR Libraries
from google.cloud import vision_v1
import fitz  # PyMuPDF, used for converting PDF pages to images

# -------------------------
# Configuration & Setup
# -------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
SEC_API_KEY = st.secrets.get("SEC_API_KEY") or os.environ.get("SEC_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in Streamlit secrets or env.")
    st.stop()

if not SEC_API_KEY:
    st.error("❌ Missing SEC API key. Please set SEC_API_KEY in Streamlit secrets or env.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()

# Initialize Google Cloud Vision client
try:
    vision_client = vision_v1.ImageAnnotatorClient()
except Exception as e:
    st.error(f"❌ Failed to initialize Google Cloud Vision client. Make sure GOOGLE_APPLICATION_CREDENTIALS is set. Error: {e}")
    vision_client = None

# -------------------------
# Utility helpers
# -------------------------
def safe_get(d: Dict, k: str, default="N/A"):
    return d.get(k, default) if isinstance(d, dict) else default

def parse_number(value: Any) -> Optional[float]:
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
        return extract_fact_value(obj[0])
    if isinstance(obj, str):
        return parse_number(obj)
    return None

def format_currency(value: Optional[float]) -> str:
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
    try:
        num = None
        if isinstance(val, (int, float)):
            num = float(val)
        else:
            num = parse_number(val)
        return format_currency(num) if num is not None else "N/A"
    except Exception:
        return "N/A"
        
@st.cache_data(ttl=3600)
def get_filing_text(url: str, debug: bool = False) -> str:
    """
    Downloads and extracts text from a HTML or PDF filing URL.
    Uses pypdf for text-based PDFs and Google Cloud Vision for image-based PDFs.
    """
    if not url:
        return ""
    
    try:
        response = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        content = response.content

        if url.lower().endswith(".htm"):
            soup = BeautifulSoup(content, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            text = soup.get_text(separator=' ', strip=True)
            return " ".join(text.split())

        elif url.lower().endswith(".pdf"):
            pdf_file_obj = io.BytesIO(content)
            
            # 1. Try to extract text with pypdf first
            try:
                reader = pypdf.PdfReader(pdf_file_obj)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() or "" + "\n"
                
                if full_text.strip():
                    return full_text
            except Exception as e:
                if debug:
                    st.warning(f"⚠️ pypdf failed (likely scanned PDF): {e}. Attempting OCR.")

            # 2. Fallback to Google Cloud Vision API for OCR
            if vision_client:
                st.warning("⚠️ PDF has no embedded text. Running OCR with Google Cloud Vision.")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name

                    doc = fitz.open(tmp_file_path)
                    ocr_text = ""
                    for i, page in enumerate(doc):
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        image = vision_v1.Image(content=img_bytes)
                        
                        response = vision_client.document_text_detection(image=image)
                        
                        if response.full_text_annotation and response.full_text_annotation.text:
                            ocr_text += response.full_text_annotation.text + "\n"
                    
                    os.unlink(tmp_file_path)
                    
                    if ocr_text.strip():
                        return ocr_text
                    else:
                        st.error("❌ Google Cloud Vision OCR failed to extract text.")
                        return ""
                
                except Exception as e:
                    if debug:
                        st.error(f"❌ Google Cloud Vision API call failed: {e}")
                    return ""
            else:
                st.warning("⚠️ Google Cloud Vision client not initialized. Cannot perform OCR.")
                return ""

        return ""

    except Exception as e:
        if debug:
            st.error(f"Failed to download or parse filing from {url}: {e}")
        return ""

# -------------------------
# RAG Pipeline Setup
# -------------------------
@st.cache_resource(ttl=3600)
def prepare_rag_pipeline(filing_texts: List[str]):
    """
    Sets up a RAG pipeline from a list of filing texts.
    Caches the vector store for faster subsequent runs.
    """
    if not filing_texts:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    for text in filing_texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend([Document(page_content=c) for c in chunks])

    if not all_chunks:
        st.warning("No text was extracted from the filings to build the RAG pipeline.")
        return None

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings_model)
    
    return vector_store.as_retriever(search_kwargs={"k": 5})

# -------------------------
# SEC Filing Text & Data Fetcher
# -------------------------
@st.cache_data(ttl=3600)
def fetch_sec_filings_and_narratives(ticker: str, quarters: int = 4, debug: bool = False) -> List[Dict]:
    filings_data: List[Dict] = []
    
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
        return []

    if not filings:
        if debug:
            st.info("No filings returned from sec-api.io for that ticker.")
        return []

    xbrl_converter_endpoint = "https://api.sec-api.io/xbrl-to-json"
    
    for filing in filings:
        xbrl_json = None
        filing_url = None
        doc_files = filing.get("documentFormatFiles", [])

        filing_url = next((doc.get("documentUrl") for doc in doc_files if doc.get("documentUrl", "").lower().endswith((".htm", ".html"))), None)
        if not filing_url:
            filing_url = next((doc.get("documentUrl") for doc in doc_files if doc.get("documentUrl", "").lower().endswith(".pdf")), None)
        
        try:
            resp_xbrl = requests.get(xbrl_converter_endpoint, params={"accessionNo": filing.get("accessionNo")}, headers={"Authorization": SEC_API_KEY}, timeout=30)
            resp_xbrl.raise_for_status()
            xbrl_json = resp_xbrl.json()
        except Exception as e:
            if debug:
                st.warning(f"⚠️ Could not convert filing to XBRL-JSON: {e}")
        
        filings_data.append({
            "filed_at": filing.get("filedAt"),
            "period": filing.get("periodOfReport"),
            "type": filing.get("formType"),
            "accession_number": filing.get("accessionNo"),
            "filing_url": filing_url,
            "full_xbrl_json": xbrl_json
        })
    return filings_data

# -------------------------
# Market data (yfinance)
# -------------------------
@st.cache_data(ttl=300)
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
            "price_history": hist["Close"].tolist()[-90:],
            "dates": [d.strftime("%Y-%m-%d") for d in hist.index.tolist()[-90:]]
        }
    except Exception as e:
        st.warning(f"⚠️ Market data fetch failed: {e}")
        return {}

# -------------------------
# AI Analysis (Gemini) - RAG-BASED
# -------------------------
@st.cache_data(ttl=7200)
def analyze_earnings_with_ai(ticker: str, sec_filings: List[Dict], market_data: Dict, retriever):
    if not retriever:
        st.error("No documents found for analysis. Please check your SEC filings.")
        return {}
    
    numerical_data = []
    for f in sec_filings:
        xbrl_json = f.get("full_xbrl_json", {})
        facts = xbrl_json.get("facts", {})
        numerical_summary = {
            "filed_at": f.get("filed_at"),
            "period": f.get("period"),
            "revenue": extract_fact_value(facts.get("us-gaap:Revenues")) or extract_fact_value(facts.get("us-gaap:SalesRevenueNet")),
            "net_income": extract_fact_value(facts.get("us-gaap:NetIncomeLoss")),
            "eps": extract_fact_value(facts.get("us-gaap:EarningsPerShareDiluted"))
        }
        numerical_data.append(numerical_summary)

    questions = [
        "What are the key drivers of financial performance?",
        "What is the company's stated strategy for the future?",
        "What are the most significant risk factors?",
        "What forward-looking statements are made?"
    ]
    
    retrieved_narrative = {}
    for q in questions:
        try:
            docs = retriever.invoke(q)
        except AttributeError:
            # Fallback for older LangChain versions
            docs = retriever.get_relevant_documents(q)
        retrieved_narrative[q] = " ".join([doc.page_content for doc in docs])

    try:
        prompt_template = f"""
You are a professional financial analyst. Provide a comprehensive analysis based on the structured financial data and retrieved narrative excerpts below.

Use only the provided context. Do not use outside knowledge.

Structured Financial Data (from XBRL):
{json.dumps(numerical_data, indent=2)}

Market Data:
{json.dumps(market_data, indent=2)}

Narrative Excerpts from SEC Filings (Retrieved via RAG):
{json.dumps(retrieved_narrative, indent=2)}

Your analysis must cover the following points:
1.  **Financial Performance and Trends:**
    * Describe the revenue and profitability trends using the structured data.
    * Explain the key drivers behind these trends based on the narrative.

2.  **Strategic Insights & Outlook:**
    * Summarize the company's strategy and future plans based on the narrative.
    * Identify any key forward-looking statements or projections.

3.  **Key Risks:**
    * Identify and summarize the most significant risk factors described in the narrative.

4.  **Overall Assessment:**
    * Provide a final investment thesis.
    * Assign a letter grade (A-F, where A is excellent, F is poor) and a recommendation (Buy, Hold, or Sell).

Respond with a single JSON object with the keys: 'financial_performance', 'strategic_insights', 'key_risks', and 'overall_assessment'. The 'overall_assessment' should be a nested object with 'investment_thesis', 'overall_grade', and 'recommendation'.
"""
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        response = llm.invoke(prompt_template)
        
        return json.loads(response.content)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ AI analysis failed to return valid JSON: {e}. Raw (truncated): {response.content[:2000] if 'response' in locals() else 'NO RESPONSE'}")
        return {}
    except Exception as e:
        st.error(f"⚠️ AI analysis failed: {e}")
        return {}


# -------------------------
# Dashboard Rendering & UI
# -------------------------
def display_analysis(analysis: Dict):
    if not analysis:
        st.warning("AI analysis data is missing or incomplete.")
        return

    st.markdown("### 🤖 Detailed AI Financial Analysis")

    if "overall_assessment" in analysis:
        final_summary = analysis["overall_assessment"]
        col1, col2 = st.columns([1, 2])
        with col1:
            grade = final_summary.get("overall_grade", "N/A")
            rec = final_summary.get("recommendation", "N/A")
            st.metric("Overall Grade", grade)
            st.metric("Recommendation", rec)
        with col2:
            thesis = final_summary.get("investment_thesis", "N/A")
            st.markdown(f"**Investment Thesis:**\n> {thesis}")

    if "financial_performance" in analysis:
        st.markdown("#### Financial Performance and Trends")
        st.markdown(analysis.get("financial_performance", "N/A"))

    if "strategic_insights" in analysis:
        st.markdown("#### Strategic Insights & Outlook")
        st.markdown(analysis.get("strategic_insights", "N/A"))
        
    if "key_risks" in analysis:
        st.markdown("#### Key Risks")
        st.markdown(analysis.get("key_risks", "N/A"))
        
    with st.expander("Show Raw AI Output"):
        st.json(analysis)

def render_dashboard():
    st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Earnings Intelligence Platform")
    st.markdown("---")
    st.info("Enter a stock ticker to get a comprehensive AI-generated earnings analysis based on SEC filing narratives.")

    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="AAPL, MSFT, GOOGL").strip().upper()
    with c2:
        quarters = st.selectbox("Quarters to analyze", [1,2,3,4], index=1)

    debug = st.checkbox("Show SEC debug logs", value=False)

    if st.button("🔍 Analyze Earnings", type="primary", use_container_width=True):
        if not ticker:
            st.warning("Please enter ticker symbol.")
            st.stop()

        with st.spinner("🔄 Collecting SEC filings and data..."):
            sec_data = fetch_sec_filings_and_narratives(ticker, quarters, debug=debug)
            
        with st.spinner("📝 Extracting text from filings and building RAG pipeline..."):
            filing_texts = [get_filing_text(f["filing_url"], debug=debug) for f in sec_data if f["filing_url"]]
            retriever = prepare_rag_pipeline(filing_texts)

        with st.spinner("📈 Fetching market data..."):
            market_data = fetch_market_data(ticker, days=90)
            
        st.subheader("Source Data Overview")
        tab1, tab2, tab3 = st.tabs(["📋 SEC Filings", "📝 Extracted Narrative", "📈 Market"])

        with tab1:
            if sec_data:
                table_data = []
                for f in sec_data:
                    xbrl_facts = f.get("full_xbrl_json", {}).get("facts", {})
                    revenue = extract_fact_value(xbrl_facts.get("us-gaap:Revenues")) or extract_fact_value(xbrl_facts.get("us-gaap:SalesRevenueNet"))
                    net_income = extract_fact_value(xbrl_facts.get("us-gaap:NetIncomeLoss"))
                    eps = extract_fact_value(xbrl_facts.get("us-gaap:EarningsPerShareDiluted"))
                    table_data.append({
                        "filed_at": f.get("filed_at"),
                        "period": f.get("period"),
                        "type": f.get("type"),
                        "revenue": safe_format(revenue),
                        "net_income": safe_format(net_income),
                        "eps": safe_format(eps)
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.info("No SEC filing data available.")

        with tab2:
            if sec_data:
                for filing in sec_data:
                    with st.expander(f"Narrative from {filing['type']} ({filing['filed_at']})"):
                        filing_text = get_filing_text(filing["filing_url"], debug=debug)
                        if not filing_text:
                            st.info("No text extracted from this filing.")
                        else:
                            st.text_area("Extracted Text (Truncated)", value=filing_text[:5000] + "...", height=300)
            else:
                st.info("No narrative text available.")
        
        with tab3:
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
        
        st.markdown("---")
        with st.spinner("🧠 Generating comprehensive analysis..."):
            analysis = analyze_earnings_with_ai(ticker, sec_data, market_data, retriever)

        display_analysis(analysis)

if __name__ == "__main__":
    render_dashboard()
