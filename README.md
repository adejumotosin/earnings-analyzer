# Earnings Analyzer

A Streamlit-based equity research application that combines SEC filings, market data, OCR, retrieval-augmented generation, and Gemini-based analysis to support company earnings research.

The project is designed to reduce the manual work involved in reading quarterly and annual filings by turning filing text into searchable context for structured research workflows.

## Core workflow

```text
Ticker / company
      |
      +------> SEC filing retrieval
      |              |
      |              +------> HTML text extraction
      |              |
      |              +------> PDF extraction
      |                         |
      |                         +------> Google Cloud Vision OCR fallback
      |
      +------> Yahoo Finance market data
      |
      v
Filing text chunking
      |
      v
MiniLM embeddings + Chroma vector store
      |
      v
Semantic retrieval
      |
      v
Gemini-assisted earnings research
      |
      v
Streamlit analysis interface
```

## Implemented components

- SEC 10-Q and 10-K filing retrieval through SEC API
- HTML filing text extraction with BeautifulSoup
- PDF text extraction with `pypdf`
- OCR fallback for image-based filings using Google Cloud Vision and PyMuPDF
- Recursive text chunking for long filings
- `all-MiniLM-L6-v2` embeddings
- Chroma vector storage and semantic retrieval
- LangChain-based RAG workflow
- Google Gemini integration for generated research responses
- Yahoo Finance market data integration
- pandas-based data handling
- Plotly visualizations
- Streamlit interface

## Why this project matters

Traditional earnings analysis requires switching between filings, price data, charts, and manual notes. This project explores a single research workflow where filing content can be retrieved semantically and analyzed alongside market information.

The goal is not to replace primary-source review, but to make the research process faster and more structured.

## Run locally

```bash
git clone https://github.com/adejumotosin/earnings-analyzer.git
cd earnings-analyzer

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required credentials using Streamlit secrets or environment variables:

```text
GEMINI_API_KEY
SEC_API_KEY
GOOGLE_APPLICATION_CREDENTIALS
```

Then launch:

```bash
streamlit run app.py
```

## Technology

- Python
- Streamlit
- SEC API
- Yahoo Finance / yfinance
- Google Gemini
- Google Cloud Vision
- LangChain
- ChromaDB
- Sentence Transformer embeddings
- BeautifulSoup
- pypdf / PyMuPDF
- pandas
- Plotly

## Current limitations

- Generated analysis can contain errors and must be checked against the original filing.
- SEC API access and Gemini access require external credentials.
- OCR quality depends on the underlying filing quality.
- Yahoo Finance is suitable for research prototyping but is not institutional-grade market data.
- The current RAG pipeline retrieves relevant text but does not yet provide rigorous page-level citation provenance for every generated statement.

## Roadmap

- Add page-level and section-level citations to generated answers
- Add automatic quarter-over-quarter and year-over-year earnings comparison
- Add guidance-versus-actual tracking
- Add earnings surprise analysis
- Add peer and sector comparison
- Add valuation and estimate-revision modules
- Add persistent research sessions and company watchlists
- Add test sets for evaluating retrieval and answer accuracy

## Disclaimer

This project is a research tool and does not provide investment advice. Financial conclusions should be verified against primary filings and reliable market-data sources before use.
