# 1️⃣ SEC EDGAR Earnings Scraper
@st.cache_data(ttl=3600, show_spinner="📋 Fetching SEC filings...")
def fetch_sec_earnings(ticker, quarters=4):
    """
    Fetch recent 10-Q/10-K filings from SEC EDGAR and extract real data.
    """
    filings_data = []
    
    # 1. Look up CIK using a Ticker-to-CIK mapping
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': 'Financial Analyzer App/1.0 (info@example.com)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        ticker_data = response.json()
        
        # Check if the response is a list before proceeding
        if not isinstance(ticker_data, list):
            st.error("❌ SEC API returned unexpected data format for ticker lookup.")
            return []

        cik = None
        for company in ticker_data:
            # Add a check to ensure 'company' is a dictionary and has the 'ticker' key
            if isinstance(company, dict) and company.get('ticker') == ticker:
                cik = str(company['cik_str']).zfill(10)
                break
        
        if not cik:
            st.warning(f"⚠️ CIK for {ticker} not found.")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch CIK for {ticker}: {e}")
        return []
    except json.JSONDecodeError:
        st.error("❌ Failed to decode JSON from SEC ticker lookup API.")
        return []

    # The rest of the function remains the same.
    # 2. Get recent filings from the submissions API
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {'User-Agent': 'Financial Analyzer App/1.0 (info@example.com)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        filings = response.json().get('filings', {}).get('recent', {})
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch filings for CIK {cik}: {e}")
        return []
    except json.JSONDecodeError:
        st.error("❌ Failed to decode JSON from SEC submissions API.")
        return []
    
    # 3. Filter for 10-Q and 10-K reports
    found_count = 0
    if filings.get('accessionNumber'):
        for i in range(len(filings['accessionNumber'])):
            if found_count >= quarters:
                break
                
            form_type = filings['form'][i]
            if form_type in ['10-Q', '10-K']:
                accession_number = filings['accessionNumber'][i].replace('-', '')
                report_date = filings['filingDate'][i]
                
                # Construct URL to the XBRL file (more reliable)
                xbrl_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/Financial_Report.xml"
                
                try:
                    xbrl_response = requests.get(xbrl_url, headers=headers)
                    xbrl_response.raise_for_status()
                    
                    # Simple XML/XBRL parsing to find key values
                    soup = BeautifulSoup(xbrl_response.content, 'xml')
                    
                    latest_date_context = soup.find(lambda tag: tag.name.endswith('context') and 'instant' in tag.prettify())
                    
                    if not latest_date_context:
                        continue

                    revenue_tags = ['us-gaap:Revenues', 'us-gaap:NetSales']
                    net_income_tags = ['us-gaap:NetIncomeLoss']
                    eps_tags = ['us-gaap:EarningsPerShareDiluted']
                    
                    revenue, net_income, eps = None, None, None
                    
                    # Iterate through tags to find values
                    for tag in revenue_tags:
                        found_tag = soup.find(tag, contextref=latest_date_context.get('id'))
                        if found_tag and found_tag.text:
                            revenue = float(found_tag.text)
                            break

                    for tag in net_income_tags:
                        found_tag = soup.find(tag, contextref=latest_date_context.get('id'))
                        if found_tag and found_tag.text:
                            net_income = float(found_tag.text)
                            break
                            
                    for tag in eps_tags:
                        found_tag = soup.find(tag, contextref=latest_date_context.get('id'))
                        if found_tag and found_tag.text:
                            eps = float(found_tag.text)
                            break

                    filing_summary = {
                        'date': report_date,
                        'type': form_type,
                        'period': f"Q{found_count + 1}",
                        'revenue': revenue,
                        'net_income': net_income,
                        'eps': eps,
                        'url': xbrl_url
                    }
                    filings_data.append(filing_summary)
                    found_count += 1
                    
                except requests.exceptions.RequestException as e:
                    st.warning(f"⚠️ Failed to fetch or parse XBRL for {ticker} filing on {report_date}: {e}")
                except Exception as e:
                    st.warning(f"⚠️ Error parsing financial data for {ticker} filing on {report_date}: {e}")

    return filings_data
