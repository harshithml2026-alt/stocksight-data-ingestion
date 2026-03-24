"""
Download SEC filings from SEC EDGAR for any company.

Usage:
  pip install requests tqdm
  python download_sec_filings.py --ticker NVDA
  python download_sec_filings.py --ticker MSFT --forms 10-K 10-Q
  python download_sec_filings.py --cik 1045810 --max 20
  python download_sec_filings.py --ticker AAPL --forms 10-K
"""

import os
import time
import argparse
import requests
from tqdm import tqdm

# -- DEFAULTS -----------------------------------------------------------------
DEFAULT_FORM_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A"]
SLEEP_BETWEEN_REQUESTS = 0.15  # seconds -- be polite to SEC servers
# -----------------------------------------------------------------------------

HEADERS = {"User-Agent": "stocksight-ingestion contact@example.com"}
BASE_URL = "https://data.sec.gov"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def lookup_cik(ticker: str) -> tuple[str, str]:
    """Return (cik, company_name) for a ticker symbol. Raises if not found."""
    print(f"Looking up CIK for ticker '{ticker.upper()}'...")
    resp = requests.get(COMPANY_TICKERS_URL, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            cik = str(entry["cik_str"])
            name = entry["title"]
            print(f"  Found: {name} (CIK: {cik})")
            return cik, name

    raise ValueError(
        f"Ticker '{ticker}' not found in SEC EDGAR. "
        "Check the ticker symbol or provide a CIK directly with --cik."
    )


def get_all_submissions(cik: str) -> list[dict]:
    """Fetch all filings metadata from the EDGAR submissions API."""
    url = f"{BASE_URL}/submissions/CIK{cik.zfill(10)}.json"
    print(f"Fetching submissions from:\n  {url}\n")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    results = _parse_filings_page(filings)

    # EDGAR paginates -- fetch additional pages if they exist
    for page_ref in data["filings"].get("files", []):
        page_url = f"{BASE_URL}/submissions/{page_ref['name']}"
        print(f"  Fetching extra page: {page_ref['name']}")
        r = requests.get(page_url, headers=HEADERS)
        r.raise_for_status()
        results.extend(_parse_filings_page(r.json()))
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return results


def _parse_filings_page(filings: dict) -> list[dict]:
    return [
        {
            "accession": filings["accessionNumber"][i],
            "form":      filings["form"][i],
            "date":      filings["filingDate"][i],
            "primary":   filings["primaryDocument"][i],
        }
        for i in range(len(filings["accessionNumber"]))
    ]


def download_file(url: str, dest_path: str) -> bool:
    """Download a file to dest_path. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR filings for any company.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--ticker", metavar="SYMBOL",
                          help="Company ticker symbol (e.g. NVDA, AAPL, MSFT)")
    id_group.add_argument("--cik", metavar="CIK",
                          help="SEC EDGAR CIK number (e.g. 1045810)")

    parser.add_argument("--forms", nargs="+", metavar="FORM",
                        default=DEFAULT_FORM_TYPES,
                        help=f"Form types to download (default: {DEFAULT_FORM_TYPES}). "
                             "Use 'ALL' to download every form type.")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="Max number of filings to download (default: all)")
    parser.add_argument("--output", default="./SEC", metavar="DIR",
                        help="Base output directory (default: ./SEC). "
                             "Files are saved to <DIR>/<TICKER>/<year>/<form>/")

    args = parser.parse_args()

    # Resolve CIK and ticker label
    if args.ticker:
        cik, company_name = lookup_cik(args.ticker)
        label = args.ticker.upper()
    else:
        cik = args.cik.lstrip("0") or "0"
        cik = args.cik  # keep original for API calls
        company_name = f"CIK{args.cik}"
        label = f"CIK{args.cik}"
        print(f"Using CIK: {args.cik}")

    output_dir = os.path.join(args.output, label)
    os.makedirs(output_dir, exist_ok=True)

    form_filter = None if (len(args.forms) == 1 and args.forms[0].upper() == "ALL") else args.forms

    # Fetch all filings metadata
    all_filings = get_all_submissions(cik)
    print(f"Total filings found: {len(all_filings)}")

    # Filter by form type
    if form_filter:
        filings = [f for f in all_filings if f["form"] in form_filter]
        print(f"After filtering to {form_filter}: {len(filings)} filings")
    else:
        filings = all_filings
        print("No form filter applied — downloading all form types")

    # Apply cap
    if args.max:
        filings = filings[:args.max]
        print(f"Capped at {args.max} filings")

    # Download
    total_downloaded = 0
    total_skipped = 0

    for filing in tqdm(filings, desc=f"Downloading {label} filings"):
        form    = filing["form"].replace("/", "_")
        date    = filing["date"]
        acc     = filing["accession"]
        primary = filing["primary"]

        if not primary:
            total_skipped += 1
            continue

        folder = acc.replace("-", "")
        file_url = f"{ARCHIVES_URL}/{cik}/{folder}/{primary}"

        year = date[:4]
        year_dir = os.path.join(output_dir, year, form)
        os.makedirs(year_dir, exist_ok=True)

        ext = os.path.splitext(primary)[1] or ".htm"
        filename = f"{label}_{form}_{date}_{acc}{ext}"
        dest = os.path.join(year_dir, filename)

        if os.path.exists(dest):
            continue  # Already downloaded

        success = download_file(file_url, dest)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if success:
            total_downloaded += 1
        else:
            total_skipped += 1

    print(f"\nDone! ({company_name})")
    print(f"   Files downloaded : {total_downloaded}")
    print(f"   Skipped          : {total_skipped}")
    print(f"   Output folder    : {os.path.abspath(output_dir)}")
    print(f"   Structure        : {label}/<year>/<form_type>/{label}_<form>_<date>_<accession>.<ext>")


if __name__ == "__main__":
    main()
