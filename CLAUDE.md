# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
python3 -m pip install requests tqdm

# Download SEC filings for a company (by ticker)
python3 download_sec_filings.py --ticker MSFT

# Download specific form types only
python3 download_sec_filings.py --ticker AAPL --forms 10-K 10-Q

# Download all form types
python3 download_sec_filings.py --ticker TSLA --forms ALL

# Cap number of downloads
python3 download_sec_filings.py --ticker NVDA --max 50

# Use CIK directly instead of ticker lookup
python3 download_sec_filings.py --cik 1045810
```

> Use `python3` and `python3 -m pip` — bare `python`/`pip` are not on PATH on this machine.

## Architecture

Single-script ingestion tool (`download_sec_filings.py`) that pulls SEC filings from EDGAR for any public company.

**Flow:**
1. Resolve CIK — either directly from `--cik`, or by fetching `https://www.sec.gov/files/company_tickers.json` and matching on ticker symbol
2. Fetch filing metadata — calls `https://data.sec.gov/submissions/CIK<padded>.json`, handles EDGAR's pagination (extra `files[]` pages)
3. Filter & cap — applies `--forms` filter and optional `--max` cap
4. Download primary documents — streams each filing's primary document (HTM or PDF) from `https://www.sec.gov/Archives/edgar/data/<CIK>/<folder>/<primary>`; skips already-downloaded files (resumable)

**Output convention:** All downloaded data lives under `SEC/` — never in the project root.

```
SEC/<TICKER>/<year>/<form_type>/<TICKER>_<form>_<date>_<accession>.<ext>
```

**SEC rate limiting:** The script sleeps 0.15s between requests (`SLEEP_BETWEEN_REQUESTS`). Do not remove this — SEC servers will throttle or block without it.

**Known limitation:** Very old filings (pre-~2001) may 404 on the primary document URL even though the metadata exists. This is normal; those files were never digitized or are no longer hosted.

## EDGAR API reference

- Company tickers → CIK: `https://www.sec.gov/files/company_tickers.json`
- Submissions metadata: `https://data.sec.gov/submissions/CIK{10-digit-padded}.json`
- Filing document: `https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/{primary_doc}`
- All requests require a `User-Agent` header (SEC requirement).
