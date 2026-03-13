import os
import json
import requests
import pdfplumber
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CGWB_URL = "https://cgwb.gov.in/sites/default/files/inline-files/january_wl_1994-2024-compressed.pdf"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_PATH = os.path.join(DATA_DIR, "cgwb_january_wl_1994_2024.pdf")
CSV_PATH = os.path.join(DATA_DIR, "cgwb_tables.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "cgwb_summary.json")


def ensure_data_dir():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def download_pdf(url: str = CGWB_URL, dest: str = PDF_PATH) -> str:
    ensure_data_dir()
    if os.path.isfile(dest):
        logger.info(f"PDF already exists at {dest}. Skipping download.")
        return dest
    logger.info(f"Downloading PDF from {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    logger.info(f"Download complete. Saved to {dest}.")
    return dest


def extract_tables_to_csv(pdf_path: str = PDF_PATH, csv_path: str = CSV_PATH, summary_path: str = SUMMARY_PATH) -> str:
    if os.path.isfile(csv_path):
        logger.info(f"CSV already exists at {csv_path}. Skipping extraction.")
        return csv_path
        
    logger.info(f"Starting PDF table extraction from {pdf_path}...")
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception as e:
                logger.warning(f"Error extracting tables on page {page_idx}: {e}")
                tables = []
            for t_idx, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue
                header = table[0]
                # Normalize header names
                header = [str(h).strip().lower().replace("\n", " ") if h is not None else f"col_{i}" for i, h in enumerate(header)]
                for row in table[1:]:
                    record = {header[i] if i < len(header) else f"col_{i}": (str(val).strip() if val is not None else None) for i, val in enumerate(row)}
                    record["_page"] = page_idx
                    record["_table"] = t_idx
                    rows.append(record)
        
    if not rows:
        logger.warning("No tables extracted from PDF. Creating empty CSV.")
        # Create empty CSV
        pd.DataFrame([]).to_csv(csv_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"pages": 0, "tables": 0, "rows": 0}, f, indent=2)
        return csv_path

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Extraction complete. {len(df)} rows saved to CSV.")

    # Basic summary
    summary = {
        "pages": df["_page"].max() if "_page" in df.columns else None,
        "tables": int(df["_table"].max()) if "_table" in df.columns else None,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path


def main():
    logger.info("Starting data ingestion process...")
    try:
        pdf = download_pdf()
        csv = extract_tables_to_csv(pdf)
        logger.info(f"Data ingestion successful.")
        logger.info(f"Saved CSV: {csv}")
        logger.info(f"Saved summary: {SUMMARY_PATH}")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()