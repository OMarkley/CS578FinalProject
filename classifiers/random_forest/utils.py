import re
import pandas as pd

def load_phishing_emails_as_dataframe(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    emails = re.split(r"--- Email #[0-9]+ ---", raw)
    emails = [email.strip() for email in emails if email.strip()]

    df = pd.DataFrame({
        "text": emails,
        "label": 0,
        "group": "modern_phishing"  # updated group name
    })
    return df
