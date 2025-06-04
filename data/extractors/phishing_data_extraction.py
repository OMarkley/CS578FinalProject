import os
import csv
import re
from email import policy
from email.parser import BytesParser


def extract_clean_plain_text(msg):
    for part in msg.iter_parts():
        if part.get_content_type() == 'text/plain':
            raw_text = part.get_content().strip()
            cleaned_text = re.sub(r'\s+', ' ', raw_text)  # Normalize whitespace

            # Remove words containing 'jose' or 'monkey' (case-insensitive)
            filtered_words = [
                word for word in cleaned_text.split()
                if 'jose' not in word.lower() and 'monkey' not in word.lower()
            ]
            final_text = ' '.join(filtered_words)
            return final_text
    return None


def process_email_file(filepath, limit=100000):
    with open(filepath, "rb") as f:
        raw_data = f.read()

    # Prepare output file path
    directory = os.path.dirname(filepath)
    output_path = os.path.join(directory, "../parsed_data/monkey_phishing_2024.csv")

    raw_emails = raw_data.split(b"\nFrom jose@monkey.org")
    count = 0

    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'group', 'label', 'text'])
        writer.writeheader()

        for i, raw_email in enumerate(raw_emails):
            if not raw_email.strip():
                continue

            if i > 0:
                raw_email = b'From jose@monkey.org' + raw_email

            try:
                msg = BytesParser(policy=policy.default).parsebytes(raw_email)
            except Exception:
                continue

            if not msg.is_multipart():
                continue

            text = extract_clean_plain_text(msg)
            if text and len(text.split()) >= 5:
                writer.writerow({
                    'index': count,  # Index starts at 0
                    'group': 'monkey_phishing_2024',
                    'label': 0,  # PHISHING
                    'text': text
                })
                count += 1

            if count >= limit:
                break

    print(f"Wrote {count} phishing email(s) to: {output_path}")


# Run the parser
process_email_file("data/raw_data/monkey_phishing_2024.txt")
