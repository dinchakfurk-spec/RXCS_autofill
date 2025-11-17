from typing import Optional
import re


# Standard GSTIN: 2 digits (state code) + 10-char PAN (5 letters, 4 digits, 1 letter)
# + 1 entity code (alphanumeric) + 'Z' + 1 checksum (alphanumeric)
GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")


def extract_gstin(text: str) -> Optional[str]:
    """Return GSTIN if present in OCR text, else None."""
    if not text:
        return None
    m = GSTIN_REGEX.search(text.upper())
    return m.group(0) if m else None


