from typing import Optional, Dict
import re

def guess_payee_name(text: str) -> Optional[str]:
    """Heuristic fallback to infer payee/account holder name from cheque OCR text."""
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Prefer "For <ENTITY>" pattern (case-insensitive)
        m = re.search(r"\bFor\s+([A-Za-z0-9 &.,()'\-]{6,})", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" -,.()")
        # Line before Authorised/Authorized Signatory
        for i, ln in enumerate(lines):
            if 'Authorised Signatory' in ln or 'Authorized Signatory' in ln:
                for j in range(max(0, i-3), i):
                    cand = lines[j].strip()
                    if 8 <= len(cand) <= 64 and not any(bad in cand.upper() for bad in ['BANK','BRANCH','IFSC','CODE','DATE','INDIA','MUMBAI','SQUARE']):
                        return cand
        # Company-like uppercase fallback
        for cand in lines:
            if cand.upper() == cand and any(tok in cand for tok in ['PRIVATE LIMITED','PVT','LTD','LIMITED','CONSULTANT','CONSULTANTS','HOSPITAL']):
                if not any(bad in cand.upper() for bad in ['BANK','BRANCH','IFSC','INDIA','MUMBAI','SQUARE']):
                    return cand
    except Exception:
        pass
    return None

def fallback_bank_fields(text: str) -> Dict[str, str]:
    """Heuristic extraction for IFSC, account number, bank name and branch from cheque OCR text."""
    out: Dict[str, str] = {"ifsc_code": "", "account_number": "", "bank_name": "", "bank_branch": ""}
    upper_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    U = text.upper()
    # IFSC
    ifsc_m = re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", U)
    if ifsc_m:
        out["ifsc_code"] = ifsc_m.group(0)
    # Account number: prefer 12-18 digits, else longest 9-18 digits
    nums = re.findall(r"\b\d{9,18}\b", text)
    nums = sorted(nums, key=lambda n: len(n), reverse=True)
    if nums:
        out["account_number"] = nums[0]
    # Bank name: first line containing 'BANK'
    for ln in upper_lines:
        if 'BANK' in ln.upper():
            tokens = [w for w in ln.split() if any(x in w.upper() for x in ['BANK','LTD']) or w.isalpha()]
            out["bank_name"] = ' '.join(tokens)
            break
    # Branch: look for 'BRANCH' or ' BR '
    for ln in upper_lines:
        if 'BRANCH' in ln.upper():
            out["bank_branch"] = ln.strip()
            break
        if ' BR ' in f' {ln.upper()} ' or ln.upper().startswith('BR '):
            out["bank_branch"] = ln.strip()
            break
    return out


