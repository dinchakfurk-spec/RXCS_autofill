import pytesseract
from pytesseract import Output
from openai import AzureOpenAI, OpenAI
import json
import logging
import os
import fitz  # PyMuPDF
import base64
from typing import Dict, Any, Optional, List
from PIL import Image, ImageOps, ImageFilter
from dotenv import load_dotenv
from services.payee_extractor import guess_payee_name, fallback_bank_fields
from services.gst_extractor import extract_gstin

logger = logging.getLogger(__name__)

class OCRProcessorService:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Tesseract setup
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR is available: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.warning(f"Tesseract OCR may not be properly configured: {e}")
        
        # OpenAI/Azure setup
        self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o')
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        if self.azure_api_key and self.azure_endpoint:
            self.client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.azure_endpoint
            )
            self.model = self.azure_model
            self.is_azure = True
            logger.info("Using Azure OpenAI")
        elif self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.model = self.openai_model
            self.is_azure = False
            logger.info("Using OpenAI")
        else:
            logger.warning("No OpenAI API key found")
            self.client = None
            self.is_azure = False
            self.model = self.openai_model
    
    def _convert_file_to_base64_images(self, file_path: str) -> List[str]:
        """Convert PDF or image file to list of base64-encoded images for GPT-4o Vision."""
        base64_images = []
        file_extension = os.path.splitext(file_path.lower())[1]
        
        try:
            if file_extension == '.pdf':
                # Allow tuning via env; default to fewer pages & slightly lower DPI for speed
                max_pages_env = os.getenv("RXCS_MAX_PAGES_PER_DOC")
                dpi_env = os.getenv("RXCS_PDF_DPI")
                try:
                    max_pages = int(max_pages_env) if max_pages_env else 3
                except ValueError:
                    max_pages = 3
                try:
                    pdf_dpi = int(dpi_env) if dpi_env else 200
                except ValueError:
                    pdf_dpi = 200

                logger.info(
                    f"Converting PDF to images: {file_path} "
                    f"(max_pages={max_pages}, dpi={pdf_dpi})"
                )
                pdf_document = fitz.open(file_path)
                page_count = len(pdf_document)
                zoom = pdf_dpi / 72.0
                matrix = fitz.Matrix(zoom, zoom)

                for page_num in range(min(page_count, max_pages)):
                    page = pdf_document[page_num]
                    # Render at configured DPI (balance of quality and size)
                    pix = page.get_pixmap(matrix=matrix)
                    # Convert to PNG bytes
                    img_bytes = pix.tobytes("png")
                    # Encode to base64
                    base64_image = base64.b64encode(img_bytes).decode('utf-8')
                    base64_images.append(base64_image)
                pdf_document.close()
            else:
                # For image files, read and convert to base64
                logger.info(f"Converting image to base64: {file_path}")
                with open(file_path, 'rb') as img_file:
                    img_bytes = img_file.read()
                    base64_image = base64.b64encode(img_bytes).decode('utf-8')
                    base64_images.append(base64_image)
            
            logger.info(f"Converted {len(base64_images)} page(s) to base64 images")
            return base64_images
        except Exception as e:
            logger.error(f"Error converting file to base64 images: {str(e)}")
            return []
    
    def base_hospital_schema(self) -> Dict[str, Any]:
        """Unified hospital payload with default empty values."""
        return {
            "hospital_name": "",
            "gst": "",
            "hospital_address": "",
            "hospital_pincode": "",
            "hospital_state": "",
            "hospital_city": "",
            "hospital_country": "",
            "payee_name": "",
            "bank_name": "",
            "account_number": "",
            "ifsc_code": "",
            "bank_branch": "",
            "hospital_email": "",
            "hospital_contact_number": "",
            "rohini_id_num": "",
            "rohini_exp_date": "",
            "no_of_beds": "",
            "staff_members": [],
            "user_email": "",
            "user_password": "",
            "subvention_for_reimbursement": "",
            "processing_fee_reimbursement": "",
            "cashless_desk_subvention": "",
            "subvention_no_cost_emi": "",
            "processing_fee_cashless": "",
            "processing_fee_nocost_emi": "",
            "onetime_verification_fee": "",
            "amount_collection_from_patient": "",
            "fixed_rate": "",
            "start_date": "",
            "end_date": "",
            "hospital_rates": []
        }

    def _guess_payee_name(self, text: str) -> Optional[str]:
        """Heuristic fallback to infer payee/account holder name from cheque OCR."""
        try:
            import re
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            # Prefer "For <ENTITY>" pattern
            # Case-insensitive, capture entity till line end
            m = re.search(r"\bFor\s+([A-Za-z0-9 &.,()'\-]{6,})", text, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" -,.()")
            # Line before Authorised/Authorized Signatory
            for i, ln in enumerate(lines):
                if 'Authorised Signatory' in ln or 'Authorized Signatory' in ln:
                    for j in range(max(0, i-3), i):
                        cand = lines[j].strip()
                        if 8 <= len(cand) <= 64:
                            if not any(bad in cand for bad in ['BANK', 'BRANCH', 'IFSC', 'CODE', 'DATE', 'INDIA', 'MUMBAI','SQUARE']):
                                return cand
            # Company-like uppercase fallback
            for cand in lines:
                if cand.upper() == cand and any(tok in cand for tok in ['PRIVATE LIMITED','PVT','LTD','LIMITED','CONSULTANT','CONSULTANTS','HOSPITAL']):
                    if not any(bad in cand for bad in ['BANK','BRANCH','IFSC','INDIA','MUMBAI','SQUARE']):
                        return cand
        except Exception:
            pass
        return None

    def _extract_email_phone(self, text: str) -> Dict[str, str]:
        """Extract first email and 10+ digit phone (handles +91, spaces, dashes)."""
        import re
        email = ""
        phone = ""
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if m:
            email = m.group(0)
        # phone: pick last 10 digits from a candidate
        pm = re.search(r"(?:\+?91[-\s]?)?(\d[\d\s-]{8,})", text)
        if pm:
            digits = re.sub(r"\D", "", pm.group(0))
            if len(digits) >= 10:
                phone = digits[-10:]
        return {"email": email, "phone": phone}

    def _extract_gstin(self, text: str) -> str:
        """Extract Indian GSTIN if present using regex."""
        import re
        # Standard GSTIN format: 2 digits + 10 PAN (5 letters, 4 digits, 1 letter) + 1 entity + Z + 1 checksum
        pattern = r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b"
        m = re.search(pattern, text.upper())
        return m.group(0) if m else ""

    def _fallback_bank_fields(self, text: str) -> Dict[str, str]:
        """Heuristic extraction for IFSC, account number, bank name and branch from cheque OCR text."""
        import re
        out = {"ifsc_code": "", "account_number": "", "bank_name": "", "bank_branch": ""}
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
            if 'BANK' in ln:
                # Keep alpha tokens and LTD if present
                tokens = [t for t in re.split(r"[^A-Z]", ln.upper()) if t]
                name = ' '.join([t for t in tokens if t and t != 'IFS' and t != 'IFSC'])
                # Re-hydrate with spaces from original line up to 'BANK'
                out["bank_name"] = ' '.join([w for w in ln.split() if any(x in w.upper() for x in ['BANK','LTD']) or w.isalpha()])
                break
        # Branch: look for 'BRANCH' or 'BR ' and keep rest of line
        for ln in upper_lines:
            if 'BRANCH' in ln:
                out["bank_branch"] = ln.strip()
                break
            if ' BR ' in f' {ln.upper()} ' or ln.upper().startswith('BR '):
                out["bank_branch"] = ln.strip()
                break
        return out

    def _infer_location(self, text: str) -> Dict[str, str]:
        """Infer pincode/city/state from OCR text (simple heuristics for India)."""
        import re
        pincode = ""
        city = ""
        state = ""
        # pincode
        pm = re.search(r"\b(\d{6})\b", text)
        if pm:
            pincode = pm.group(1)
        # state list (partial common)
        states = [
            'Maharashtra','Gujarat','Karnataka','Tamil Nadu','Telangana','Delhi','Uttar Pradesh','Madhya Pradesh',
            'West Bengal','Bihar','Rajasthan','Punjab','Haryana','Kerala','Odisha','Assam','Chhattisgarh','Jharkhand'
        ]
        upper = text.upper()
        for s in states:
            if s.upper() in upper:
                state = s
                break
        # city guess: take word before pincode if present
        if pincode:
            before = text[:text.find(pincode)].strip()
            tokens = [t.strip(" ,:\n\t-") for t in before.split() if t.strip()]
            if tokens:
                city = tokens[-1].strip(',')
        return {"pincode": pincode, "city": city, "state": state}

    def extract_text_from_document(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a single document using Tesseract OCR
        """
        try:
            file_extension = os.path.splitext(file_path.lower())[1]
            
            extracted_text = ""
            pages_processed = 0
            
            if file_extension == '.pdf':
                logger.info(f"Processing PDF file: {file_path}")
                # Open PDF with PyMuPDF
                pdf_document = fitz.open(file_path)
                
                for page_num in range(len(pdf_document)):
                    logger.info(f"Processing page {page_num+1}/{len(pdf_document)}")
                    
                    # Get the page
                    page = pdf_document[page_num]

                    # Pass 1: try to read the embedded text layer directly
                    page_text = (page.get_text("text") or "").strip()

                    # If no text layer, fall back to multi-pass OCR
                    if not page_text:
                        # 300 DPI render
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        page_text = self._ocr_image_multi_pass(img)

                        # If still low yield, try 400 DPI render
                        if not page_text or len(page_text) < 20:
                            pix_hd = page.get_pixmap(matrix=fitz.Matrix(400/72, 400/72))
                            img_hd = Image.frombytes("RGB", [pix_hd.width, pix_hd.height], pix_hd.samples)
                            page_text = self._ocr_image_multi_pass(img_hd)
                    extracted_text += page_text + "\n"
                    pages_processed += 1
                
                pdf_document.close()
            else:
                logger.info(f"Processing image file: {file_path}")
                image = Image.open(file_path)
                extracted_text = self._ocr_image_multi_pass(image)
                pages_processed = 1
            
            return {
                'success': True,
                'text': extracted_text.strip(),
                'pages_processed': pages_processed
            }
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return {
                'success': False,
                'text': '',
                'error': str(e)
            }
    
    def process_registration_certificate(self, file_path: str) -> Dict[str, Any]:
        """
        Process Registration Certificate and extract specific fields using GPT-4o Vision:
        - Hospital Name
        - Hospital Address
        - Pincode
        - State
        - City
        - Country
        - Number of Beds
        """
        try:
            # Convert file to base64 images
            logger.info("Converting Registration Certificate to images for GPT-4o Vision...")
            base64_images = self._convert_file_to_base64_images(file_path)
            
            if not base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert document to images"
                }
            
            # Prepare content with images
            content = [
                {
                    "type": "text",
                    "text": """You are extracting specific fields from a Hospital Registration Certificate document.

Extract ONLY these fields (if not present, return empty string):
- Hospital Name
- Hospital Address
- Pincode
- State
- City
- Country
- Number of Beds
- Email (if visible)
- Contact Number/Phone (if visible)

Return ONLY valid JSON with EXACTLY these keys:
{
  "hospital_name": "",
  "hospital_address": "",
  "pincode": "",
  "state": "",
  "city": "",
  "country": "",
  "number_of_beds": "",
  "email": "",
  "phone": ""
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Registration Certificate to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)

            # Map into unified schema
            schema = self.base_hospital_schema()
            schema["hospital_name"] = extracted_fields.get("hospital_name", "")
            addr = extracted_fields.get("hospital_address", "") or extracted_fields.get("address", "")
            schema["hospital_address"] = addr
            schema["hospital_pincode"] = extracted_fields.get("pincode", "")
            schema["hospital_state"] = extracted_fields.get("state", "")
            schema["hospital_city"] = extracted_fields.get("city", "")
            schema["hospital_country"] = extracted_fields.get("country", "")
            schema["no_of_beds"] = extracted_fields.get("number_of_beds", "")
            if extracted_fields.get("email"): 
                schema["hospital_email"] = extracted_fields.get("email", "")
            if extracted_fields.get("phone"): 
                schema["hospital_contact_number"] = extracted_fields.get("phone", "")

            return {
                'success': True,
                'document_type': 'registration_certificate',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': len(base64_images)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Registration Certificate processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_cancelled_check(self, file_path: str) -> Dict[str, Any]:
        """
        Process Cancelled Check and extract specific fields using GPT-4o Vision:
        - Payee Name
        - Bank Name
        - Account Number
        - IFSC Code
        - Bank Branch
        """
        try:
            # Convert file to base64 images
            logger.info("Converting Cancelled Check to images for GPT-4o Vision...")
            base64_images = self._convert_file_to_base64_images(file_path)
            
            if not base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert document to images"
                }
            
            # Prepare content with images
            content = [
                {
                    "type": "text",
                    "text": """You are extracting fields from a Cancelled Cheque document.

Guidelines:
- 'payee_name' means the ACCOUNT HOLDER name printed on the cheque (person/company).
- It is NOT the bank name, branch name, IFSC, or the words 'Authorized/Authorised Signatory'.
- If you see phrases like 'For <ENTITY>', that <ENTITY> is usually the account holder.
- Prefer long uppercase entity names over short words. If truly not present, leave empty.

Extract ONLY these fields (empty string if missing):
- Payee Name (account holder name)
- Bank Name
- Account Number (should be 9-18 digits)
- IFSC Code (format: 4 letters, 0, 6 alphanumeric)
- Bank Branch (full branch address/name)

Return ONLY valid JSON with EXACTLY these keys:
{
  "payee_name": "",
  "bank_name": "",
  "account_number": "",
  "ifsc_code": "",
  "bank_branch": ""
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Cancelled Check to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)

            # Map into unified schema
            schema = self.base_hospital_schema()
            schema["payee_name"] = extracted_fields.get("payee_name", "")
            schema["bank_name"] = extracted_fields.get("bank_name", "")
            schema["account_number"] = extracted_fields.get("account_number", "")
            schema["ifsc_code"] = extracted_fields.get("ifsc_code", "")
            schema["bank_branch"] = extracted_fields.get("bank_branch", "")

            return {
                'success': True,
                'document_type': 'cancelled_check',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': len(base64_images)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Cancelled Check processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_rohini_id(self, file_path: str) -> Dict[str, Any]:
        """
        Process Rohini ID and extract specific fields using GPT-4o Vision:
        - Rohini ID Number
        - Rohini Expiry Date
        """
        try:
            # Convert file to base64 images
            logger.info("Converting Rohini ID to images for GPT-4o Vision...")
            base64_images = self._convert_file_to_base64_images(file_path)
            
            if not base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert document to images"
                }
            
            # Prepare content with images
            content = [
                {
                    "type": "text",
                    "text": """You are extracting specific fields from a Rohini ID document.

Extract ONLY these fields (if not present, return empty string):
- Rohini ID Number
- Rohini Expiry Date

Return ONLY valid JSON with EXACTLY these keys:
{
  "rohini_id_number": "",
  "rohini_expiry_date": ""
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Rohini ID to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)

            schema = self.base_hospital_schema()
            schema["rohini_id_num"] = extracted_fields.get("rohini_id_number", "")
            schema["rohini_exp_date"] = extracted_fields.get("rohini_expiry_date", "")

            return {
                'success': True,
                'document_type': 'rohini_id',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': len(base64_images)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Rohini ID processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_hospital_rate_list(self, file_path: str) -> Dict[str, Any]:
        """
        Process Hospital Rate List and extract multiple rates with fields using GPT-4o Vision:
        For each rate:
        - Room Type
        - Bed Charges
        - Doctor Charges
        - Nursing Charges
        - Special Dr Charges
        - Special Nurse Charges
        - GST (%)
        """
        try:
            # Convert file to base64 images
            logger.info("Converting Hospital Rate List to images for GPT-4o Vision...")
            base64_images = self._convert_file_to_base64_images(file_path)
            
            if not base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert document to images"
                }
            
            # Prepare content with images
            content = [
                {
                    "type": "text",
                    "text": """You are extracting hospital rates from a Hospital Rate List document.

Extract ALL rates present in the document. Each rate should include:
- Room Type
- Bed Charges
- Doctor Charges
- Nursing Charges
- Special Dr Charges
- Special Nurse Charges
- GST (%)

If any field is not present for a rate, return empty string.

Return ONLY valid JSON with this structure:
{
  "rates": [
    {
      "rate_number": "1",
      "room_type": "",
      "bed_charges": "",
      "doctor_charges": "",
      "nursing_charges": "",
      "special_dr_charges": "",
      "special_nurse_charges": "",
      "gst_percentage": ""
    }
  ],
  "hospital_pincode": "",
  "hospital_state": "",
  "hospital_city": ""
}

Extract as many rates as present in the document. Also extract location information if visible."""
                }
            ]
            
            # Add all pages as images
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Hospital Rate List to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)

            schema = self.base_hospital_schema()
            rates_out = []
            for idx, r in enumerate(extracted_fields.get("rates", []) or [], start=1):
                rates_out.append({
                    "room_type": r.get("room_type", ""),
                    "custom_room_type": r.get("custom_room_type", ""),
                    "bed_charges": r.get("bed_charges", ""),
                    "doctor_charges": r.get("doctor_charges", ""),
                    "nursing_charges": r.get("nursing_charges", ""),
                    "special_dr_charges": r.get("special_dr_charges", ""),
                    "special_nurse_charges": r.get("special_nurse_charges", "")
                })
            if extracted_fields.get("hospital_pincode"): 
                schema["hospital_pincode"] = extracted_fields.get("hospital_pincode", "")
            if extracted_fields.get("hospital_state"): 
                schema["hospital_state"] = extracted_fields.get("hospital_state", "")
            if extracted_fields.get("hospital_city"): 
                schema["hospital_city"] = extracted_fields.get("hospital_city", "")
            schema["hospital_rates"] = rates_out

            return {
                'success': True,
                'document_type': 'hospital_rate_list',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': len(base64_images)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Hospital Rate List processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_hospital_gst(self, file_path: str) -> Dict[str, Any]:
        """
        Process Hospital GST document and extract specific fields using GPT-4o Vision:
        - GST Number
        """
        try:
            # Convert file to base64 images
            logger.info("Converting Hospital GST document to images for GPT-4o Vision...")
            base64_images = self._convert_file_to_base64_images(file_path)
            
            if not base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert document to images"
                }
            
            # Prepare content with images
            content = [
                {
                    "type": "text",
                    "text": """You are extracting the GST registration number from a hospital GST/invoice document.

Map any of these labels to a single field gst_number: GST, GSTIN, GST No., GST TIN, GSTIN/UIN, GST Registration No.
If multiple are present, choose the primary GSTIN for the hospital/issuer. If not found, return empty string.

GSTIN format is typically: 2 digits (state code) + 10 characters (PAN) + 1 entity code + Z + 1 checksum = 15 characters total.

Return ONLY valid JSON with EXACTLY these keys:
{
  "gst_number": ""
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Hospital GST document to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)
            
            schema = self.base_hospital_schema()
            gst = extracted_fields.get("gst_number", "")
            schema["gst"] = gst

            return {
                'success': True,
                'document_type': 'hospital_gst',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': len(base64_images)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Hospital GST processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def process_signed_mou(self, file_path: str) -> Dict[str, Any]:
        """
        Process Signed MOU document and extract as many fields as possible into the unified hospital schema.
        """
        try:
            logger.info("Converting Signed MOU to images for GPT-4o Vision...")
            base64_images = self._convert_file_to_base64_images(file_path)

            if not base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert document to images"
                }

            schema_template = self.base_hospital_schema()
            content = [
                {
                    "type": "text",
                    "text": f"""You are extracting structured data from a Signed MOU (Memorandum of Understanding) between a hospital and an insurer/TPA.

The target JSON schema has EXACTLY these keys (fill as many as possible; leave others empty strings or empty lists):
{json.dumps(schema_template, indent=2)}

Rules:
- If a value is clearly present, populate it verbatim.
- If unsure or not present, leave it empty string "" (or [] for lists).
- Do NOT add or remove keys; return ONLY this schema.
"""
                }
            ]

            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

            if not self.client:
                raise ValueError("No OpenAI client configured")

            logger.info("Sending Signed MOU to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()

            extracted_fields = json.loads(response_text)

            schema = self.base_hospital_schema()
            for key in schema.keys():
                if key in extracted_fields and extracted_fields[key] is not None:
                    schema[key] = extracted_fields[key]

            return {
                'success': True,
                'document_type': 'signed_mou',
                'extracted_text': '',
                'fields': schema,
                'pages_processed': len(base64_images)
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for Signed MOU: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Signed MOU processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    def _ocr_image_multi_pass(self, image: Image.Image) -> str:
        """Run multiple OCR passes with progressive enhancement for low-quality inputs."""
        try:
            variants = []
            base = image
            # Grayscale and autocontrast
            gray = ImageOps.autocontrast(base.convert('L'))
            variants.append(gray)
            # Sharpen
            variants.append(gray.filter(ImageFilter.SHARPEN))
            # Thresholded variants (simple thresholds)
            for th in (100, 130, 160):
                try:
                    bw = gray.point(lambda p, t=th: 255 if p > t else 0)
                    variants.append(bw)
                except Exception:
                    pass
            # Upscaled variants
            for scale in (1.5, 2.0, 3.0):
                try:
                    up = gray.resize((int(gray.width * scale), int(gray.height * scale)))
                    variants.append(up)
                    variants.append(up.filter(ImageFilter.SHARPEN))
                except Exception:
                    pass

            results: list[str] = []
            for v in variants:
                for psm in (6, 4, 11, 3):
                    for oem in (1, 3):
                        try:
                            txt = pytesseract.image_to_string(
                                v, lang='eng', config=f'--oem {oem} --psm {psm}'
                            )
                            txt = (txt or '').strip()
                            if txt:
                                results.append(txt)
                        except Exception:
                            pass

            if not results:
                return ""
            # Prefer the longest text as heuristic
            results.sort(key=lambda s: len(s), reverse=True)
            return results[0]
        except Exception:
            return ""

