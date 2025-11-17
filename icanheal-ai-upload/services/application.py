import pytesseract
from openai import AzureOpenAI, OpenAI
import json
import logging
import os
import fitz  # PyMuPDF
import base64
from typing import Dict, Any, Optional, List
from PIL import Image, ImageFilter, ImageOps
from dotenv import load_dotenv
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)

class InsuranceProcessorService:
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
        
        # Define insurance company and TPA provider lists
        self.insurance_companies = [
            "National Insurance Co. Ltd.",
            "Go Digit General Insurance Ltd.",
            "Allianz General Insurance Co. Ltd.",
            "MS General Insurance Co. Ltd.",
            "AXA General Insurance Co. Ltd.",
            "ERGO General Insurance Co. Ltd.",
            "Future Generali India Insurance Co. Ltd.",
            "New India Assurance Co. Ltd.",
            "Iffco Tokio General Insurance Co. Ltd.",
            "Reliance General Insurance Co. Ltd.",
            "Sundaram General Insurance Co. Ltd.",
            "The Oriental Insurance Co. Ltd.",
            "AIG General Insurance Co. Ltd.",
            "SBI General Insurance Co. Ltd.",
            "Acko General Insurance Ltd.",
            "Navi General Insurance Ltd.",
            "Edelweiss General Insurance Co. Ltd.",
            "ICICI Lombard General Insurance Co. Ltd.",
            "Mahindra General Insurance Co. Ltd.",
            "Liberty General Insurance Ltd.",
            "HDI General Insurance Co. Ltd.",
            "Raheja QBE General Insurance Co. Ltd.",
            "Shriram General Insurance Co. Ltd.",
            "United India Insurance Co. Ltd.",
            "Manipal Cigna Health Insurance Company Limited",
            "Birla Health Insurance Co. Ltd.",
            "Magma HDI General Insurance Co. Ltd.",
            "Max Bupa Health Insurance Company Ltd.",
            "Care Health Insurance Ltd.",
            "Universal Sompo General Insurance Co. Ltd.",
            "Zuno General Insurance Ltd.",
            "Zurich Kotak General Insurance"
        ]
        
        self.tpa_providers = [
            "Medi Assist Insurance",
            "MDIndia Health Insurance",
            "Paramount Health Insurance",
            "Heritage Health Insurance",
            "Family Health Plan Insurance",
            "Raksha Health Insurance",
            "Vidal Health Insurance",
            "Volo Health Insurance",
            "Medsave Health Insurance",
            "Genins India Insurance",
            "Health India Insurance",
            "Good Health Insurance",
            "Park Mediclaim Insurance",
            "Safeway Insurance",
            "Ericson Insurance"
        ]
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Apply light preprocessing to improve OCR accuracy."""
        try:
            # Grayscale
            processed = image.convert("L")
            # Improve contrast
            processed = ImageOps.autocontrast(processed)
            # Sharpen slightly
            processed = processed.filter(ImageFilter.SHARPEN)
            # Binarize
            processed = processed.point(lambda x: 0 if x < 160 else 255, '1')
            return processed
        except Exception:
            # If any preprocessing fails, return the original image
            return image

    def detect_gender(self, text: str) -> Optional[str]:
        """Detect gender using common Aadhaar card cues (English + Hindi)."""
        if not text:
            return None
        low = text.lower()

        # Label-based direct patterns
        labeled = re.search(r"\b(?:gender|sex)\s*[:\-]?\s*(male|female|m\b|f\b)\b", low)
        if labeled:
            val = labeled.group(1)
            if val in ("male", "m"):
                return "male"
            if val in ("female", "f"):
                return "female"

        # Nearby window after label
        for label_match in re.finditer(r"\b(?:gender|sex)\b", low):
            start = max(0, label_match.start())
            window = low[start:start + 40]
            if re.search(r"\bmale\b|\bm\b", window):
                return "male"
            if re.search(r"\bfemale\b|\bf\b", window):
                return "female"

        # Hindi tokens
        if re.search(r"पुरुष", text):
            return "male"
        if re.search(r"महिला", text):
            return "female"

        # Fuzzy english variants
        if re.search(r"\bm[a@][l1i][e3]\b", low):
            return "male"
        if re.search(r"\bf[e3]m[a@][l1i][e3]?\b", low):
            return "female"

        # Contextual pattern '/ male' or '/ female'
        if re.search(r"/\s*male\b", low):
            return "male"
        if re.search(r"/\s*female\b", low):
            return "female"

        # Final simple fallback
        if re.search(r"\bfemale\b", low):
            return "female"
        if re.search(r"\bmale\b", low):
            return "male"
        return None

    def detect_gender_fuzzy(self, text: str) -> Optional[str]:
        """Fuzzy gender detection for noisy OCR tokens."""
        if not text:
            return None
        low = re.sub(r"[^a-z]", "", text.lower())
        if not low:
            return None
        if self.calculate_similarity(low, "male") >= 0.7:
            return "male"
        if self.calculate_similarity(low, "female") >= 0.7:
            return "female"
        for tok in re.findall(r"[a-zA-Z]{3,10}", text):
            t = tok.lower()
            if self.calculate_similarity(t, "male") >= 0.75:
                return "male"
            if self.calculate_similarity(t, "female") >= 0.75:
                return "female"
        return None

    def try_gender_from_crops(self, file_path: str) -> Optional[str]:
        """Heuristic crops around where gender is usually printed on Aadhaar and OCR with focused configs."""
        try:
            img = Image.open(file_path)
        except Exception:
            return None

        width, height = img.size
        boxes_rel = [
            (0.10, 0.45, 0.60, 0.80),
            (0.20, 0.40, 0.75, 0.75),
            (0.45, 0.40, 0.95, 0.75),
        ]
        psms = [7, 6, 3, 13]
        langs = ['eng', 'eng+hin']

        for (lx, ly, rx, ry) in boxes_rel:
            box = (int(width*lx), int(height*ly), int(width*rx), int(height*ry))
            crop = img.crop(box)
            crop = crop.resize((crop.width*2, crop.height*2))
            crop = self._preprocess_image_for_ocr(crop)
            for lang in langs:
                for psm in psms:
                    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
                    try:
                        txt = pytesseract.image_to_string(crop, lang=lang, config=cfg)
                    except Exception:
                        try:
                            txt = pytesseract.image_to_string(crop, lang='eng', config=cfg)
                        except Exception:
                            txt = ''
                    g = self.detect_gender(txt) or self.detect_gender_fuzzy(txt)
                    if g:
                        return g
        return None
    
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
                    # Render at configured DPI
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
    
    def base_insurance_schema(self) -> Dict[str, Any]:
        """Unified insurance payload with default empty values."""
        return {
            # Policy-level
            "insurance_company": "",
            "tpa_provider": "",
            "tpa_id": "",
            "policy_type": "",
            "policy_number": "",
            "sum_insured": "",
            "balance_sum_insured": "",
            "co_pay_percentage": "",
            "policy_start_date": "",
            "policy_end_date": "",
            # Members list
            "members": [],
            # Pre-Auth block (flat top-level for simplicity)
            "hospital_name": "",
            "treating_doctor": "",
            "application_type": "",
            "estimated_treatment_cost": "",
            "preauth_reference_number": "",
            "tentative_admission_date": "",
            "patient_name": "",
            "diagnosis_medical_condition": ""
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching by removing extra spaces and converting to lowercase."""
        if not text:
            return ""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        # Remove common punctuation
        normalized = re.sub(r'[.,;:!?]', '', normalized)
        return normalized
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def find_best_insurance_company_match(self, ocr_text: str) -> str:
        """
        Find the best matching insurance company from the predefined list.
        Returns the exact company name from the list if a good match is found.
        """
        if not ocr_text or not ocr_text.strip():
            return ""
        
        ocr_normalized = self.normalize_text(ocr_text)
        best_match = ""
        best_score = 0.0
        
        for company in self.insurance_companies:
            company_normalized = self.normalize_text(company)
            
            # Check for exact match first
            if ocr_normalized == company_normalized:
                return company
            
            # Check if OCR text contains the company name
            if company_normalized in ocr_normalized:
                return company
            
            # Check if company name contains OCR text (for partial matches)
            if ocr_normalized in company_normalized:
                return company
            
            # Calculate similarity score
            similarity = self.calculate_similarity(ocr_normalized, company_normalized)
            
            # Also check similarity with key words from company name
            company_words = company_normalized.split()
            for word in company_words:
                if len(word) > 3:  # Only check words longer than 3 characters
                    word_similarity = self.calculate_similarity(ocr_normalized, word)
                    similarity = max(similarity, word_similarity)
            
            if similarity > best_score:
                best_score = similarity
                best_match = company
        
        # Return the best match if similarity is above threshold
        if best_score >= 0.6:  # 60% similarity threshold
            logger.info(f"Matched insurance company: '{ocr_text}' -> '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"No good match found for insurance company: '{ocr_text}' (best score: {best_score:.2f})")
        return ocr_text  # Return original text if no good match found
    
    def find_best_tpa_provider_match(self, ocr_text: str) -> str:
        """
        Find the best matching TPA provider from the predefined list.
        Returns the exact TPA name from the list if a good match is found.
        """
        if not ocr_text or not ocr_text.strip():
            return ""
        
        ocr_normalized = self.normalize_text(ocr_text)
        best_match = ""
        best_score = 0.0
        
        for tpa in self.tpa_providers:
            tpa_normalized = self.normalize_text(tpa)
            
            # Check for exact match first
            if ocr_normalized == tpa_normalized:
                return tpa
            
            # Check if OCR text contains the TPA name
            if tpa_normalized in ocr_normalized:
                return tpa
            
            # Check if TPA name contains OCR text (for partial matches)
            if ocr_normalized in tpa_normalized:
                return tpa
            
            # Calculate similarity score
            similarity = self.calculate_similarity(ocr_normalized, tpa_normalized)
            
            # Also check similarity with key words from TPA name
            tpa_words = tpa_normalized.split()
            for word in tpa_words:
                if len(word) > 3:  # Only check words longer than 3 characters
                    word_similarity = self.calculate_similarity(ocr_normalized, word)
                    similarity = max(similarity, word_similarity)
            
            if similarity > best_score:
                best_score = similarity
                best_match = tpa
        
        # Return the best match if similarity is above threshold
        if best_score >= 0.6:  # 60% similarity threshold
            logger.info(f"Matched TPA provider: '{ocr_text}' -> '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"No good match found for TPA provider: '{ocr_text}' (best score: {best_score:.2f})")
        return ocr_text  # Return original text if no good match found

    def extract_text_from_document(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a single document using Tesseract OCR
        """
        try:
            file_extension = os.path.splitext(file_path.lower())[1]
            
            extracted_text = ""
            pages_processed = 0
            
            tess_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"

            if file_extension == '.pdf':
                logger.info(f"Processing PDF file: {file_path}")
                # Open PDF with PyMuPDF
                pdf_document = fitz.open(file_path)
                
                for page_num in range(len(pdf_document)):
                    logger.info(f"Processing page {page_num+1}/{len(pdf_document)}")
                    
                    # Get the page
                    page = pdf_document[page_num]
                    
                    # Convert page to image (pixmap)
                    pix = page.get_pixmap(matrix=fitz.Matrix(400/72, 400/72))  # 400 DPI for better accuracy
                    
                    # Convert pixmap to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # Preprocess image for better OCR
                    img = self._preprocess_image_for_ocr(img)
                    
                    # Extract text using Tesseract
                    page_text = pytesseract.image_to_string(img, lang='eng', config=tess_config)
                    extracted_text += page_text + "\n"
                    pages_processed += 1
                
                pdf_document.close()
            else:
                logger.info(f"Processing image file: {file_path}")
                image = Image.open(file_path)
                image = self._preprocess_image_for_ocr(image)
                extracted_text = pytesseract.image_to_string(image, lang='eng', config=tess_config)
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
    
    def process_insurance_copy(self, file_paths: list) -> Dict[str, Any]:
        """
        Process Insurance Copy and extract specific fields:
        - Insurance Company
        - TPA Provider
        - TPA ID
        - Policy Type
        - Policy Number
        - Sum Insured
        - Balance Sum Insured
        - Co-Pay (%)
        - Policy Start Date
        - Policy End Date
        """
        try:
            # Step 1: Extract text from all files using OCR
            logger.info(f"Extracting text from {len(file_paths)} Insurance Copy files...")
            
            all_extracted_text = ""
            total_pages = 0
            processing_results = []
            
            for i, file_path in enumerate(file_paths):
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                ocr_result = self.extract_text_from_document(file_path)
                
                if not ocr_result['success']:
                    logger.warning(f"OCR failed for {file_path}: {ocr_result.get('error')}")
                    processing_results.append({
                        'file': file_path,
                        'success': False,
                        'error': ocr_result.get('error')
                    })
                    continue
                
                extracted_text = ocr_result['text']
                if extracted_text.strip():
                    all_extracted_text += f"\n\n--- Document {i+1} ---\n{extracted_text}"
                    total_pages += ocr_result['pages_processed']
                
                processing_results.append({
                    'file': file_path,
                    'success': True,
                    'pages_processed': ocr_result['pages_processed']
                })
            
            if not all_extracted_text.strip():
                return {
                    'success': False,
                    'error': "No text could be extracted from any of the documents",
                    'processing_results': processing_results
                }
            
            # Step 2: Send to LLM to extract specific fields
            logger.info("Extracting Insurance Copy fields using LLM...")
            
            prompt = f"""
You are extracting specific fields from an Insurance Copy document.

Extract ONLY these fields (if not present, return empty string):
- Insurance Company
- TPA Provider
- TPA ID
- Policy Type
- Policy Number
- Sum Insured
- Balance Sum Insured
- Co-Pay (%)
- Policy Start Date
- Policy End Date

OCR Text from all documents:
{all_extracted_text}

Return ONLY valid JSON with EXACTLY these keys:
{{
  "insurance_company": "",
  "tpa_provider": "",
  "tpa_id": "",
  "policy_type": "",
  "policy_number": "",
  "sum_insured": "",
  "balance_sum_insured": "",
  "co_pay_percentage": "",
  "policy_start_date": "",
  "policy_end_date": ""
}}
"""

            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": prompt}
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
            
            # Map to unified schema with matching
            schema = self.base_insurance_schema()
            
            # Apply matching for insurance company and TPA provider
            raw_insurance_company = extracted_fields.get("insurance_company", "")
            raw_tpa_provider = extracted_fields.get("tpa_provider", "")
            
            schema["insurance_company"] = self.find_best_insurance_company_match(raw_insurance_company)
            schema["tpa_provider"] = self.find_best_tpa_provider_match(raw_tpa_provider)
            
            # Other fields remain as extracted
            schema["tpa_id"] = extracted_fields.get("tpa_id", "")
            schema["policy_type"] = extracted_fields.get("policy_type", "")
            schema["policy_number"] = extracted_fields.get("policy_number", "")
            schema["sum_insured"] = extracted_fields.get("sum_insured", "")
            schema["balance_sum_insured"] = extracted_fields.get("balance_sum_insured", "")
            schema["co_pay_percentage"] = extracted_fields.get("co_pay_percentage", "")
            schema["policy_start_date"] = extracted_fields.get("policy_start_date", "")
            schema["policy_end_date"] = extracted_fields.get("policy_end_date", "")

            return {
                'success': True,
                'document_type': 'insurance_copy',
                'extracted_text': all_extracted_text,
                'fields': schema,
                'pages_processed': total_pages,
                'files_processed': len(file_paths),
                'processing_results': processing_results
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Insurance Copy processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_insurance_members(self, file_paths: list) -> Dict[str, Any]:
        """
        Process Insurance Copy and extract member details:
        For each member:
        - Member Type
        - Full Name
        - Residential Address
        - Pincode
        - State
        - City
        - Country
        - Phone Number
        - Email
        - Gender
        - Occupation
        - Salary/Business details
        - Aadhaar Number
        - PAN Number
        """
        try:
            # Step 1: Extract text from all files using OCR
            logger.info(f"Extracting text from {len(file_paths)} Insurance Copy files for member details...")
            
            all_extracted_text = ""
            total_pages = 0
            processing_results = []
            
            for i, file_path in enumerate(file_paths):
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                ocr_result = self.extract_text_from_document(file_path)
                
                if not ocr_result['success']:
                    logger.warning(f"OCR failed for {file_path}: {ocr_result.get('error')}")
                    processing_results.append({
                        'file': file_path,
                        'success': False,
                        'error': ocr_result.get('error')
                    })
                    continue
                
                extracted_text = ocr_result['text']
                if extracted_text.strip():
                    all_extracted_text += f"\n\n--- Document {i+1} ---\n{extracted_text}"
                    total_pages += ocr_result['pages_processed']
                
                processing_results.append({
                    'file': file_path,
                    'success': True,
                    'pages_processed': ocr_result['pages_processed']
                })
            
            if not all_extracted_text.strip():
                return {
                    'success': False,
                    'error': "No text could be extracted from any of the documents",
                    'processing_results': processing_results
                }
            
            # Step 2: Send to LLM to extract member details
            logger.info("Extracting Insurance Member details using LLM...")
            
            prompt = f"""
Extract member information from insurance documents.

For each member found, extract these fields:
- Member Type (Policy Holder/Patient/Caretaker)
- Full Name
- Residential Address
- Pincode
- State
- City
- Country
- Phone Number
- Email
- Gender
- Occupation
- Company/Firm Name
- Years of Experience
- Salary/Business Income
- Aadhaar Number
- PAN Number

Document text:
{all_extracted_text}

Return JSON format:
{{
  "members": [
    {{
      "member_type": "",
      "full_name": "",
      "residential_address": "",
      "pincode": "",
      "state": "",
      "city": "",
      "country": "",
      "phone_number": "",
      "email": "",
      "gender": "",
      "occupation": "",
      "company_firm_name": "",
      "years_of_experience": "",
      "salary_business_income": "",
      "aadhaar_number": "",
      "pan_number": ""
    }}
  ]
}}
"""

            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)
            
            schema = self.base_insurance_schema()
            members = extracted_fields.get("members", []) or []
            # Normalize member genders to lowercase
            for m in members:
                try:
                    if m.get("gender"):
                        gv = str(m["gender"]).strip().lower()
                        if gv in ("male", "female"):
                            m["gender"] = gv
                except Exception:
                    continue
            schema["members"] = members

            return {
                'success': True,
                'document_type': 'insurance_members',
                'extracted_text': all_extracted_text,
                'fields': schema,
                'pages_processed': total_pages,
                'files_processed': len(file_paths),
                'processing_results': processing_results
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Insurance Members processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_claim_filling_doc(self, file_paths: list) -> Dict[str, Any]:
        """
        Process Claim Filling Document and extract specific fields:
        - Hospital Name
        - Treating Doctor
        - Application Type
        - Hospital Bill Amount
        - Claim Intimation Number
        - Tentative Admission Date
        """
        try:
            # Step 1: Extract text from all files using OCR
            logger.info(f"Extracting text from {len(file_paths)} Claim Filling Document files...")
            
            all_extracted_text = ""
            total_pages = 0
            processing_results = []
            
            for i, file_path in enumerate(file_paths):
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                ocr_result = self.extract_text_from_document(file_path)
                
                if not ocr_result['success']:
                    logger.warning(f"OCR failed for {file_path}: {ocr_result.get('error')}")
                    processing_results.append({
                        'file': file_path,
                        'success': False,
                        'error': ocr_result.get('error')
                    })
                    continue
                
                extracted_text = ocr_result['text']
                if extracted_text.strip():
                    all_extracted_text += f"\n\n--- Document {i+1} ---\n{extracted_text}"
                    total_pages += ocr_result['pages_processed']
                
                processing_results.append({
                    'file': file_path,
                    'success': True,
                    'pages_processed': ocr_result['pages_processed']
                })
            
            if not all_extracted_text.strip():
                return {
                    'success': False,
                    'error': "No text could be extracted from any of the documents",
                    'processing_results': processing_results
                }
            
            # Step 2: Send to LLM to extract specific fields
            logger.info("Extracting Claim Filling Document fields using LLM...")
            
            prompt = f"""
Extract claim filling information from documents.

Extract these fields:
- Hospital Name
- Treating Doctor
- Application Type
- Hospital Bill Amount
- Claim Intimation Number
- Tentative Admission Date

Document text:
{all_extracted_text}

Return JSON format:
{{
  "hospital_name": "",
  "treating_doctor": "",
  "application_type": "",
  "hospital_bill_amount": "",
  "claim_intimation_number": "",
  "tentative_admission_date": ""
}}
"""

            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": prompt}
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
            
            schema = self.base_insurance_schema()
            schema["hospital_name"] = extracted_fields.get("hospital_name", "")
            schema["treating_doctor"] = extracted_fields.get("treating_doctor", "")
            schema["application_type"] = extracted_fields.get("application_type", "")
            schema["hospital_bill_amount"] = extracted_fields.get("hospital_bill_amount", "")
            schema["claim_intimation_number"] = extracted_fields.get("claim_intimation_number", "")
            schema["tentative_admission_date"] = extracted_fields.get("tentative_admission_date", "")

            return {
                'success': True,
                'document_type': 'claim_filling_doc',
                'extracted_text': all_extracted_text,
                'fields': schema,
                'pages_processed': total_pages,
                'files_processed': len(file_paths),
                'processing_results': processing_results
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Claim Filling Document processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_insurance_copy_only(self, file_paths: list) -> Dict[str, Any]:
        """
        Process ONLY Insurance Copy to extract policy details and member count using GPT-4o Vision.
        This should be called FIRST to determine how many members we have.
        """
        try:
            # Convert all files to base64 images
            logger.info(f"Converting {len(file_paths)} Insurance Copy files to images for GPT-4o Vision...")
            
            all_base64_images = []
            total_pages = 0
            processing_results = []
            
            for i, file_path in enumerate(file_paths):
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                base64_images = self._convert_file_to_base64_images(file_path)
                
                if not base64_images:
                    logger.warning(f"Failed to convert {file_path} to images")
                    processing_results.append({
                        'file': file_path,
                        'success': False,
                        'error': "Failed to convert to images"
                    })
                    continue
                
                all_base64_images.extend(base64_images)
                total_pages += len(base64_images)
                
                processing_results.append({
                    'file': file_path,
                    'success': True,
                    'pages_processed': len(base64_images)
                })
            
            if not all_base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert any documents to images",
                    'processing_results': processing_results
                }
            
            # Prepare content with all images
            content = [
                {
                    "type": "text",
                    "text": """You are extracting specific fields from an Insurance Copy document.

Extract ONLY these fields (if not present, return empty string):
- Insurance Company
- TPA Provider
- TPA ID
- Policy Type
- Policy Number
- Sum Insured
- Balance Sum Insured
- Co-Pay (%)
- Policy Start Date
- Policy End Date
- Number of Members (count how many members are covered)
- Member Names (list all member names found)
- Member Genders (list gender for each member name in same order)

Return ONLY valid JSON with EXACTLY these keys:
{
  "insurance_company": "",
  "tpa_provider": "",
  "tpa_id": "",
  "policy_type": "",
  "policy_number": "",
  "sum_insured": "",
  "balance_sum_insured": "",
  "co_pay_percentage": "",
  "policy_start_date": "",
  "policy_end_date": "",
  "number_of_members": 0,
  "member_names": [],
  "member_genders": []
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in all_base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Insurance Copy to GPT-4o Vision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract structured data from documents. Return only valid JSON."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            extracted_fields = json.loads(response_text)
            
            # Map to unified schema with matching
            schema = self.base_insurance_schema()
            
            # Apply matching for insurance company and TPA provider
            raw_insurance_company = extracted_fields.get("insurance_company", "")
            raw_tpa_provider = extracted_fields.get("tpa_provider", "")
            
            schema["insurance_company"] = self.find_best_insurance_company_match(raw_insurance_company)
            schema["tpa_provider"] = self.find_best_tpa_provider_match(raw_tpa_provider)
            
            # Other fields remain as extracted
            schema["tpa_id"] = extracted_fields.get("tpa_id", "")
            schema["policy_type"] = extracted_fields.get("policy_type", "")
            schema["policy_number"] = extracted_fields.get("policy_number", "")
            schema["sum_insured"] = extracted_fields.get("sum_insured", "")
            schema["balance_sum_insured"] = extracted_fields.get("balance_sum_insured", "")
            schema["co_pay_percentage"] = extracted_fields.get("co_pay_percentage", "")
            schema["policy_start_date"] = extracted_fields.get("policy_start_date", "")
            schema["policy_end_date"] = extracted_fields.get("policy_end_date", "")

            # Normalize genders list to lowercase
            normalized_member_genders = []
            try:
                for g in extracted_fields.get("member_genders", []) or []:
                    gv = str(g).strip().lower()
                    if gv in ("male", "female"):
                        normalized_member_genders.append(gv)
            except Exception:
                normalized_member_genders = extracted_fields.get("member_genders", []) or []

            return {
                'success': True,
                'document_type': 'insurance_copy_only',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': total_pages,
                'files_processed': len(file_paths),
                'processing_results': processing_results,
                'number_of_members': extracted_fields.get("number_of_members", 0),
                'member_names': extracted_fields.get("member_names", []),
                'member_genders': normalized_member_genders
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Insurance Copy processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_member_documents(self, member_index: int, aadhaar_files: list = None, pan_files: list = None, salary_files: list = None) -> Dict[str, Any]:
        """
        Process individual member documents (Aadhaar, PAN, Salary) for a specific member using GPT-4o Vision.
        This should be called AFTER insurance copy processing to get member count.
        
        Args:
            member_index: Index of the member (0-based)
            aadhaar_files: List of Aadhaar card files for this member
            pan_files: List of PAN card files for this member  
            salary_files: List of salary slip files for this member
        """
        try:
            # Convert all files to base64 images
            logger.info(f"Converting member documents to images for GPT-4o Vision...")
            
            all_base64_images = []
            total_pages = 0
            processing_results = []
            document_types = []
            
            # Process Aadhaar files
            if aadhaar_files:
                for file_path in aadhaar_files:
                    base64_images = self._convert_file_to_base64_images(file_path)
                    if base64_images:
                        all_base64_images.extend(base64_images)
                        total_pages += len(base64_images)
                        document_types.extend(['aadhaar'] * len(base64_images))
                    processing_results.append({
                        'file': file_path,
                        'document_type': 'aadhaar',
                        'success': bool(base64_images),
                        'error': None if base64_images else "Failed to convert to images"
                    })
            
            # Process PAN files
            if pan_files:
                for file_path in pan_files:
                    base64_images = self._convert_file_to_base64_images(file_path)
                    if base64_images:
                        all_base64_images.extend(base64_images)
                        total_pages += len(base64_images)
                        document_types.extend(['pan'] * len(base64_images))
                    processing_results.append({
                        'file': file_path,
                        'document_type': 'pan',
                        'success': bool(base64_images),
                        'error': None if base64_images else "Failed to convert to images"
                    })
            
            # Process Salary files
            if salary_files:
                for file_path in salary_files:
                    base64_images = self._convert_file_to_base64_images(file_path)
                    if base64_images:
                        all_base64_images.extend(base64_images)
                        total_pages += len(base64_images)
                        document_types.extend(['salary'] * len(base64_images))
                    processing_results.append({
                        'file': file_path,
                        'document_type': 'salary',
                        'success': bool(base64_images),
                        'error': None if base64_images else "Failed to convert to images"
                    })
            
            if not all_base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert any member documents to images",
                    'processing_results': processing_results
                }
            
            # Prepare content with all images
            content = [
                {
                    "type": "text",
                    "text": """Extract member information from documents (Aadhaar card, PAN card, Salary slip).

Extract these fields:
- Full Name
- Residential Address
- Pincode
- State
- City
- Country
- Phone Number
- Email
- Gender (male/female)
- Occupation
- Company/Firm Name
- Years of Experience
- Salary/Business Income
- Aadhaar Number (12 digits, may be formatted as XXXX XXXX XXXX)
- PAN Number (10 characters: 5 letters, 4 digits, 1 letter)

Return JSON format:
{
  "member_type": "Policy Holder",
  "full_name": "",
  "residential_address": "",
  "pincode": "",
  "state": "",
  "city": "",
  "country": "",
  "phone_number": "",
  "email": "",
  "gender": "",
  "occupation": "",
  "company_firm_name": "",
  "years_of_experience": "",
  "salary_business_income": "",
  "aadhaar_number": "",
  "pan_number": ""
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in all_base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending member documents to GPT-4o Vision...")
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
            
            # Normalize gender to lowercase
            if extracted_fields.get("gender"):
                gv = str(extracted_fields["gender"]).strip().lower()
                if gv in ("male", "female"):
                    extracted_fields["gender"] = gv

            return {
                'success': True,
                'document_type': 'member_documents',
                'member_index': member_index,
                'extracted_text': '',  # No OCR text needed
                'member_details': extracted_fields,
                'pages_processed': total_pages,
                'files_processed': len(processing_results),
                'processing_results': processing_results,
                'member_documents': [{"type": dt, "file": ""} for dt in document_types]
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Member documents processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_preauth_doc(self, file_paths: list) -> Dict[str, Any]:
        """
        Process Pre-Authorization Document and extract specific fields using GPT-4o Vision:
        - Hospital Name
        - Treating Doctor
        - Application Type (Pre-Authorization)
        - Estimated Treatment Cost
        - Pre-Auth Reference Number
        - Tentative Admission Date
        - Patient Name
        - Diagnosis/Medical Condition
        """
        try:
            # Convert all files to base64 images
            logger.info(f"Converting {len(file_paths)} Pre-Authorization Document files to images for GPT-4o Vision...")
            
            all_base64_images = []
            total_pages = 0
            processing_results = []
            
            for i, file_path in enumerate(file_paths):
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                base64_images = self._convert_file_to_base64_images(file_path)
                
                if not base64_images:
                    logger.warning(f"Failed to convert {file_path} to images")
                    processing_results.append({
                        'file': file_path,
                        'success': False,
                        'error': "Failed to convert to images"
                    })
                    continue
                
                all_base64_images.extend(base64_images)
                total_pages += len(base64_images)
                
                processing_results.append({
                    'file': file_path,
                    'success': True,
                    'pages_processed': len(base64_images)
                })
            
            if not all_base64_images:
                return {
                    'success': False,
                    'error': "Failed to convert any documents to images",
                    'processing_results': processing_results
                }
            
            # Prepare content with all images
            content = [
                {
                    "type": "text",
                    "text": """Extract pre-authorization information from documents.

Extract these fields:
- Hospital Name
- Treating Doctor
- Application Type (should be "Pre-Authorization" or similar)
- Estimated Treatment Cost
- Pre-Auth Reference Number
- Tentative Admission Date
- Patient Name
- Diagnosis/Medical Condition

Return JSON format:
{
  "hospital_name": "",
  "treating_doctor": "",
  "application_type": "",
  "estimated_treatment_cost": "",
  "preauth_reference_number": "",
  "tentative_admission_date": "",
  "patient_name": "",
  "diagnosis_medical_condition": ""
}"""
                }
            ]
            
            # Add all pages as images
            for base64_image in all_base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not self.client:
                raise ValueError("No OpenAI client configured")
            
            logger.info("Sending Pre-Authorization Document to GPT-4o Vision...")
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
            
            schema = self.base_insurance_schema()
            schema["hospital_name"] = extracted_fields.get("hospital_name", "")
            schema["treating_doctor"] = extracted_fields.get("treating_doctor", "")
            schema["application_type"] = extracted_fields.get("application_type", "")
            schema["estimated_treatment_cost"] = extracted_fields.get("estimated_treatment_cost", "")
            schema["preauth_reference_number"] = extracted_fields.get("preauth_reference_number", "")
            schema["tentative_admission_date"] = extracted_fields.get("tentative_admission_date", "")
            schema["patient_name"] = extracted_fields.get("patient_name", "")
            schema["diagnosis_medical_condition"] = extracted_fields.get("diagnosis_medical_condition", "")

            return {
                'success': True,
                'document_type': 'preauth_doc',
                'extracted_text': '',  # No OCR text needed
                'fields': schema,
                'pages_processed': total_pages,
                'files_processed': len(file_paths),
                'processing_results': processing_results
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Pre-Authorization Document processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
