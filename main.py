import os
import tempfile
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler as default_http_exception_handler
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

from services.onboarding import OCRProcessorService
from services.application import InsuranceProcessorService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security scheme for Swagger UI
security = HTTPBearer(auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Document OCR & Auto-Fill API",
    description="Extract text from hospital documents using Tesseract OCR and auto-fill forms using LLM",
    version="1.0.0"
)

# Add security scheme to OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "Token",
            "description": "Enter your token (RXCS_AI_TOKEN). Just enter the token value, 'Bearer' prefix is added automatically."
        }
    }
    # Add security to all protected endpoints (those with get_current_user dependency)
    # Also ensure file upload endpoints have proper multipart/form-data content type
    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if isinstance(operation, dict):
                # Check route dependencies
                for route in app.routes:
                    if hasattr(route, "path") and route.path == path and hasattr(route, "dependencies"):
                        for dep in route.dependencies:
                            dep_str = str(dep)
                            if "get_current_user" in dep_str or "verify_token" in dep_str:
                                operation["security"] = [{"BearerAuth": []}]
                                break
                
                # Force multipart/form-data for file upload endpoints
                if method.lower() == "post" and path in [
                    "/api/process_hospital_documents",
                    "/api/process_insurance_and_preauth",
                    "/api/process_member_documents"
                ]:
                    # Ensure requestBody uses multipart/form-data
                    if "requestBody" in operation:
                        request_body = operation["requestBody"]
                        if "content" in request_body:
                            # Remove any application/json content type
                            if "application/json" in request_body["content"]:
                                del request_body["content"]["application/json"]
                            # Ensure multipart/form-data exists
                            if "multipart/form-data" not in request_body["content"]:
                                request_body["content"]["multipart/form-data"] = {
                                    "schema": {
                                        "type": "object",
                                        "properties": {}
                                    }
                                }
                            # Update schema to ensure file fields are shown as file uploads
                            schema = request_body["content"]["multipart/form-data"]["schema"]
                            if "properties" not in schema:
                                schema["properties"] = {}
                            
                            # For hospital documents endpoint
                            if path == "/api/process_hospital_documents":
                                for field in ["registration_certificate", "cancelled_check", "rohini_id", 
                                            "hospital_rate_list", "hospital_gst", "signed_mou"]:
                                    schema["properties"][field] = {
                                        "type": "array",
                                        "items": {"type": "string", "format": "binary"},
                                        "description": f"Upload {field.replace('_', ' ').title()} files"
                                    }
                            
                            # For member documents endpoint
                            elif path == "/api/process_member_documents":
                                for field in ["aadhar_card", "pan_card", "salary_slip"]:
                                    schema["properties"][field] = {
                                        "type": "array",
                                        "items": {"type": "string", "format": "binary"},
                                        "description": f"Upload {field.replace('_', ' ').title()} files"
                                    }
                            
                            # For insurance endpoint
                            elif path == "/api/process_insurance_and_preauth":
                                for field in ["policy_doc", "files", "ecard", "pre_auth_form"]:
                                    schema["properties"][field] = {
                                        "type": "array",
                                        "items": {"type": "string", "format": "binary"},
                                        "description": f"Upload {field.replace('_', ' ').title()} files"
                                    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ocr_processor = OCRProcessorService()
insurance_processor = InsuranceProcessorService()

# Authentication setup - wrapper to make it work with both Security and manual checking
async def verify_token(request: Request = None, credentials: Optional[HTTPAuthorizationCredentials] = None):
    """Verify Authorization header matches RXCS_AI_TOKEN from .env"""
    expected_token = os.getenv('RXCS_AI_TOKEN', '').strip()
    
    # If no token set in .env, allow access (backward compatible)
    if not expected_token:
        logger.warning("RXCS_AI_TOKEN not set in .env - authentication disabled")
        return True
    
    token = None
    
    # First try to get token from HTTPBearer credentials (for Swagger UI)
    if credentials:
        token = credentials.credentials
    
    # Fallback: check request headers manually
    if not token and request:
        auth_header = (
            request.headers.get('Authorization') or 
            request.headers.get('authorization') or 
            request.headers.get('AUTHORIZATION') or
            ''
        )
        
        # Also check if client sent it in a custom header
        if not auth_header:
            for alt_header in ['X-API-Key', 'x-api-key', 'X-Auth-Token', 'x-auth-token', 'Token', 'token']:
                alt_value = request.headers.get(alt_header)
                if alt_value:
                    auth_header = alt_value
                    logger.info(f"Found token in alternative header: {alt_header}")
                    break
        
        if auth_header:
            token = auth_header.strip()
            # Remove "Bearer " prefix if present
            if token.lower().startswith('bearer '):
                token = token[7:].strip()
            elif token.lower().startswith('token '):
                token = token[6:].strip()
    
    if not token:
        logger.warning(f"Authorization header missing")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    expected_token = expected_token.strip()
    
    # Compare tokens (exact match, case-sensitive)
    if token != expected_token:
        logger.warning(f"Token mismatch!")
        logger.warning(f"Expected token: '{expected_token}' (length: {len(expected_token)})")
        logger.warning(f"Received token: '{token}' (length: {len(token)})")
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    logger.debug("Token verified successfully")
    return True

# Dependency function that works with FastAPI Security
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """Dependency that extracts credentials and verifies token"""
    # Pass both request and credentials to verify_token
    # verify_token will handle the case when credentials is None
    return await verify_token(request=request, credentials=credentials)

@app.get("/")
async def root():
    print("Root endpoint called")
    """Redirect to API documentation"""
    return {"message": "Hospital Document OCR & Auto-Fill API", "docs": "/docs"}

# --- Graceful handling: return null-schema instead of 422 when a file field is sent as text ---
def _schema_with_nulls(data: dict) -> dict:
    def _conv(v):
        if isinstance(v, dict):
            return {k: _conv(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_conv(x) for x in v]
        if isinstance(v, str):
            return None if v == "" else v
        return v
    return _conv(data)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    path = request.url.path
    # Log the validation error for debugging
    logger.warning(f"Validation error on {path}: {exc.errors()}")
    
    # Check if the error is due to file fields receiving strings instead of files
    errors = exc.errors()
    is_file_validation_error = any(
        error.get('type') == 'value_error' and 'UploadFile' in str(error.get('msg', ''))
        for error in errors
    )
    
    # If a file field was posted as plain text (e.g., "string"), treat it as missing and respond with empty schema
    if path == "/api/process_insurance_and_preauth":
        base = insurance_processor.base_insurance_schema()
        return JSONResponse(status_code=200, content={
            'success': True,
            'document_type': 'insurance_and_preauth',
            'fields': base,  # Return base schema with empty strings, not nulls
            'pages_processed': 0,
            'files_processed': 0,
            'number_of_members': 0,
            'member_names': []
        })
    if path == "/api/process_hospital_documents":
        base = ocr_processor.base_hospital_schema()
        # Return base schema directly (same format as endpoint returns)
        return JSONResponse(status_code=200, content=base)
    if path == "/api/process_member_documents":
        # Try to process files even when validation fails (e.g., when strings are sent instead of files)
        # This allows processing with any combination of documents
        try:
            form = await request.form()
            parsed_aadhar = []
            parsed_pan = []
            parsed_salary = []
            
            # Extract actual UploadFile objects, filtering out strings
            if 'aadhar_card' in form:
                aadhar_values = form.getlist('aadhar_card')
                for val in aadhar_values:
                    if isinstance(val, UploadFile) and val.filename:
                        parsed_aadhar.append(val)
            
            if 'pan_card' in form:
                pan_values = form.getlist('pan_card')
                for val in pan_values:
                    if isinstance(val, UploadFile) and val.filename:
                        parsed_pan.append(val)
            
            if 'salary_slip' in form:
                salary_values = form.getlist('salary_slip')
                for val in salary_values:
                    if isinstance(val, UploadFile) and val.filename:
                        parsed_salary.append(val)
            
            # If we have any files, process them
            if parsed_aadhar or parsed_pan or parsed_salary:
                logger.info(f"Processing member documents from validation error handler (Aadhar: {len(parsed_aadhar)}, PAN: {len(parsed_pan)}, Salary: {len(parsed_salary)})")
                # Process the files using the service
                temp_file_paths = []
                saved_aadhar = []
                saved_pan = []
                saved_salary = []
                
                try:
                    # Save Aadhar files
                    for file in parsed_aadhar:
                        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                        file_extension = os.path.splitext(file.filename.lower())[1]
                        if file_extension in allowed_extensions:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                                content = await file.read()
                                if content:
                                    temp_file.write(content)
                                    temp_path = temp_file.name
                                    temp_file_paths.append(temp_path)
                                    saved_aadhar.append(temp_path)
                    
                    # Save PAN files
                    for file in parsed_pan:
                        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                        file_extension = os.path.splitext(file.filename.lower())[1]
                        if file_extension in allowed_extensions:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                                content = await file.read()
                                if content:
                                    temp_file.write(content)
                                    temp_path = temp_file.name
                                    temp_file_paths.append(temp_path)
                                    saved_pan.append(temp_path)
                    
                    # Save Salary files
                    for file in parsed_salary:
                        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                        file_extension = os.path.splitext(file.filename.lower())[1]
                        if file_extension in allowed_extensions:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                                content = await file.read()
                                if content:
                                    temp_file.write(content)
                                    temp_path = temp_file.name
                                    temp_file_paths.append(temp_path)
                                    saved_salary.append(temp_path)
                    
                    # Process if we have valid files
                    if temp_file_paths:
                        result = await asyncio.to_thread(
                            insurance_processor.process_member_documents,
                            member_index=0,
                            aadhaar_files=saved_aadhar or None,
                            pan_files=saved_pan or None,
                            salary_files=saved_salary or None
                        )
                        
                        # Clean up temp files
                        for temp_path in temp_file_paths:
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass
                        
                        if result.get('success'):
                            return JSONResponse(status_code=200, content={
                                'success': True,
                                'member_details': result.get('member_details', {}),
                                'pages_processed': result.get('pages_processed', 0),
                                'files_processed': result.get('files_processed', 0),
                                'processing_results': result.get('processing_results', [])
                            })
                except Exception as e:
                    logger.error(f"Error processing files in exception handler: {e}", exc_info=True)
                    # Clean up temp files on error
                    for temp_path in temp_file_paths:
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
        
        except Exception as e:
            logger.warning(f"Could not process files from validation error: {e}")
        
        # If processing failed or no files found, return empty response
        return JSONResponse(status_code=200, content={
            'success': True,
            'message': 'No member documents processed',
            'member_index': 0,
            'member_details': {
                'member_type': '',
                'full_name': '',
                'father_name': '',
                'residential_address': '',
                'pincode': '',
                'state': '',
                'city': '',
                'country': '',
                'phone_number': '',
                'email': '',
                'gender': '',
                'occupation': '',
                'company_firm_name': '',
                'years_of_experience': '',
                'salary_business_income': '',
                'aadhaar_number': '',
                'pan_number': ''
            },
            'extracted_text': '',
            'pages_processed': 0,
            'files_processed': 0,
            'processing_results': []
        })
    # For other routes, keep default 422 shape but ensure serializable content
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    # Log HTTP exceptions for debugging (except 401 which is expected for auth)
    if exc.status_code != 401:
        logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    
    # Gracefully handle 422 on member documents
    if request.url.path == "/api/process_member_documents" and exc.status_code == 422:
        return JSONResponse(status_code=200, content={
            'success': True,
            'message': 'No member documents processed',
            'member_index': 0,
            'member_details': {
                'member_type': '',
                'full_name': '',
                'father_name': '',
                'residential_address': '',
                'pincode': '',
                'state': '',
                'city': '',
                'country': '',
                'phone_number': '',
                'email': '',
                'gender': '',
                'occupation': '',
                'company_firm_name': '',
                'years_of_experience': '',
                'salary_business_income': '',
                'aadhaar_number': '',
                'pan_number': ''
            },
            'extracted_text': '',
            'pages_processed': 0,
            'files_processed': 0,
            'processing_results': []
        })
    return await default_http_exception_handler(request, exc)

# ===== Hospital Application Endpoint (hosp_appli.py) =====

from enum import Enum
from pydantic import BaseModel
from typing import List

class HospitalDocumentType(str, Enum):
    registration_certificate = "registration_certificate"
    cancelled_check = "cancelled_check"
    rohini_id = "rohini_id"
    hospital_rate_list = "hospital_rate_list"
    hospital_gst = "hospital_gst"
    signed_mou = "signed_mou"

class InsuranceDocumentType(str, Enum):
    insurance_copy = "insurance_copy"
    insurance_members = "insurance_members"
    claim_filling_doc = "claim_filling_doc"

class DocumentWithType(BaseModel):
    file: UploadFile
    document_type: HospitalDocumentType

class InsuranceDocumentWithType(BaseModel):
    file: UploadFile
    document_type: InsuranceDocumentType

@app.post("/api/process_hospital_documents", dependencies=[Depends(get_current_user)])
async def process_hospital_documents(
    registration_certificate: Optional[List[UploadFile]] = File(None, description="Upload Registration Certificate files"),
    cancelled_check: Optional[List[UploadFile]] = File(None, description="Upload Cancelled Check files"),
    rohini_id: Optional[List[UploadFile]] = File(None, description="Upload Rohini ID files"),
    hospital_rate_list: Optional[List[UploadFile]] = File(None, description="Upload Hospital Rate List files"),
    hospital_gst: Optional[List[UploadFile]] = File(None, description="Upload Hospital GST files"),
    signed_mou: Optional[List[UploadFile]] = File(None, description="Upload Signed MOU files")
):
    """
    Process hospital documents and extract specific fields based on document type
    
    **Parameters:**
    - `files`: List of documents to process (PDF, PNG, JPG, JPEG)
    - `document_type`: Type of document
    
    **Returns:**
    - Extracted text from OCR
    - Extracted structured fields
    """
    try:
        # Log received files for debugging
        logger.info(f"Received files - registration_certificate: {len(registration_certificate or [])}, "
                   f"cancelled_check: {len(cancelled_check or [])}, rohini_id: {len(rohini_id or [])}, "
                   f"hospital_rate_list: {len(hospital_rate_list or [])}, hospital_gst: {len(hospital_gst or [])}, "
                   f"signed_mou: {len(signed_mou or [])}")
        
        # Build selected document types from provided file fields
        selected_types: dict[str, List[UploadFile]] = {
            'registration_certificate': registration_certificate or [],
            'cancelled_check': cancelled_check or [],
            'rohini_id': rohini_id or [],
            'hospital_rate_list': hospital_rate_list or [],
            'hospital_gst': hospital_gst or [],
            'signed_mou': signed_mou or [],
        }
        # Keep only those with at least one file
        selected_types = {k: v for k, v in selected_types.items() if v}

        if not selected_types:
            logger.warning("No files provided in request")
            raise HTTPException(status_code=400, detail="Please select at least one document type and upload at least one file")

        logger.info(f"Processing document types: {list(selected_types.keys())}")

        temp_file_paths: list[str] = []
        results: list[dict] = []
        processor_map = {
            'registration_certificate': ocr_processor.process_registration_certificate,
            'cancelled_check': ocr_processor.process_cancelled_check,
            'rohini_id': ocr_processor.process_rohini_id,
            'hospital_rate_list': ocr_processor.process_hospital_rate_list,
            'hospital_gst': ocr_processor.process_hospital_gst,
            'signed_mou': ocr_processor.process_signed_mou,
        }
        combined = ocr_processor.base_hospital_schema()
        processing_jobs: list[tuple[str, str]] = []

        try:
            for doc_type_name, file_list in selected_types.items():
                file = file_list[0]

                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    results.append({
                        "document_type": doc_type_name,
                        "success": False,
                        "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
                    })
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                temp_file_paths.append(temp_file_path)
                processing_jobs.append((doc_type_name, temp_file_path))

            async def run_processor(doc_type_name: str, temp_file_path: str):
                processor = processor_map.get(doc_type_name)
                if not processor:
                    raise HTTPException(status_code=400, detail=f"Invalid document type: {doc_type_name}")
                result = await asyncio.to_thread(processor, temp_file_path)
                return doc_type_name, result

            tasks = [run_processor(doc_type_name, temp_path) for doc_type_name, temp_path in processing_jobs]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for outcome in task_results:
                if isinstance(outcome, Exception):
                    logger.error(f"Processing error: {outcome}")
                    continue
                doc_type_name, result = outcome
                if not result.get('success'):
                    logger.warning(f"Processing failed for {doc_type_name}: {result.get('error')}")
                    results.append({
                        "document_type": doc_type_name,
                        "success": False,
                        "error": result.get('error', 'Unknown error'),
                    })
                else:
                    results.append({
                        "document_type": doc_type_name,
                        "success": True,
                        "extracted_text": result.get('extracted_text'),
                        "extracted_fields": result.get('fields'),
                        "pages_processed": result.get('pages_processed'),
                    })
                    fields = result.get('fields', {}) or {}
                    for k, v in fields.items():
                        if v in (None, "", []):
                            continue
                        if k == 'hospital_rates':
                            if isinstance(v, list) and v:
                                combined['hospital_rates'] = v
                        elif k == 'no_of_beds':
                            try:
                                combined['no_of_beds'] = int(v) if isinstance(v, str) and v.isdigit() else v
                            except Exception:
                                combined['no_of_beds'] = v
                        else:
                            combined[k] = v

            return combined
        finally:
            for temp_file_path in temp_file_paths:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ===== Old Insurance Endpoint Removed - Use Sequential Endpoints Instead =====

# ===== Modified Insurance Processing Endpoints =====

@app.post("/api/process_insurance_and_preauth", dependencies=[Depends(get_current_user)])
async def process_insurance_and_preauth(
    request: Request,
    policy_doc: List[UploadFile] = File(default=[], description="policy_doc", alias="policy_doc"),
    files: List[UploadFile] = File(default=[], description="files (alias of policy_doc)", alias="files"),
    ecard: List[UploadFile] = File(default=[], description="ecard", alias="ecard"),
    pre_auth_form: List[UploadFile] = File(default=[], description="pre_auth_form", alias="pre_auth_form")
):
    """
    Process Insurance Copy + Pre-Authorization Documents in ONE endpoint.
    
    Returns:
    - Policy details (insurance company, policy number, etc.)
    - Number of members
    - Member names list
    - Pre-auth details (hospital, doctor, cost, etc.)
    """
    try:
        if not policy_doc and not files and not ecard and not pre_auth_form:
            # Collect debug info when nothing was bound to expected fields
            try:
                form = await request.form()
                form_keys = list(form.keys())
            except Exception as e:
                form_keys = [f"<unreadable:{e}>"]
            debug_info = {
                "reason": "no_files_in_expected_fields",
                "expected_fields": ["policy_doc", "files", "ecard", "pre_auth_form"],
                "received_form_keys": form_keys,
                "content_type": request.headers.get("content-type", "")
            }
            logger.warning(f"Bad request: {debug_info}")
            raise HTTPException(status_code=400, detail=debug_info)
        
        logger.info(f"Processing {len(policy_doc)+len(files)} Insurance Copy files, {len(ecard)} E-Card files and {len(pre_auth_form)} Pre-Auth files...")
        
        insurance_temp_paths: List[str] = []
        ecard_temp_paths: List[str] = []
        preauth_temp_paths: List[str] = []
        invalid_files: List[dict] = []
        
        try:
            # Save insurance copy files (policy_doc + files alias)
            for file in list(policy_doc or []) + list(files or []):
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    invalid_files.append({"field": "policy_doc|files", "filename": file.filename, "reason": f"invalid_extension:{file_extension}"})
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    if not content:
                        invalid_files.append({"field": "policy_doc|files", "filename": file.filename, "reason": "empty_file"})
                        continue
                    temp_file.write(content)
                    temp_path = temp_file.name
                insurance_temp_paths.append(temp_path)
            
            # Save e-card files
            for file in ecard:
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    invalid_files.append({"field": "ecard", "filename": file.filename, "reason": f"invalid_extension:{file_extension}"})
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    if not content:
                        invalid_files.append({"field": "ecard", "filename": file.filename, "reason": "empty_file"})
                        continue
                    temp_file.write(content)
                    temp_path = temp_file.name
                ecard_temp_paths.append(temp_path)
            
            # Save preauth files
            for file in pre_auth_form:
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    invalid_files.append({"field": "pre_auth_form", "filename": file.filename, "reason": f"invalid_extension:{file_extension}"})
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    if not content:
                        invalid_files.append({"field": "pre_auth_form", "filename": file.filename, "reason": "empty_file"})
                        continue
                    temp_file.write(content)
                    temp_path = temp_file.name
                preauth_temp_paths.append(temp_path)
            
            if not insurance_temp_paths and not ecard_temp_paths and not preauth_temp_paths:
                try:
                    form = await request.form()
                    form_keys = list(form.keys())
                except Exception as e:
                    form_keys = [f"<unreadable:{e}>"]
                debug_info = {
                    "reason": "no_valid_files_after_filtering",
                    "invalid_files": invalid_files,
                    "received_form_keys": form_keys,
                    "content_type": request.headers.get("content-type", ""),
                    "fields_received_counts": {
                        "policy_doc": len(policy_doc or []),
                        "files": len(files or []),
                        "ecard": len(ecard or []),
                        "pre_auth_form": len(pre_auth_form or [])
                    }
                }
                logger.warning(f"Bad request: {debug_info}")
                raise HTTPException(status_code=400, detail=debug_info)
            
            insurance_result = None
            preauth_result = None

            # Process insurance copy (insurance + ecard combined)
            insurance_task = None
            preauth_task = None

            combined_insurance_paths = insurance_temp_paths + ecard_temp_paths
            if combined_insurance_paths:
                insurance_task = asyncio.create_task(
                    asyncio.to_thread(insurance_processor.process_insurance_copy_only, combined_insurance_paths)
                )

            if preauth_temp_paths:
                preauth_task = asyncio.create_task(
                    asyncio.to_thread(insurance_processor.process_preauth_doc, preauth_temp_paths)
                )

            if insurance_task:
                insurance_result = await insurance_task
                if not insurance_result.get('success'):
                    raise HTTPException(status_code=400, detail={
                        "reason": "insurance_processing_failed",
                        "error": insurance_result.get('error', 'Processing failed')
                    })

            if preauth_task:
                preauth_result = await preauth_task
                if not preauth_result.get('success'):
                    raise HTTPException(status_code=400, detail={
                        "reason": "preauth_processing_failed",
                        "error": preauth_result.get('error', 'Processing failed')
                    })
            
            # Combine both results into unified format
            combined_fields = {}
            extracted_texts = []
            total_pages = 0
            total_files = 0
            all_processing_results = []
            number_of_members = 0
            member_names = []
            member_genders = []
            
            if insurance_result:
                combined_fields.update(insurance_result['fields'])
                extracted_texts.append(insurance_result['extracted_text'])
                total_pages += insurance_result['pages_processed']
                total_files += insurance_result['files_processed']
                all_processing_results.extend(insurance_result.get('processing_results', []))
                number_of_members = insurance_result.get('number_of_members', 0)
                member_names = insurance_result.get('member_names', [])
                member_genders = insurance_result.get('member_genders', [])
            
            if preauth_result:
                combined_fields.update({
                    'hospital_name': preauth_result['fields'].get('hospital_name', ''),
                    'treating_doctor': preauth_result['fields'].get('treating_doctor', ''),
                    'application_type': preauth_result['fields'].get('application_type', ''),
                    'estimated_treatment_cost': preauth_result['fields'].get('estimated_treatment_cost', ''),
                    'preauth_reference_number': preauth_result['fields'].get('preauth_reference_number', ''),
                    'tentative_admission_date': preauth_result['fields'].get('tentative_admission_date', ''),
                    'patient_name': preauth_result['fields'].get('patient_name', ''),
                    'diagnosis_medical_condition': preauth_result['fields'].get('diagnosis_medical_condition', '')
                })
                extracted_texts.append(preauth_result['extracted_text'])
                total_pages += preauth_result['pages_processed']
                total_files += preauth_result['files_processed']
                all_processing_results.extend(preauth_result.get('processing_results', []))
            
            return {
                'success': True,
                'document_type': 'insurance_and_preauth',
                'fields': combined_fields,
                'pages_processed': total_pages,
                'files_processed': total_files,
                'number_of_members': number_of_members,
                'member_names': member_names,
                'member_genders': member_genders
            }
            
        finally:
            # Clean up temporary files
            for temp_file_path in insurance_temp_paths + ecard_temp_paths + preauth_temp_paths:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Combined processing error: {str(e)}")


@app.post("/api/process_member_documents", dependencies=[Depends(get_current_user)])
async def process_member_documents(
    request: Request,
    # Primary expected fields (only these three are supported now)
    aadhar_card: Optional[List[UploadFile]] = File(default=None, description="Upload Aadhar card files", alias="aadhar_card"),
    pan_card: Optional[List[UploadFile]] = File(default=None, description="Upload PAN card files", alias="pan_card"),
    salary_slip: Optional[List[UploadFile]] = File(default=None, description="Upload salary slip files", alias="salary_slip"),
):
    """
    Process individual member documents: Aadhar card, PAN card, Salary slip.
    
    This endpoint supports concurrent requests. Each request is processed independently.
    You can provide any combination of documents (1, 2, or all 3) - the endpoint will process whatever is provided.
    
    Returns:
    - Member details extracted from documents
    """
    # Generate unique request ID for tracking concurrent requests
    request_id = str(uuid.uuid4())[:8]
    try:
        # Manually parse form to handle cases where fields are sent as strings (from Swagger UI or other clients)
        # This allows processing even when only some documents are provided
        try:
            form = await request.form()
            # Filter out string values and only keep actual UploadFile objects
            parsed_aadhar = []
            parsed_pan = []
            parsed_salary = []
            
            # Handle aadhar_card
            if 'aadhar_card' in form:
                aadhar_values = form.getlist('aadhar_card')
                for val in aadhar_values:
                    if isinstance(val, UploadFile) and val.filename:
                        parsed_aadhar.append(val)
            
            # Handle pan_card
            if 'pan_card' in form:
                pan_values = form.getlist('pan_card')
                for val in pan_values:
                    if isinstance(val, UploadFile) and val.filename:
                        parsed_pan.append(val)
            
            # Handle salary_slip
            if 'salary_slip' in form:
                salary_values = form.getlist('salary_slip')
                for val in salary_values:
                    if isinstance(val, UploadFile) and val.filename:
                        parsed_salary.append(val)
            
            # Use parsed files if FastAPI didn't parse them correctly, otherwise use the function parameters
            if not aadhar_card and parsed_aadhar:
                aadhar_card = parsed_aadhar
            if not pan_card and parsed_pan:
                pan_card = parsed_pan
            if not salary_slip and parsed_salary:
                salary_slip = parsed_salary
                
        except Exception as e:
            logger.warning(f"[Request {request_id}] Could not parse form manually: {e}, using function parameters")
        
        # Reject empty submissions with detailed debug to fix client-side mismatches
        if not any([aadhar_card, pan_card, salary_slip]):
            try:
                form = await request.form()
                form_keys = list(form.keys())
            except Exception as e:
                form_keys = [f"<unreadable:{e}>"]
            raise HTTPException(
                status_code=400,
                detail={
                    'reason': 'no_files_in_expected_fields',
                    'expected_fields': ['aadhar_card', 'pan_card', 'salary_slip'],
                    'received_form_keys': form_keys,
                    'content_type': request.headers.get('content-type', '')
                }
            )
        
        logger.info(f"[Request {request_id}] Processing member documents...")
        debug_flag = False
        try:
            qp = dict(request.query_params)
            debug_flag = qp.get('debug') in ('1', 'true', 'yes') or request.headers.get('x-debug') in ('1', 'true', 'yes')
        except Exception:
            debug_flag = False
        
        temp_file_paths: List[str] = []
        
        try:
            # Save files per category to preserve mapping
            saved_aadhar = []
            saved_pan = []
            saved_salary = []
            invalid_files = []

            # For debug: record received counts per field
            files_received_counts = {
                'aadhar_card': len(aadhar_card or []),
                'pan_card': len(pan_card or []),
                'salary_slip': len(salary_slip or []),
            }
            
            # Use only the primary Aadhaar field
            merged_aadhar_inputs: List[UploadFile] = list(aadhar_card or [])

            for file in (merged_aadhar_inputs or []):
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    invalid_files.append({'field': 'aadhar_like', 'filename': file.filename, 'reason': f'invalid_extension:{file_extension}'})
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    if not content:
                        logger.warning(f"Empty file content received for {file.filename}")
                        invalid_files.append({'field': 'aadhar_like', 'filename': file.filename, 'reason': 'empty_file'})
                        continue
                    temp_file.write(content)
                    temp_path = temp_file.name
                temp_file_paths.append(temp_path)
                saved_aadhar.append(temp_path)

            for file in (pan_card or []):
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    invalid_files.append({'field': 'pan_card', 'filename': file.filename, 'reason': f'invalid_extension:{file_extension}'})
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    if not content:
                        logger.warning(f"Empty file content received for {file.filename}")
                        invalid_files.append({'field': 'pan_card', 'filename': file.filename, 'reason': 'empty_file'})
                        continue
                    temp_file.write(content)
                    temp_path = temp_file.name
                temp_file_paths.append(temp_path)
                saved_pan.append(temp_path)

            for file in (salary_slip or []):
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
                file_extension = os.path.splitext(file.filename.lower())[1]
                if file_extension not in allowed_extensions:
                    logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                    invalid_files.append({'field': 'salary_slip', 'filename': file.filename, 'reason': f'invalid_extension:{file_extension}'})
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    if not content:
                        logger.warning(f"Empty file content received for {file.filename}")
                        invalid_files.append({'field': 'salary_slip', 'filename': file.filename, 'reason': 'empty_file'})
                        continue
                    temp_file.write(content)
                    temp_path = temp_file.name
                temp_file_paths.append(temp_path)
                saved_salary.append(temp_path)

            # Generic 'files' field removed; only three explicit fields are supported
            
            if not temp_file_paths:
                raise HTTPException(status_code=400, detail="No valid files uploaded")
            
            # Process member documents via service and return mapped data
            # Using asyncio.to_thread ensures blocking operations don't block other concurrent requests
            logger.info(f"[Request {request_id}] Starting document processing (Aadhar: {len(saved_aadhar)}, PAN: {len(saved_pan)}, Salary: {len(saved_salary)})")
            result = await asyncio.to_thread(
                insurance_processor.process_member_documents,
                member_index=0,
                aadhaar_files=saved_aadhar or None,
                pan_files=saved_pan or None,
                salary_files=saved_salary or None
            )
            logger.info(f"[Request {request_id}] Document processing completed")
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result.get('error', 'Processing failed'))
            
            if debug_flag:
                # Provide rich debug info without leaking full text
                extracted_text = result.get('extracted_text', '') or ''
                preview = extracted_text[:1200]
                return {
                    'success': True,
                    'member_details': result['member_details'],
                    'debug': {
                        'files_received_counts': files_received_counts,
                        'invalid_files': invalid_files,
                        'pages_processed': result.get('pages_processed', 0),
                        'files_processed': result.get('files_processed', 0),
                        'processing_results': result.get('processing_results', []),
                        'member_documents_types': [d.get('type') for d in result.get('member_documents', [])],
                        'extracted_text_preview': preview
                    }
                }
            else:
                return {
                    'success': True,
                    'member_details': result['member_details']
                }
            
        finally:
            # Clean up temporary files
            for temp_file_path in temp_file_paths:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {e}")
        
    except HTTPException as e:
        logger.warning(f"[Request {request_id}] HTTPException: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"[Request {request_id}] Member documents processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Member documents processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Hospital Document OCR & Auto-Fill service is running",
        "version": "1.0.0",
        "services": {
            "ocr_space": "OCR Space API integration",
            "llm": "OpenAI GPT for field extraction"
        }
    }

@app.get("/upload-form", response_class=HTMLResponse)
async def upload_form():
    """Simple HTML form for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hospital Document OCR & Auto-Fill</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="file"], select, input[type="text"] { padding: 10px; border: 1px solid #ccc; border-radius: 4px; width: 300px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Hospital Document OCR & Auto-Fill</h1>
        <p>Upload hospital documents to extract information and auto-fill forms</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Upload Document:</label>
                <input type="file" id="file" name="file" accept=".pdf,.png,.jpg,.jpeg" required>
            </div>
            
            <div class="form-group">
                <label for="document_type">Document Type:</label>
                <select id="document_type" name="document_type">
                    <option value="registration certificate">Registration Certificate</option>
                    <option value="rohini id">Rohini ID</option>
                    <option value="aadhaar">Aadhaar Card</option>
                    <option value="cancelled check">Cancelled Check</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="target_fields">Target Fields (comma-separated):</label>
                <input type="text" id="target_fields" name="target_fields" 
                       value="hospital_name,hospital_email,hospital_address,hospital_contact_number" 
                       placeholder="hospital_name,hospital_email,hospital_address">
            </div>
            
            <button type="submit">Extract & Auto-Fill</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                formData.append('document_type', document.getElementById('document_type').value);
                formData.append('target_fields', document.getElementById('target_fields').value);
                
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Processing document... Please wait.';
                
                try {
                    const response = await fetch('/api/extract-and-fill', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = '<h3>Extraction Complete!</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } else {
                        resultDiv.innerHTML = '<h3>Error:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<h3>Error:</h3><p>' + error.message + '</p>';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)