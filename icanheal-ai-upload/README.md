# Hospital Document OCR & Auto-Fill API

A streamlined FastAPI application that extracts text from hospital documents using Tesseract OCR and auto-fills forms using LLM.

## Features

- **OCR Text Extraction**: Uses Tesseract OCR for accurate text extraction
- **LLM Field Mapping**: OpenAI GPT integration to extract structured fields
- **Auto-Fill Forms**: Automatically maps extracted data to form fields
- **Multiple Document Types**: Support for Registration Certificate, Rohini ID, Aadhaar, Cancelled Check
- **Simple API**: Clean REST API for easy integration



## Setup

1. **Install Tesseract OCR:**
   - Download and install Tesseract from: https://github.com/tesseract-ocr/tesseract
   - For Windows: Add Tesseract to PATH or update the path in `services/ocr_space_service.py`
   - For Linux: `sudo apt-get install tesseract-ocr`
   - For Mac: `brew install tesseract`

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```bash
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Or for Azure OpenAI:
     ```bash
     AZURE_OPENAI_API_KEY=your_azure_api_key_here
     AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
     AZURE_OPENAI_MODEL=gpt-4o
     ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## API Endpoints

### POST /api/extract-and-fill
Extract text from a single document and auto-fill specified fields.

**Parameters:**
- `file`: Document file (PDF, PNG, JPG, JPEG)
- `document_type`: Type of document
- `target_fields`: Comma-separated fields to extract

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "document_type": "registration certificate",
  "ocr_result": {
    "text_extracted": "Hospital Name: ABC Hospital...",
    "confidence": 85.5,
    "pages_processed": 1
  },
  "extracted_data": {
    "hospital_name": "ABC Hospital",
    "number_of_beds": "50",
    "hospital_email": "info@abchospital.com"
  },
  "mapped_fields": {
    "hospital_name": "ABC Hospital",
    "hospital_email": "info@abchospital.com"
  }
}
```

### POST /api/process-multiple-documents
Process multiple documents and combine all extracted information.

**Parameters:**
- `registration_certificate`: Registration Certificate file
- `rohini_id_doc`: Rohini ID document
- `aadhaar_card`: Aadhaar Card
- `cancelled_check`: Cancelled Check

**Response:**
```json
{
  "success": true,
  "documents_processed": 4,
  "combined_hospital_data": {
    "hospital_name": "ABC Hospital",
    "number_of_beds": "50",
    "rohini_id_num": "123456789",
    "owner_name": "Dr. John Doe",
    "payee_name": "ABC Hospital",
    "bank_name": "State Bank of India"
  },
  "ready_for_form_fill": true
}
```

### GET /health
Health check endpoint.

### GET /upload-form
Simple HTML form for testing document uploads.

## Integration with Hospital CRM

This OCR engine can be easily integrated with your Hospital CRM system:

1. **Upload documents** from the CRM frontend
2. **Call the OCR API** to extract information
3. **Auto-fill form fields** with extracted data
4. **Save to database** using existing CRM APIs

## Usage Example

```python
import requests

# Upload document and extract information
files = {'file': open('hospital_registration.pdf', 'rb')}
data = {
    'document_type': 'registration certificate',
    'target_fields': 'hospital_name,hospital_email,hospital_address'
}

response = requests.post('http://localhost:5000/api/extract-and-fill', files=files, data=data)
result = response.json()

# Use extracted data to auto-fill form
extracted_fields = result['mapped_fields']
print(f"Hospital Name: {extracted_fields.get('hospital_name')}")
```

## Error Handling

The API includes comprehensive error handling:
- Invalid file types
- OCR extraction failures
- LLM processing errors
- Missing API keys

All errors are returned with descriptive messages and appropriate HTTP status codes.

## Performance

- **OCR Space API**: Fast and accurate text extraction
- **OpenAI GPT**: Efficient field mapping with minimal tokens
- **Async Processing**: Non-blocking file uploads and processing
- **Temporary Files**: Automatic cleanup of uploaded files

## License

This project is open source and available under the MIT License."# icanheal-ai" 
