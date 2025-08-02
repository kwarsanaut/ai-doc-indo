# API Documentation

## Base URL
```
Development: http://localhost:8000
Production: https://api.yourdomain.com
```

## Authentication

All API endpoints (except `/health` and `/docs`) require authentication using JWT tokens.

### Get Authentication Token

**Endpoint:** `POST /auth/token`

**Request Body:**
```json
{
  "email": "user@company.com",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Usage:**
Include the token in the Authorization header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Document Processing

### Process Document

**Endpoint:** `POST /process-document`

**Description:** Process a document with full AI extraction including entity extraction, classification, and summarization.

**Request:**
```bash
curl -X POST "http://localhost:8000/process-document" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "document_info": {
      "filename": "invoice_sample.pdf",
      "type": "pdf",
      "pages": 2,
      "processing_time": 3.45,
