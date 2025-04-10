import os
import shutil
import json
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from scripts.text_extractor import extract_text
from scripts.resume_parser import parse_resume_with_bert

# Set absolute path for templates
TEMPLATE_DIR = r"C:\Users\Senior Mitnik\Desktop\try 5\template"
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# FastAPI Initialization
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path for uploads
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
logging.basicConfig(level=logging.INFO)

@app.get("/", response_class=HTMLResponse)
async def serve_upload_form(request: Request):
    """Serve the upload form HTML page"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")  # Endpoint for file uploads
async def upload_resumes(files: list[UploadFile] = File(...)):
    """Handle file uploads and resume parsing"""
    results = []
    
    for file in files:
        try:
            # Validate file extension
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file.filename}. Only PDF, DOCX, and TXT files are allowed."
                )

            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Process the file
            extracted_text = extract_text(file_path)
            parsed_resume = parse_resume_with_bert(extracted_text)
            
            if not isinstance(parsed_resume, dict):
                raise ValueError("Unexpected parser output format")

            # Calculate score if not already present
            if "Score" not in parsed_resume:
                parsed_resume["Score"] = 0  # Default score if not calculated

            results.append({
                "Filename": file.filename,
                "Name": parsed_resume.get("Name", "N/A"),
                "Email": parsed_resume.get("Email", "N/A"),
                "Contact": parsed_resume.get("Contact", "N/A"),
                "Industry": parsed_resume.get("Industry", "N/A"),
                "Score": parsed_resume["Score"],
                "Details": json.dumps(parsed_resume, indent=2)
            })

        except Exception as e:
            logging.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "Filename": file.filename,
                "Error": str(e)
            })
    
    # Sort results by score (highest first)
    results.sort(key=lambda x: x.get("Score", 0), reverse=True)
    
    return {"results": results}  # FastAPI automatically converts this to JSON

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)