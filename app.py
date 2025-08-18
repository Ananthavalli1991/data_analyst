# app.py
# Refactored Flask application for concurrent, time-limited data processing with IMAGE SUPPORT.

import os
import io
import json
import re
import concurrent.futures
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
# --- All your existing file handlers ---
import csv
import xml.etree.ElementTree as ET
import google.generativeai as genai
from bs4 import BeautifulSoup
import openpyxl
import PyPDF2
import pandas as pd
import yaml
from google.api_core.exceptions import ResourceExhausted
from PIL import Image # NEW: Import the Image module from Pillow
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")

# --- API Key Configuration ---
genai.configure(api_key=GEMINI_API_KEY)

# --- CORRECTED: Model Fallback Function ---
def get_gemini_model():
    """Attempts to get the Pro model, falls back to Flash if it fails by making a test API call."""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        # This is the crucial line to ensure the fallback works correctly.
        # It makes a minimal API call to check for ResourceExhausted exception.
        model.generate_content("test")
        print("Using gemini-2.5-pro model.")
        return model
    except ResourceExhausted as e:
        print(f"Rate limit hit for gemini-2.5-pro: {e}. Falling back to gemini-2.5-flash.")
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        # Catch any other unexpected errors and fall back
        print(f"An unexpected error occurred: {e}. Falling back to gemini-2.5-flash.")
        return genai.GenerativeModel('gemini-2.5-flash')

# --- File Handlers (with new image handler) ---

def handle_text_file(file_content): return file_content.decode('utf-8')
def handle_html_file(file_content): return BeautifulSoup(file_content, 'html.parser').get_text(separator=' ', strip=True)
def handle_markdown_file(file_content): return file_content.decode('utf-8')
def handle_csv_file(file_content):
    csv_reader = csv.reader(io.StringIO(file_content.decode('utf-8')))
    try:
        header = next(csv_reader)
        rows = [",".join(row) for row in csv_reader]
        return ",".join(header) + "\n" + "\n".join(rows)
    except StopIteration:
        return "CSV file is empty."
def handle_json_file(file_content): return json.dumps(json.loads(file_content), indent=2)
def handle_pdf_file(file_content):
    pdf_stream = io.BytesIO(file_content)
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_stream)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        return f"Error processing PDF: {str(e)}"
    return text
def handle_excel_file(file_content):
    stream = io.BytesIO(file_content)
    data = ""
    try:
        workbook = openpyxl.load_workbook(stream, read_only=True)
        for sheet in workbook.sheetnames:
            sh = workbook[sheet]
            data += f"\n--- Sheet: {sheet} ---\n"
            for row in sh.iter_rows():
                row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                data += ",".join(row_data) + "\n"
    except Exception as e:
        return f"Error processing Excel: {str(e)}"
    return data
def handle_sql_file(file_content): return file_content.decode('utf-8')
def handle_xml_file(file_content):
    try:
        root = ET.fromstring(file_content)
        return ET.tostring(root, encoding='unicode', method='xml')
    except ET.ParseError as e:
        return f"Error parsing XML: {str(e)}"
def handle_parquet_file(file_content):
    try:
        df = pd.read_parquet(io.BytesIO(file_content))
        return df.to_json(orient="records", indent=2)
    except Exception as e:
        return f"Error processing Parquet: {str(e)}"
def handle_yaml_file(file_content):
    try:
        data = yaml.safe_load(file_content)
        return yaml.dump(data, sort_keys=False)
    except yaml.YAMLError as e:
        return f"Error processing YAML: {str(e)}"

def handle_image_file(file_content):
    """
    Opens the image content using Pillow and returns the Image object.
    """
    try:
        image = Image.open(io.BytesIO(file_content))
        return image
    except Exception as e:
        # If Pillow can't open it, return the error as a string
        return f"Error processing image with Pillow: {str(e)}"

# MODIFIED: Added image file extensions
FILE_HANDLERS = {
    # Text-based files
    '.txt': handle_text_file, '.html': handle_html_file, '.md': handle_markdown_file,
    '.csv': handle_csv_file, '.json': handle_json_file, '.pdf': handle_pdf_file,
    '.xlsx': handle_excel_file, '.xls': handle_excel_file, '.sql': handle_sql_file,
    '.xml': handle_xml_file, '.parquet': handle_parquet_file, '.yaml': handle_yaml_file,
    # NEW: Image files
    '.png': handle_image_file, '.jpg': handle_image_file, '.jpeg': handle_image_file, '.webp': handle_image_file
}

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'] # NEW: Helper list to identify image types

# --- Core Logic Functions (Refactored for Multimodality) ---

# MODIFIED: This function now accepts the model instance as an argument.
def analyze_with_gemini(questions, text_context, image_parts, gemini_model):
    if not gemini_model:
        return json.dumps([{"error": "Gemini Model not initialized"}])
    
    # Combine all text into a single prompt string
    full_text_prompt = f"""You are a precise data analyst. Your task is to analyze the provided data (which includes text and images) and answer the questions with EXACT JSON array format.

CRITICAL INSTRUCTIONS:
1. Return ONLY a valid JSON array - nothing else.
2. Each answer should be an element in the array.
3. For numbers or decimals, use numeric types (e.g., 42, 0.485782), not strings.
4. For strings, use double-quoted strings (e.g., "Titanic").
5. Do NOT include any explanations, comments, or markdown formatting like ```json or ```.

--- Questions to Answer ---
{questions}

--- Data Context ---
{text_context}

Return only the JSON array:"""

    # Build the final prompt list: [text, image1, image2, ...]
    prompt_parts = [full_text_prompt] + image_parts
    
    try:
        response = gemini_model.generate_content(
            prompt_parts, # MODIFIED: Pass the list of parts
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        return json.dumps([f"Gemini API Error: {str(e)}"])

def robust_json_parser(text_response):
    match = re.search(r'```json\s*(\[.*\])\s*```', text_response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text_response.strip()
    try:
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, list):
            return parsed_json
        else:
            return [parsed_json]
    except json.JSONDecodeError:
        return [f"Failed to decode JSON from model response: {text_response}"]

# MODIFIED: Returns a tuple indicating content type (text/image)
def process_single_file(file_storage):
    filename = file_storage.filename
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension in FILE_HANDLERS:
        try:
            # IMPORTANT: Re-position the stream for each worker to read the full content
            file_storage.seek(0)
            file_content = file_storage.read()
            handler = FILE_HANDLERS[file_extension]
            processed_data = handler(file_content)
            
            # Distinguish between image and text data
            if file_extension in IMAGE_EXTENSIONS:
                return ('image', processed_data, filename)
            else:
                text_data = f"\n--- Content from {filename} ---\n{processed_data}\n"
                return ('text', text_data, filename)

        except Exception as e:
            error_text = f"\n--- Error processing {filename}: {str(e)} ---\n"
            return ('text', error_text, filename) # Return errors as text context
            
    return ('text', '', filename) # Return empty for unsupported files

# MODIFIED: Accepts model instance as argument
def process_request_worker(request_files, model_instance):
    # This ensures the file is only read once and can be passed to the thread pool
    questions_file_content = request_files['questions.txt'].read().decode('utf-8')
    questions = questions_file_content
    
    text_context_parts = []
    image_parts = []
    
    # Exclude questions.txt from the list of files to process
    data_files = [f for k, f in request_files.items() if k != 'questions.txt']
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, f): f for f in data_files}
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                content_type, data, filename = future.result()
                if content_type == 'image':
                    # Append Pillow Image object to image_parts list
                    if isinstance(data, Image.Image):
                        image_parts.append(data)
                    else: # Handle case where image processing failed
                        text_context_parts.append(f"\n--- Content from {filename} ---\n{data}\n")
                else:
                    # Append text string to text_context_parts list
                    text_context_parts.append(data)
            except Exception as e:
                filename = future_to_file[future].filename
                print(f"Error processing file {filename} in thread: {e}")
                text_context_parts.append(f"\n--- Exception processing {filename}: {e} ---\n")

    # Join all text parts into a single string
    final_text_context = "".join(text_context_parts)
    
    # Pass separated text and image data to Gemini
    analysis_result_text = analyze_with_gemini(questions, final_text_context, image_parts, model_instance)
    
    final_json_array = robust_json_parser(analysis_result_text)
    
    return final_json_array

# --- API Endpoint (Controller - Unchanged) ---
app = Flask(__name__)
@app.route('/api/', methods=['POST'])
def data_analyst_agent():
    REQUEST_TIMEOUT = 295.0
    
    if 'questions.txt' not in request.files:
        return jsonify({"error": "questions.txt file is required."}), 400
        
    # Get the model instance with the fallback logic
    model_instance = get_gemini_model() 

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Pass both request.files and the model_instance to the worker function
        future = executor.submit(process_request_worker, request.files, model_instance)
        try:
            result = future.result(timeout=REQUEST_TIMEOUT)
            return jsonify(result), 200
        except concurrent.futures.TimeoutError:
            print("Request processing timed out. Returning timeout response.")
            # Read the questions file again to determine the number of questions.
            questions_content = request.files['questions.txt'].read().decode('utf-8')
            num_questions = len(questions_content.splitlines())
            timeout_response = ["TIMEOUT" for _ in range(num_questions)]
            return jsonify(timeout_response), 200
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# ------------------ Main (for running with a production server) ------------------

if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port, threads=8)
