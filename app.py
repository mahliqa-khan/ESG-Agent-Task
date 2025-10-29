#!/usr/bin/env python3
"""
Flask web application for PDF upload and conversion
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import base64
import requests
import json
from werkzeug.utils import secure_filename
import json_to_html
import uuid
from datetime import datetime
import re
from llm_handler import get_llm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Store processing results in memory (for simple demo)
processing_results = {}

# Store document text and chat history for each upload
document_store = {}


def extract_text_locally(pdf_path):
    """Fallback: Extract text from PDF using local libraries"""
    import pymupdf  # PyMuPDF (fitz)

    print(f"  📚 Using local PDF extraction as fallback...")

    try:
        doc = pymupdf.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Format similar to API response
            pages.append({
                'page_number': page_num + 1,
                'markdown': text
            })

        doc.close()

        result = {
            'total_pages': len(pages),
            'pages': pages,
            'processing_time': 0,
            'source': 'local_extraction'
        }

        print(f"  ✅ Local extraction: {len(pages)} pages extracted")
        return result

    except Exception as e:
        print(f"  ❌ Local extraction failed: {e}")
        raise


def process_pdf_with_api(pdf_path):
    """Process a PDF file using the OCR API with fallback to local extraction"""

    try:
        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode()

        # API endpoint
        url = "https://xumengshe--pdf-ocr-api-fastapi-app.modal.run/process-pdf"

        # Send request
        print(f"  🌐 Attempting API extraction...")
        response = requests.post(
            url,
            json={"pdf_base64": pdf_data},
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )

        # Check response
        if response.status_code == 200:
            print(f"  ✅ API extraction successful")
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    except Exception as api_error:
        print(f"  ⚠️ API extraction failed: {str(api_error)[:100]}")
        print(f"  🔄 Falling back to local PDF extraction...")

        # Fallback to local extraction
        return extract_text_locally(pdf_path)


def fix_latex_syntax(text):
    """Fix common LaTeX syntax issues from OCR"""

    # First, protect existing math delimiters from further processing
    # Store them temporarily with placeholders
    math_blocks = []

    def store_math(match):
        math_blocks.append(match.group(0))
        return f'MATHBLOCK{len(math_blocks)-1}MATHBLOCK'

    # Protect existing $$...$$ and $...$
    text = re.sub(r'\$\$[^\$]+\$\$', store_math, text)
    text = re.sub(r'\$[^\$]+\$', store_math, text)

    # Fix subscripts: f{2} -> f_2, v{1} -> v_1, etc.
    text = re.sub(r'([a-zA-Z])(\{)(\d+)(\})', r'\1_\3', text)

    # Fix superscripts if needed: x{2} after ^ -> x^2
    text = re.sub(r'\^(\{)(\d+)(\})', r'^\2', text)

    # Convert π to \pi
    text = text.replace('π', r'\pi')

    # Fix markdown italics in math: *c* -> c, *M* -> M (when near math operators)
    text = re.sub(r'\*([a-zA-Z])\*', r'\1', text)

    # IMPROVED: Convert LaTeX display math delimiters \[...\] to $$...$$
    # First handle LaTeX-style \[...\] delimiters (common in OCR output)
    def convert_latex_display_math(match):
        content = match.group(1)
        # Clean up the content
        content = re.sub(r'([a-zA-Z])(\{)(\d+)(\})', r'\1_\3', content)
        content = content.strip()
        # Remove trailing equation references like \quad (1.1)
        content = re.sub(r',?\s*\\quad\s*\([0-9.]+\)\s*$', '', content)
        return f'$${content}$$'

    # Match \[...\] (LaTeX display math delimiters)
    text = re.sub(r'\\\[(.+?)\\\]', convert_latex_display_math, text, flags=re.DOTALL)

    # Convert LaTeX inline math delimiters \(...\) to $...$
    def convert_latex_inline_math(match):
        content = match.group(1)
        # Clean up the content
        content = re.sub(r'([a-zA-Z])(\{)(\d+)(\})', r'\1_\3', content)
        return f'${content}$'

    # Match \(...\) (LaTeX inline math delimiters)
    text = re.sub(r'\\\((.+?)\\\)', convert_latex_inline_math, text)

    # Also handle regular [...] with LaTeX content for backwards compatibility
    # BUT: Don't convert \left[ or \right[ - these are LaTeX bracket commands!
    def convert_bracket_math(match):
        full_match = match.group(0)
        before = match.group(1) if match.lastindex >= 1 else ""
        content = match.group(2)

        # Skip \left[ and \right[ patterns
        if before and before in ('\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg'):
            return full_match

        # List of common LaTeX commands and symbols
        latex_indicators = [
            r'\\', r'\geq', r'\leq', r'\neq', r'\in', r'\partial', r'\quad', r'\text',
            r'\Delta', r'\Omega', r'\mathbb', r'\subset', r'\equiv', r'\int', r'\sum',
            r'\prod', r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\pi',
            r'\theta', r'\lambda', r'\mu', r'\sigma', r'\infty', r'\nabla', r'\times',
            r'\cdot', r'\pm', r'\mp', r'\to', r'\rightarrow', r'\leftarrow', r'\approx',
            r'\sim', r'\propto', r'\perp', r'\parallel', r'\cap', r'\cup', r'\vee', r'\wedge',
            '_', '^'  # subscripts and superscripts are strong indicators
        ]

        # Check if content has LaTeX
        if any(ind in content for ind in latex_indicators):
            # Clean up the content
            content = re.sub(r'([a-zA-Z])(\{)(\d+)(\})', r'\1_\3', content)
            content = content.strip()
            content = re.sub(r',?\s*\\quad\s*\([0-9.]+\)\s*$', '', content)
            return f'$${content}$$'
        return full_match

    # Match [...] but capture what comes before to check for \left, \right etc
    text = re.sub(r'(\\left|\\right|\\big|\\Big|\\bigg|\\Bigg)?\[([^\[\]]+)\]', convert_bracket_math, text)

    # IMPROVED: Wrap parentheses containing LaTeX commands in inline math
    def wrap_math(match):
        content = match.group(1)

        # Don't wrap if already processed
        if 'MATHBLOCK' in content:
            return match.group(0)

        # List of LaTeX indicators
        latex_indicators = [
            r'\\geq', r'\\leq', r'\\neq', r'\\in', r'\\subset', r'\\supset',
            r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\epsilon', r'\\pi',
            r'\\mathbb', r'\\Omega', r'\\Delta', r'\\partial',
            r'\\sum', r'\\prod', r'\\int', r'\\nabla', r'\\times', r'\\cdot'
        ]

        # Check if content has LaTeX commands
        if any(ind in content for ind in latex_indicators):
            # Fix subscripts in the content
            content = re.sub(r'([a-zA-Z])(\{)(\d+)(\})', r'\1_\3', content)
            return f'$({content})$'

        return match.group(0)

    # Wrap parentheses containing LaTeX commands
    text = re.sub(r'\(([^)]+)\)', wrap_math, text)

    # Handle inline fractions: (1 - c/M) should become $(1 - c/M)$
    # This is now handled by the above pattern, but we can add specific fraction handling

    # IMPORTANT: Fix underscores in math contexts to prevent markdown emphasis
    # Wrap sequences like v_{i} that are not yet in math mode
    def protect_subscripts(match):
        # If already in a MATHBLOCK, don't touch
        if 'MATHBLOCK' in match.group(0):
            return match.group(0)
        var = match.group(1)
        subscript = match.group(2)
        # Wrap in inline math
        return f'${var}_{{{subscript}}}$'

    # Pattern: variable_subscript that's not already in math
    text = re.sub(r'([a-zA-Z])_([a-zA-Z0-9]+)\b', protect_subscripts, text)

    # Restore protected math blocks
    for i, block in enumerate(math_blocks):
        text = text.replace(f'MATHBLOCK{i}MATHBLOCK', block)

    return text


def extract_clean_text(api_result):
    """Extract clean text from API result for LLM processing"""
    clean_text = []

    # Add validation
    if not api_result:
        print("⚠️ Warning: API result is empty")
        return ""

    pages = api_result.get('pages', [])
    print(f"📄 Extracting text from {len(pages)} pages...")

    for idx, page_data in enumerate(pages, 1):
        markdown_content = page_data.get('markdown', '')

        if not markdown_content:
            print(f"  ⚠️ Page {idx}: No markdown content found")
            continue

        # Remove OCR tags
        text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', markdown_content)
        text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)

        # Clean up extra whitespace
        text = re.sub(r'\n\n\n+', '\n\n', text)

        # Fix LaTeX syntax
        text = fix_latex_syntax(text)

        cleaned = text.strip()
        if cleaned:
            clean_text.append(cleaned)
            print(f"  ✓ Page {idx}: Extracted {len(cleaned)} characters")
        else:
            print(f"  ⚠️ Page {idx}: No text after cleaning")

    full_text = '\n\n'.join(clean_text)
    print(f"✅ Total extracted text: {len(full_text)} characters from {len(clean_text)} pages")
    return full_text


@app.route('/')
def index():
    """Homepage with upload form"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""

    if 'pdf_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['pdf_file']

    if file.filename == '':
        return "No file selected", 400

    if not file.filename.lower().endswith('.pdf'):
        return "Only PDF files are allowed", 400

    # Generate unique ID for this upload
    upload_id = str(uuid.uuid4())

    # Save uploaded file
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
    file.save(pdf_path)

    # Store initial status
    processing_results[upload_id] = {
        'status': 'processing',
        'filename': filename,
        'timestamp': datetime.now().isoformat()
    }

    try:
        # Process PDF with API
        result = process_pdf_with_api(pdf_path)

        # Save JSON result
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f)

        # Extract clean text for LLM
        document_text = extract_clean_text(result)

        # Generate summary using LLM
        summary = "Summary generation in progress..."
        try:
            llm = get_llm()
            summary = llm.generate_summary(document_text)
        except Exception as e:
            summary = f"Summary not available. Set GEMINI_API_KEY to enable. Error: {str(e)}"

        # Store document text for chat
        document_store[upload_id] = {
            'text': document_text,
            'chat_history': []
        }

        # Generate HTML
        html_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.html")
        json_to_html.convert_to_html(json_path, html_path)

        # Update status
        processing_results[upload_id] = {
            'status': 'completed',
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'total_pages': result.get('total_pages', 0),
            'processing_time': result.get('processing_time', 0),
            'html_path': html_path,
            'summary': summary
        }

        # Redirect to view page
        return redirect(url_for('view_result', upload_id=upload_id))

    except Exception as e:
        processing_results[upload_id] = {
            'status': 'failed',
            'filename': filename,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return f"Processing failed: {str(e)}", 500

    finally:
        # Clean up uploaded PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@app.route('/view/<upload_id>')
def view_result(upload_id):
    """View the processed result with summary and chat"""

    if upload_id not in processing_results:
        return "Result not found", 404

    result = processing_results[upload_id]

    if result['status'] == 'processing':
        return "Still processing...", 202

    if result['status'] == 'failed':
        return f"Processing failed: {result.get('error', 'Unknown error')}", 500

    # Render result page with summary and chat
    return render_template('result.html',
                         upload_id=upload_id,
                         filename=result['filename'],
                         total_pages=result['total_pages'],
                         processing_time=result['processing_time'],
                         summary=result.get('summary', 'No summary available'))


@app.route('/api/document/<upload_id>')
def get_document(upload_id):
    """Get the processed HTML document"""
    if upload_id not in processing_results:
        return "Result not found", 404

    result = processing_results[upload_id]
    html_path = result['html_path']
    return send_file(html_path, mimetype='text/html')


@app.route('/api/chat/<upload_id>', methods=['POST'])
def chat(upload_id):
    """Handle chat requests for a specific document"""

    if upload_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404

    data = request.json
    user_question = data.get('question', '')

    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Get LLM handler
        llm = get_llm()

        # Get document data
        doc_data = document_store[upload_id]
        document_text = doc_data['text']
        chat_history = doc_data['chat_history']

        # Generate response
        answer = llm.chat(document_text, user_question, chat_history)

        # Store in chat history
        chat_history.append({
            'user': user_question,
            'assistant': answer
        })

        return jsonify({
            'answer': answer,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': f'Chat error: {str(e)}',
            'success': False
        }), 500


@app.route('/history')
def history():
    """Show upload history"""
    return render_template('history.html', results=processing_results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
