#!/usr/bin/env python3
"""
ESG Scoring Platform - Flask Application
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import io
from datetime import datetime
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env file
load_dotenv()

# Import our modules
import database as db
from llm_handler import get_llm
from app import process_pdf_with_api, extract_clean_text
import json_to_html
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize database
db.init_db()

# Job tracking for async analysis
analysis_jobs = {}
jobs_lock = threading.Lock()


@app.route('/')
def index():
    """Chat-based dashboard - FAST with Gemini 2.0 Flash"""
    questions = db.get_all_scoring_questions()
    return render_template('chat_dashboard.html', questions=questions)


@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Simple test endpoint to verify API is reachable"""
    print(f"✅ Test endpoint called with {request.method}")
    return jsonify({
        'success': True,
        'message': 'API is working!',
        'method': request.method,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/list-pdfs', methods=['GET'])
def list_pdfs():
    """List PDFs available in the project directory"""
    import glob
    pdf_files = glob.glob('*.pdf')
    return jsonify({
        'success': True,
        'pdfs': pdf_files
    })


@app.route('/dashboard')
def dashboard():
    """Legacy dashboard - redirect to simple version"""
    return redirect(url_for('index'))


@app.route('/old-dashboard')
def old_dashboard():
    """Old company-based dashboard"""
    companies = db.get_all_companies()

    # Calculate stats
    total_companies = len(companies)
    total_reports = sum(c.get('report_count', 0) for c in companies)

    # Get latest scores for companies
    for company in companies:
        # This is simplified - in production you'd query scores properly
        company['latest_score'] = None
        company['env_score'] = None
        company['social_score'] = None
        company['gov_score'] = None

    # Calculate average score (placeholder)
    avg_score = 0
    latest_year = max((c.get('latest_year') for c in companies if c.get('latest_year')), default=None)

    stats = {
        'total_companies': total_companies,
        'total_reports': total_reports,
        'avg_score': avg_score,
        'latest_year': latest_year
    }

    return render_template('esg_dashboard.html', companies=companies, stats=stats)


@app.route('/api/companies/add', methods=['POST'])
def add_company():
    """Add a new company"""
    name = request.form.get('name')
    ticker = request.form.get('ticker') or None
    sector = request.form.get('sector') or None
    country = request.form.get('country') or None

    if not name:
        return "Company name is required", 400

    company_id = db.add_company(name, ticker, sector, country)
    return redirect(url_for('company_detail', company_id=company_id))


@app.route('/company/<int:company_id>')
def company_detail(company_id):
    """Company detail page with report history and document library"""
    company = db.get_company(company_id)

    if not company:
        return "Company not found", 404

    # Get all reports with scores
    reports = db.get_company_history(company_id)

    # Get document library
    documents = db.get_company_documents(company_id)

    return render_template('company_detail.html', company=company, reports=reports, documents=documents)


@app.route('/company/<int:company_id>/document/upload', methods=['POST'])
def upload_document(company_id):
    """Upload a document to the company's document library"""
    if 'pdf_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['pdf_file']
    if file.filename == '':
        return "No file selected", 400

    if not file.filename.lower().endswith('.pdf'):
        return "Only PDF files allowed", 400

    # Get form data
    document_type = request.form.get('document_type')
    document_year = request.form.get('document_year')
    description = request.form.get('description')

    # Generate unique ID
    upload_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
    file.save(pdf_path)

    try:
        # Process PDF with OCR API
        result = process_pdf_with_api(pdf_path)

        # Save JSON result
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f)

        # Generate HTML
        html_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.html")
        json_to_html.convert_to_html(json_path, html_path)

        # Add document to library
        db.add_document(
            company_id=company_id,
            document_type=document_type,
            document_year=int(document_year) if document_year else None,
            filename=filename,
            upload_id=upload_id,
            file_path=html_path,
            total_pages=result.get('total_pages', 0),
            description=description
        )

        return redirect(url_for('company_detail', company_id=company_id))

    except Exception as e:
        print(f"Error processing document: {e}")
        return f"Processing failed: {str(e)}", 500
    finally:
        # Clean up uploaded file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@app.route('/company/<int:company_id>/upload', methods=['POST'])
def upload_report(company_id):
    """Upload and process an ESG report"""
    if 'pdf_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['pdf_file']
    if file.filename == '':
        return "No file selected", 400

    if not file.filename.lower().endswith('.pdf'):
        return "Only PDF files allowed", 400

    # Get form data
    report_year = int(request.form.get('report_year'))
    report_type = request.form.get('report_type')

    # Generate unique ID
    upload_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
    file.save(pdf_path)

    try:
        # Process PDF with OCR API
        result = process_pdf_with_api(pdf_path)

        # Save JSON result
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f)

        # Generate HTML
        html_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.html")
        json_to_html.convert_to_html(json_path, html_path)

        # Extract text for AI analysis
        document_text = extract_clean_text(result)

        # Add report to database
        report_id = db.add_report(
            company_id=company_id,
            report_year=report_year,
            report_type=report_type,
            filename=filename,
            upload_id=upload_id,
            file_path=html_path,
            total_pages=result.get('total_pages', 0)
        )

        # Perform ESG scoring asynchronously (in background)
        # For now, we'll do it synchronously
        perform_esg_scoring(report_id, document_text)

        return redirect(url_for('company_detail', company_id=company_id))

    except Exception as e:
        print(f"Error processing report: {e}")
        return f"Processing failed: {str(e)}", 500
    finally:
        # Clean up uploaded file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def perform_esg_scoring(report_id, document_text):
    """
    Perform ESG scoring using AI analysis
    """
    try:
        # Get active framework
        framework = db.get_active_framework()
        if not framework:
            print("No active framework found")
            return

        # Get LLM
        llm = get_llm()

        # Create scoring prompt
        prompt = create_scoring_prompt(framework, document_text)

        # Get AI analysis
        # Using chat function for now - in production use dedicated scoring
        analysis = llm.chat(document_text, prompt, [])

        # Parse AI response (simplified - in production parse structured JSON)
        # For now, generate placeholder scores
        scores = {
            "overall_score": 75.0,
            "environmental_score": 78.0,
            "social_score": 72.0,
            "governance_score": 76.0,
            "detailed_scores": {
                "Environmental": {
                    "Carbon Emissions": 4,
                    "Energy Management": 4,
                    "Water & Waste": 3,
                    "Biodiversity": 3
                },
                "Social": {
                    "Labor Practices": 4,
                    "Health & Safety": 3,
                    "Community Impact": 3,
                    "Supply Chain": 4
                },
                "Governance": {
                    "Board Structure": 4,
                    "Ethics & Compliance": 4,
                    "Transparency": 3,
                    "Risk Management": 4
                }
            },
            "extracted_data": {},
            "ai_reasoning": analysis
        }

        # Save scores to database
        db.save_esg_score(
            report_id=report_id,
            framework_version=framework['version'],
            overall_score=scores['overall_score'],
            env_score=scores['environmental_score'],
            social_score=scores['social_score'],
            gov_score=scores['governance_score'],
            detailed_scores=scores['detailed_scores'],
            extracted_data=scores['extracted_data'],
            ai_reasoning=scores['ai_reasoning']
        )

        print(f"✅ ESG scoring completed for report {report_id}")

    except Exception as e:
        print(f"Error in ESG scoring: {e}")


def create_scoring_prompt(framework, document_text):
    """Create a prompt for ESG scoring"""
    criteria_text = json.dumps(framework['criteria'], indent=2)

    prompt = f"""You are an ESG analyst. Analyze this ESG/sustainability report and provide detailed scoring based on the following framework:

SCORING FRAMEWORK:
{criteria_text}

INSTRUCTIONS:
1. For each criterion, assign a score from 1-5 based on the rubric
2. Extract specific data points mentioned (emissions, diversity metrics, etc.)
3. Provide clear reasoning for each score
4. Identify any gaps in disclosure
5. Note strengths and areas for improvement

Please analyze the report thoroughly and provide detailed scoring with justification."""

    return prompt


@app.route('/report/<int:report_id>/view')
def view_report(report_id):
    """View processed report HTML"""
    # Get report from database
    # For now, redirect to the HTML file
    # In production, load from database and render properly
    return "Report viewer coming soon", 200


@app.route('/report/<int:report_id>/export')
def export_report(report_id):
    """Export ESG scorecard as PDF"""
    # This will generate a professional PDF report with scores and reasoning
    return "Export functionality coming soon", 200


@app.route('/framework')
def view_framework():
    """View current ESG framework"""
    framework = db.get_active_framework()
    return jsonify(framework) if framework else ("No framework found", 404)


# ==================== QUESTIONNAIRE EVALUATION ROUTES ====================

@app.route('/company/<int:company_id>/evaluation/start', methods=['GET', 'POST'])
def start_company_evaluation(company_id):
    """Start evaluation using documents from the library"""
    if request.method == 'GET':
        # Show document selection page
        documents = db.get_company_documents(company_id)
        template = db.get_active_template()
        company = db.get_company(company_id)

        return render_template('select_documents.html',
                             company=company,
                             documents=documents,
                             template=template)

    else:  # POST
        # Get selected documents
        selected_doc_ids = request.form.getlist('document_ids')
        evaluation_year = request.form.get('evaluation_year')

        if not selected_doc_ids:
            return "Please select at least one document", 400

        # Convert to integers
        selected_doc_ids = [int(doc_id) for doc_id in selected_doc_ids]

        try:
            # Get the active template
            template = db.get_active_template()
            if not template:
                return "No active questionnaire template found. Please run setup_sample_questionnaire.py first.", 404

            # Create evaluation with multiple documents
            evaluation_id = db.create_evaluation_with_documents(
                company_id=company_id,
                template_id=template['id'],
                document_ids=selected_doc_ids,
                evaluation_year=int(evaluation_year) if evaluation_year else None
            )

            # Get questions
            questions = db.get_template_questions(template['id'])

            # Get all document texts
            documents = db.get_evaluation_documents(evaluation_id)
            combined_text = get_combined_document_text(documents)

            if not combined_text:
                return "Could not load document text. Please re-upload the documents.", 404

            # Run AI auto-answer for all questions across all documents
            auto_answer_all_questions(evaluation_id, questions, combined_text, documents)

            # Redirect to evaluation interface
            return redirect(url_for('view_evaluation', evaluation_id=evaluation_id))

        except Exception as e:
            print(f"Error starting evaluation: {e}")
            import traceback
            traceback.print_exc()
            return f"Failed to start evaluation: {str(e)}", 500


@app.route('/evaluation/<int:report_id>/start')
def start_evaluation(report_id):
    """Start a new questionnaire evaluation for a report"""
    try:
        # Get the active template
        template = db.get_active_template()
        if not template:
            return "No active questionnaire template found. Please run setup_sample_questionnaire.py first.", 404

        # Create evaluation session
        evaluation_id = db.create_evaluation(report_id, template['id'])

        # Get questions
        questions = db.get_template_questions(template['id'])

        # Get report document text
        # For now, we'll need to load the processed text from the report
        # In production, this would load from stored text or re-extract from PDF
        document_text = get_report_text(report_id)

        if not document_text:
            return "Report text not available. Please re-upload the report.", 404

        # Run AI auto-answer for all questions
        auto_answer_all_questions(evaluation_id, questions, document_text)

        # Redirect to evaluation interface
        return redirect(url_for('view_evaluation', evaluation_id=evaluation_id))

    except Exception as e:
        print(f"Error starting evaluation: {e}")
        return f"Failed to start evaluation: {str(e)}", 500


@app.route('/evaluation/<int:evaluation_id>')
def view_evaluation(evaluation_id):
    """View the evaluation interface"""
    # Get evaluation details
    with db.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT e.*, r.filename, r.report_year, c.name as company_name, t.name as template_name
            FROM evaluations e
            JOIN reports r ON e.report_id = r.id
            JOIN companies c ON r.company_id = c.id
            JOIN questionnaire_templates t ON e.template_id = t.id
            WHERE e.id = ?
        ''', (evaluation_id,))
        row = cursor.fetchone()
        evaluation = dict(row) if row else None

    if not evaluation:
        return "Evaluation not found", 404

    # Get all answers with questions
    answers = db.get_evaluation_answers(evaluation_id)

    # Get chat history
    chat_history = db.get_evaluation_chat(evaluation_id)

    return render_template('evaluation.html',
                         evaluation=evaluation,
                         answers=answers,
                         chat_history=chat_history)


@app.route('/api/evaluation/<int:evaluation_id>/verify/<int:answer_id>', methods=['POST'])
def verify_answer(evaluation_id, answer_id):
    """Mark an answer as human-verified"""
    try:
        with db.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE answers
                SET human_verified = 1
                WHERE id = ? AND evaluation_id = ?
            ''', (answer_id, evaluation_id))

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/evaluation/<int:evaluation_id>/edit/<int:answer_id>', methods=['POST'])
def edit_answer(evaluation_id, answer_id):
    """Edit an answer (human override)"""
    try:
        data = request.get_json()
        answer_text = data.get('answer_text')
        score = data.get('score')
        notes = data.get('notes')

        db.update_answer(answer_id, answer_text, score, human_verified=True, notes=notes)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/evaluation/<int:evaluation_id>/chat', methods=['POST'])
def evaluation_chat(evaluation_id):
    """Handle chat messages during evaluation"""
    try:
        data = request.get_json()
        user_message = data.get('message')
        question_id = data.get('question_id')

        # Get report text and current question/answer context
        context = get_evaluation_context(evaluation_id, question_id)

        # Get AI response
        llm = get_llm()
        ai_response = llm.chat(
            context['document_text'],
            user_message,
            context['chat_history']
        )

        # Save chat message
        db.save_chat_message(evaluation_id, user_message, ai_response, question_id)

        return jsonify({
            "success": True,
            "response": ai_response
        })

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/evaluation/<int:evaluation_id>/complete', methods=['POST'])
def complete_evaluation_route(evaluation_id):
    """Mark evaluation as complete"""
    try:
        db.complete_evaluation(evaluation_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== AI AUTO-ANSWER ENGINE ====================

def get_report_text(report_id):
    """Get the document text for a report"""
    try:
        with db.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT upload_id FROM reports WHERE id = ?', (report_id,))
            result = cursor.fetchone()

            if not result:
                return None

            upload_id = result['upload_id']
            json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")

            if not os.path.exists(json_path):
                return None

            with open(json_path, 'r') as f:
                result_data = json.load(f)
                return extract_clean_text(result_data)
    except Exception as e:
        print(f"Error loading report text: {e}")
        return None


def get_combined_document_text(documents):
    """Combine text from multiple documents with document labels"""
    combined_text = ""

    for doc in documents:
        try:
            upload_id = doc['upload_id']
            json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    result_data = json.load(f)
                    doc_text = extract_clean_text(result_data)

                    # Add document header
                    combined_text += f"\n\n========== DOCUMENT: {doc['filename']} ({doc['document_type']}) ==========\n\n"
                    combined_text += doc_text
        except Exception as e:
            print(f"Error loading document {doc.get('filename')}: {e}")

    return combined_text


def auto_answer_all_questions(evaluation_id, questions, document_text, documents=None):
    """
    Use AI to automatically answer all questions based on the document(s)
    documents: list of document dicts with filename, document_type for source tracking
    """
    try:
        llm = get_llm()

        print(f"Starting auto-answer for {len(questions)} questions...")
        if documents:
            print(f"  Searching across {len(documents)} documents")

        for idx, question in enumerate(questions, 1):
            print(f"  Answering question {idx}/{len(questions)}: {question['question_number']}")

            # Create targeted prompt for this question
            doc_context = ""
            if documents and len(documents) > 1:
                doc_list = ", ".join([f"{d['filename']} ({d['document_type']})" for d in documents])
                doc_context = f"\n\nNOTE: You are searching across multiple documents: {doc_list}\nWhen citing evidence, mention which document it came from.\n"

            prompt = f"""You are an ESG analyst reviewing sustainability documents. Answer the following question based ONLY on the information in the document(s) provided.{doc_context}

QUESTION: {question['question_text']}

CATEGORY: {question['category']} ({question['pillar']})

GUIDANCE: {question['scoring_guidance']}

INSTRUCTIONS:
1. Provide a clear, factual answer based on what's in the document
2. If the information is disclosed, extract specific data points, metrics, and facts
3. If the information is NOT in the document, state "Not disclosed" or "Information not found"
4. Cite specific evidence by including relevant quotes or data
5. Indicate your confidence level (0-100%) based on clarity and completeness of disclosure
6. If you find the information, note which page(s) it appears on

FORMAT YOUR RESPONSE AS:
ANSWER: [Your answer here]
EVIDENCE: [Relevant quotes or data from the document]
CONFIDENCE: [0-100]
PAGE_REFERENCES: [Page numbers where found, e.g., "5, 12-14"]

If information is not found, still provide this format but note "Not disclosed" in the answer."""

            try:
                # Get AI response
                response = llm.chat(document_text, prompt, [])

                # Parse response
                parsed = parse_ai_answer(response)

                # Track source documents
                source_docs_json = None
                if documents:
                    source_docs_json = json.dumps([{
                        'filename': d['filename'],
                        'document_type': d['document_type']
                    } for d in documents])

                # Save answer to database
                db.save_answer(
                    evaluation_id=evaluation_id,
                    question_id=question['id'],
                    answer_text=parsed['answer'],
                    score=None,  # Score will be assigned during review
                    confidence=parsed['confidence'],
                    evidence=parsed['evidence'],
                    page_references=parsed['page_references'],
                    source_documents=source_docs_json,
                    ai_generated=True
                )

            except Exception as e:
                print(f"    Error answering question {question['question_number']}: {e}")
                # Save error state
                db.save_answer(
                    evaluation_id=evaluation_id,
                    question_id=question['id'],
                    answer_text=f"Error processing: {str(e)}",
                    confidence=0.0,
                    ai_generated=True
                )

        print(f"✅ Auto-answer completed for evaluation {evaluation_id}")

    except Exception as e:
        print(f"Error in auto_answer_all_questions: {e}")
        raise


def parse_ai_answer(response_text):
    """Parse AI response into structured format"""
    result = {
        'answer': '',
        'evidence': '',
        'confidence': 0.5,
        'page_references': ''
    }

    try:
        # Simple parsing - look for our format markers
        lines = response_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('ANSWER:'):
                current_section = 'answer'
                result['answer'] = line.replace('ANSWER:', '').strip()
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                result['evidence'] = line.replace('EVIDENCE:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                current_section = 'confidence'
                conf_text = line.replace('CONFIDENCE:', '').strip()
                # Extract number from confidence text
                import re
                conf_match = re.search(r'(\d+)', conf_text)
                if conf_match:
                    result['confidence'] = float(conf_match.group(1)) / 100.0
            elif line.startswith('PAGE_REFERENCES:'):
                current_section = 'page_references'
                result['page_references'] = line.replace('PAGE_REFERENCES:', '').strip()
            elif current_section and line:
                # Continue multi-line sections
                result[current_section] += ' ' + line

        # Fallback: if parsing failed, just use the whole response as answer
        if not result['answer']:
            result['answer'] = response_text[:500]  # First 500 chars

    except Exception as e:
        print(f"Error parsing AI response: {e}")
        result['answer'] = response_text[:500]

    return result


def get_evaluation_context(evaluation_id, question_id=None):
    """Get context for chat - document text, current question, existing answer"""
    context = {
        'document_text': '',
        'chat_history': []
    }

    try:
        # Get evaluation and report
        with db.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.upload_id
                FROM evaluations e
                JOIN reports r ON e.report_id = r.id
                WHERE e.id = ?
            ''', (evaluation_id,))
            result = cursor.fetchone()

            if result:
                upload_id = result['upload_id']
                json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        result_data = json.load(f)
                        context['document_text'] = extract_clean_text(result_data)

        # Get chat history
        chat_records = db.get_evaluation_chat(evaluation_id)
        context['chat_history'] = [
            {'role': 'user', 'content': msg['user_message']}
            for msg in chat_records
        ] + [
            {'role': 'assistant', 'content': msg['ai_response']}
            for msg in chat_records
        ]

    except Exception as e:
        print(f"Error getting evaluation context: {e}")

    return context


# ==================== SCORING QUESTIONS MANAGEMENT ====================

@app.route('/scoring-questions')
def manage_scoring_questions():
    """Manage scoring questions library"""
    questions = db.get_all_scoring_questions()
    return render_template('scoring_questions.html', questions=questions)


@app.route('/api/scoring-questions/add', methods=['POST'])
def add_scoring_question_route():
    """Add a new scoring question"""
    data = request.get_json()

    question_id = db.add_scoring_question(
        question_text=data['question_text'],
        sub_questions=data['sub_questions'],
        rubric=data['rubric'],
        category=data.get('category')
    )

    return jsonify({'success': True, 'question_id': question_id})


@app.route('/api/scoring-questions/<int:question_id>/edit', methods=['POST'])
def edit_scoring_question_route(question_id):
    """Edit a scoring question"""
    data = request.get_json()

    db.update_scoring_question(
        question_id=question_id,
        question_text=data['question_text'],
        sub_questions=data['sub_questions'],
        rubric=data['rubric'],
        category=data.get('category')
    )

    return jsonify({'success': True})


@app.route('/api/scoring-questions/<int:question_id>/delete', methods=['POST'])
def delete_scoring_question_route(question_id):
    """Delete a scoring question"""
    db.delete_scoring_question(question_id)
    return jsonify({'success': True})


# ==================== DOCUMENT SCORING WORKFLOW ====================

@app.route('/document/<int:document_id>/score')
def score_document(document_id):
    """Score a document using scoring questions"""
    # Get document details
    with db.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.*, c.name as company_name
            FROM documents d
            JOIN companies c ON d.company_id = c.id
            WHERE d.id = ?
        ''', (document_id,))
        row = cursor.fetchone()
        document = dict(row) if row else None

    if not document:
        return "Document not found", 404

    # Get all scoring questions
    questions = db.get_all_scoring_questions()

    # Get existing scoring results for this document
    existing_results = db.get_document_scoring_results(document_id)

    return render_template('score_document.html',
                         document=document,
                         questions=questions,
                         existing_results=existing_results)


@app.route('/api/document/<int:document_id>/apply-question', methods=['POST'])
def apply_scoring_question(document_id):
    """Apply a scoring question to a document"""
    try:
        data = request.get_json()
        question_id = data['question_id']

        # Get document details
        with db.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
            row = cursor.fetchone()
            document = dict(row) if row else None

        if not document:
            return jsonify({'success': False, 'error': 'Document not found'}), 404

        # Get question
        question = db.get_scoring_question(question_id)
        if not question:
            return jsonify({'success': False, 'error': 'Question not found'}), 404

        # Load document text
        upload_id = document['upload_id']
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")

        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'Document content not found'}), 404

        with open(json_path, 'r') as f:
            result_data = json.load(f)
            document_text = extract_clean_text(result_data)

        # Perform two-stage scoring
        scoring_result = perform_two_stage_scoring(question, document_text)

        # Save result
        result_id = db.save_scoring_result(
            document_id=document_id,
            company_id=document['company_id'],
            scoring_question_id=question_id,
            main_answer=scoring_result['main_answer'],
            score=scoring_result['score'],
            reasoning=scoring_result['reasoning'],
            evidence=scoring_result.get('evidence'),
            page_references=scoring_result.get('page_references')
        )

        return jsonify({
            'success': True,
            'result_id': result_id,
            'result': scoring_result
        })

    except Exception as e:
        print(f"Error applying scoring question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def perform_two_stage_scoring(question, document_text):
    """
    Two-stage scoring process:
    1. Answer the main question from the document
    2. Evaluate the answer against sub-questions and rubric to assign score (1-5)
    """
    try:
        llm = get_llm()

        # Stage 1: Answer the main question
        print(f"Stage 1: Answering main question")

        answer_prompt = f"""Based on the following document, please answer this question:

QUESTION: {question['question_text']}

Provide a detailed answer based on the information in the document. Include:
1. Direct answer to the question
2. Specific evidence and quotes from the document
3. Page references if available
4. Any relevant data points or metrics

If the information is not in the document, clearly state that."""

        main_answer = llm.chat(document_text, answer_prompt, [])

        # Stage 2: Score the answer against the rubric
        print(f"Stage 2: Scoring the answer against rubric")

        scoring_prompt = f"""You are an ESG evaluator. You need to score an answer based on specific criteria.

ORIGINAL QUESTION: {question['question_text']}

THE ANSWER PROVIDED:
{main_answer}

EVALUATION CRITERIA (Sub-questions to assess):
{question['sub_questions']}

SCORING RUBRIC (1-5 scale):
{question['rubric']}

Based on how well the answer addresses the sub-questions and meets the rubric criteria, provide:
1. A score from 1.0 to 5.0 (in 0.5 increments only: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
2. Detailed reasoning explaining why you gave this score
3. Reference which sub-questions were answered and which were missing

FORMAT YOUR RESPONSE EXACTLY AS:
SCORE: [number between 1.0 and 5.0]
REASONING: [Your detailed reasoning here explaining the score based on the rubric]"""

        scoring_response = llm.chat(main_answer, scoring_prompt, [])

        # Parse score and reasoning
        parsed = parse_scoring_response(scoring_response)

        return {
            'main_answer': main_answer,
            'score': parsed['score'],
            'reasoning': parsed['reasoning']
        }

    except Exception as e:
        print(f"Error in two-stage scoring: {e}")
        raise


def parse_scoring_response(response):
    """Parse AI scoring response to extract score and reasoning"""
    result = {
        'score': 0.0,
        'reasoning': ''
    }

    try:
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('SCORE:'):
                score_text = line.replace('SCORE:', '').strip()
                # Extract number
                import re
                score_match = re.search(r'(\d+\.?\d*)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                    # Round to nearest 0.5
                    score = round(score * 2) / 2
                    # Clamp between 1.0 and 5.0
                    score = max(1.0, min(5.0, score))
                    result['score'] = score

            elif line.startswith('REASONING:'):
                # Get all remaining text as reasoning
                result['reasoning'] = line.replace('REASONING:', '').strip()
                # Append subsequent lines
                for j in range(i + 1, len(lines)):
                    result['reasoning'] += ' ' + lines[j].strip()
                break

        # Fallback
        if not result['reasoning']:
            result['reasoning'] = response

    except Exception as e:
        print(f"Error parsing scoring response: {e}")
        result['reasoning'] = response

    return result


# ==================== SIMPLE WORKFLOW - MULTI-DOCUMENT PROCESSING ====================

@app.route('/api/process-documents', methods=['POST'])
def process_documents():
    """Process multiple documents with selected questions - OPTIMIZED"""
    try:
        # Get uploaded files
        files = request.files.getlist('documents')
        question_ids = json.loads(request.form.get('question_ids', '[]'))

        print(f"📥 Received {len(files)} files and {len(question_ids)} questions")

        if not files or len(files) == 0:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400

        if not question_ids:
            return jsonify({'success': False, 'error': 'No questions selected'}), 400

        # Generate session ID for these results
        session_id = str(uuid.uuid4())
        results = []

        # Get LLM once (reuse connection)
        llm = get_llm()
        print("✅ LLM initialized")

        # Process each document
        for idx, file in enumerate(files, 1):
            if not file.filename.lower().endswith('.pdf'):
                print(f"⚠️ Skipping non-PDF: {file.filename}")
                continue

            print(f"📄 Processing {idx}/{len(files)}: {file.filename}")

            # Save and process PDF
            upload_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
            file.save(pdf_path)

            try:
                # Process PDF with OCR API
                print(f"  🔍 Extracting text from {filename}...")
                result = process_pdf_with_api(pdf_path)

                # Save JSON result
                json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{upload_id}.json")
                with open(json_path, 'w') as f:
                    json.dump(result, f)

                # Extract text
                document_text = extract_clean_text(result)
                print(f"  ✅ Extracted {len(document_text)} characters")

                # Apply each selected question
                doc_scores = []
                for q_idx, question_id in enumerate(question_ids, 1):
                    question = db.get_scoring_question(question_id)
                    if question:
                        print(f"  📊 Scoring question {q_idx}/{len(question_ids)}: {question['question_text'][:50]}...")

                        try:
                            # FAST VERSION: Single-stage scoring with combined prompt
                            scoring_result = perform_fast_scoring(question, document_text, llm)

                            doc_scores.append({
                                'question_id': question_id,
                                'question_text': question['question_text'],
                                'category': question.get('category'),
                                'main_answer': scoring_result['main_answer'],
                                'score': scoring_result['score'],
                                'reasoning': scoring_result['reasoning']
                            })
                            print(f"  ✅ Score: {scoring_result['score']}")
                        except Exception as e:
                            print(f"  ❌ Error scoring question: {e}")
                            doc_scores.append({
                                'question_id': question_id,
                                'question_text': question['question_text'],
                                'category': question.get('category'),
                                'main_answer': f'Error: {str(e)}',
                                'score': 0.0,
                                'reasoning': 'Processing failed'
                            })

                results.append({
                    'filename': filename,
                    'upload_id': upload_id,
                    'total_pages': result.get('total_pages', 0),
                    'scores': doc_scores
                })
                print(f"✅ Completed {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                # Add error result
                results.append({
                    'filename': filename,
                    'upload_id': upload_id,
                    'total_pages': 0,
                    'scores': [],
                    'error': str(e)
                })
            finally:
                # Clean up uploaded file
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

        # Save results to session storage
        session_results[session_id] = {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        print(f"🎉 All done! Session: {session_id}")

        return jsonify({
            'success': True,
            'session_id': session_id,
            'documents_processed': len([r for r in results if 'error' not in r])
        })

    except Exception as e:
        print(f"❌ Fatal error processing documents: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def perform_fast_scoring(question, document_text, llm):
    """
    FASTER VERSION: Single-stage scoring instead of two-stage
    Combines answering and scoring into one LLM call
    """
    try:
        print(f"    🚀 Fast scoring mode")

        # Combined prompt that does both answer extraction and scoring
        combined_prompt = f"""You are an ESG analyst. Analyze the document and provide a scored answer.

QUESTION: {question['question_text']}

EVALUATION CRITERIA:
{question['sub_questions']}

SCORING RUBRIC (1-5 scale):
{question['rubric']}

INSTRUCTIONS:
1. Find relevant information in the document to answer the question
2. Score the answer based on how well it meets the evaluation criteria
3. Provide detailed reasoning that explains WHY the answer is valid

FORMAT YOUR RESPONSE EXACTLY AS:
ANSWER: [Your answer from the document - be specific with data/quotes]
SCORE: [number between 1.0 and 5.0 in 0.5 increments]
REASONING: [Provide comprehensive reasoning covering:
1. WHY the answer is valid:
   - Cite specific evidence (sections, page numbers, quotes)
   - Explain how each component meets the evaluation criteria
   - Justify why the listed items qualify as valid answers
2. WHY this specific score was given:
   - Reference the scoring rubric levels (1.0-5.0)
   - Explain which rubric criteria were met vs. not met
   - Justify why it's this score and not higher/lower
   - Connect the quality/completeness of evidence to the score
Do NOT just repeat the answer - provide analytical reasoning]

If the information is not in the document, state "Not disclosed" and score appropriately."""

        # Single LLM call
        response = llm.chat(document_text, combined_prompt, [])

        # Parse response
        parsed = parse_fast_scoring_response(response)

        return {
            'main_answer': parsed['answer'],
            'score': parsed['score'],
            'reasoning': parsed['reasoning']
        }

    except Exception as e:
        print(f"    ❌ Fast scoring error: {e}")
        raise


def parse_fast_scoring_response(response):
    """Parse the combined scoring response"""
    result = {
        'answer': '',
        'score': 0.0,
        'reasoning': ''
    }

    try:
        lines = response.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if line_stripped.startswith('ANSWER:'):
                current_section = 'answer'
                result['answer'] = line_stripped.replace('ANSWER:', '').strip()
            elif line_stripped.startswith('SCORE:'):
                current_section = 'score'
                score_text = line_stripped.replace('SCORE:', '').strip()
                import re
                score_match = re.search(r'(\d+\.?\d*)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                    score = round(score * 2) / 2  # Round to nearest 0.5
                    score = max(1.0, min(5.0, score))  # Clamp 1-5
                    result['score'] = score
            elif line_stripped.startswith('REASONING:'):
                current_section = 'reasoning'
                result['reasoning'] = line_stripped.replace('REASONING:', '').strip()
                # Append subsequent lines
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        result['reasoning'] += ' ' + lines[j].strip()
                break
            elif current_section and line_stripped and not line_stripped.startswith(('ANSWER:', 'SCORE:', 'REASONING:')):
                # Continue multi-line sections
                result[current_section] += ' ' + line_stripped

        # Fallback
        if not result['answer']:
            result['answer'] = response[:500]
        if not result['reasoning']:
            result['reasoning'] = 'See answer above'

    except Exception as e:
        print(f"Parse error: {e}")
        result['answer'] = response[:500]
        result['reasoning'] = 'Parsing failed'

    return result


@app.route('/results/<session_id>')
def view_results(session_id):
    """View results for a session"""
    results_data = session_results.get(session_id)

    if not results_data:
        return "Results not found or expired", 404

    results = results_data['results']

    # Calculate summary stats
    total_questions = 0
    total_scores = []

    for doc_result in results:
        total_questions = len(doc_result['scores'])
        for score in doc_result['scores']:
            total_scores.append(score['score'])

    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0

    return render_template('results_page.html',
                         session_id=session_id,
                         results=results,
                         total_questions=total_questions,
                         avg_score=avg_score)


@app.route('/api/export-results/<session_id>')
def export_results(session_id):
    """Export results as CSV"""
    results_data = session_results.get(session_id)

    if not results_data:
        return "Results not found", 404

    # Generate CSV
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['Document', 'Question', 'Category', 'Score', 'Answer', 'Reasoning'])

    # Data
    for doc_result in results_data['results']:
        for score in doc_result['scores']:
            writer.writerow([
                doc_result['filename'],
                score['question_text'],
                score.get('category', ''),
                score['score'],
                score['main_answer'],
                score['reasoning']
            ])

    # Return as download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'esg_scores_{session_id[:8]}.csv'
    )


# Session storage for results (in production, use database)
session_results = {}


# ==================== CHAT-BASED ANALYSIS (GEMINI 2.0 FLASH) ====================

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of analysis job"""
    with jobs_lock:
        job = analysis_jobs.get(job_id)

        if not job:
            return jsonify({'success': False, 'error': 'Job not found'}), 404

        return jsonify({
            'success': True,
            'status': job['status'],
            'progress': job.get('progress', 0),
            'results': job.get('results'),
            'error': job.get('error')
        })


def run_analysis_in_background(job_id, saved_files, questions_text, scoring_criteria, provider):
    """Background function that performs the actual analysis"""
    try:
        with jobs_lock:
            analysis_jobs[job_id]['status'] = 'processing'
            analysis_jobs[job_id]['progress'] = 10

        from openai import OpenAI
        from huggingface_hub import InferenceClient

        print(f"🔄 [Job {job_id}] Starting background analysis...")
        print(f"⚡ FAST MODE: {len(saved_files)} docs")
        print(f"📋 Questions: {questions_text[:100]}...")
        print(f"⭐ Scoring: {scoring_criteria[:100]}...")

        # Configure client based on provider
        print(f"🤖 Using LLM Provider: {provider}")

        if provider == 'qwen':
            client = OpenAI(
                base_url=config.QWEN_API_URL,
                api_key="not-needed"
            )
            model_name = config.LLM_MODELS['qwen']['model']
        elif provider == 'openrouter':
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
            model = None
            model_name = config.LLM_MODELS['openrouter']['model']
        elif provider == 'huggingface':
            client = InferenceClient(token=config.HUGGINGFACE_API_KEY)
            model = None
            model_name = config.LLM_MODELS['huggingface']['model']
        elif provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            client = None
            model_name = 'gemini-2.0-flash-exp'
        else:
            raise Exception(f'Unsupported provider: {provider}')

        results = []
        total_files = len(saved_files)

        # Process each document
        for file_idx, (pdf_path, filename) in enumerate(saved_files):
            try:
                print(f"📄 Processing: {filename}")

                with jobs_lock:
                    analysis_jobs[job_id]['progress'] = 10 + int((file_idx / total_files) * 80)

                # Extract text
                result = process_pdf_with_api(pdf_path)
                json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_{filename}.json")
                with open(json_path, 'w') as f:
                    json.dump(result, f)

                document_text = extract_clean_text(result)
                total_pages = result.get('total_pages', 0)

                if not document_text or len(document_text) < 100:
                    print(f"  ⚠️ WARNING: Very little text extracted ({len(document_text)} chars)")
                else:
                    print(f"  ✅ Extracted {len(document_text)} chars from {total_pages} pages")

                # For chat-based analysis, we use the questions_text directly
                # No database lookup needed

                # Decide whether to use chunking
                doc_size_chars = len(document_text)
                use_chunking = doc_size_chars > 200000 or total_pages > 100  # Higher threshold for chat mode

                # For chat mode, always use single-pass analysis with simplified prompt
                print(f"  📄 Document: ({doc_size_chars} chars, {total_pages} pages)")
                print(f"  ⚡ Using single-pass analysis with chat-mode prompt...")

                # Truncate document if too large (keep first 100k chars)
                if doc_size_chars > 100000:
                    document_text = document_text[:100000] + "\n\n[Document truncated for analysis...]"
                    print(f"  ✂️ Truncated to 100k chars for faster processing")

                mega_prompt = f"""You are analyzing an ESG/sustainability document.

Document: {filename}
Total Length: {len(document_text)} characters

Questions to answer:
{questions_text}

Scoring Criteria:
{scoring_criteria}

Document Text:
{document_text}

Analyze the document and answer all the questions based on the scoring criteria provided.

Return your response as a JSON object with this structure:
{{
  "summary": "Brief summary of the document and main findings",
  "analysis": "Detailed analysis addressing all the questions",
  "overall_score": 7.5,
  "key_findings": [
    "Finding 1",
    "Finding 2"
  ]
}}

IMPORTANT:
- Use **Bold Headers:** to organize your analysis
- Keep paragraphs short (2-3 sentences max)
- Use bullet points for lists
- Return ONLY the JSON object
- No markdown code blocks (no ```json)
- Ensure valid JSON formatting"""

                print(f"  🚀 Calling {model_name} for analysis...")

                try:
                    if provider == 'huggingface':
                        # HuggingFace InferenceClient uses text_generation
                        response_text = client.text_generation(
                            mega_prompt,
                            model=model_name,
                            max_new_tokens=2000
                        )
                        print(f"  ✅ Got response from Hugging Face")
                    elif provider in ['qwen', 'openrouter']:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": mega_prompt}],
                            temperature=0.7
                        )
                        response_text = response.choices[0].message.content

                        import re
                        if '<think>' in response_text:
                            response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
                            print(f"  ✅ Got response (cleaned from thinking tags)")
                        else:
                            print(f"  ✅ Got response")
                    else:  # gemini
                        response = model.generate_content(mega_prompt, request_options={"timeout": 60})
                        response_text = response.text

                    print(f"  ✅ Response size: {len(response_text)} chars")

                    # Ensure we have some response
                    if not response_text or len(response_text.strip()) < 10:
                        print(f"  ⚠️ WARNING: Response is empty or too short!")
                        response_text = "No analysis was generated. Please try again or check your API configuration."

                    # Parse JSON response for chat mode
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    analysis_data = None

                    if json_match:
                        try:
                            analysis_data = json.loads(json_match.group())
                            print(f"  ✅ Parsed JSON response successfully")
                        except json.JSONDecodeError as e:
                            print(f"  ⚠️ Failed to parse JSON: {e}")
                            analysis_data = None
                    else:
                        print(f"  ⚠️ No JSON found in response, using raw text")

                    # Build the response in the format the frontend expects
                    if analysis_data and isinstance(analysis_data, dict):
                        # Format as a single "question" result with the full analysis
                        summary = analysis_data.get('summary', '')
                        analysis = analysis_data.get('analysis', '')
                        findings = analysis_data.get('key_findings', [])

                        formatted_answer = ""
                        if summary:
                            # Clean up any JSON artifacts
                            summary_clean = str(summary).replace('{', '').replace('}', '').strip()
                            formatted_answer += f"**Summary:**\n{summary_clean}\n\n"
                        if analysis:
                            # Clean up any JSON artifacts
                            analysis_clean = str(analysis).replace('{', '').replace('}', '').strip()
                            formatted_answer += f"**Analysis:**\n{analysis_clean}\n\n"
                        if findings and len(findings) > 0:
                            formatted_answer += "**Key Findings:**\n"
                            for finding in findings:
                                # Ensure finding is a string and clean up any artifacts
                                finding_str = str(finding).replace('{', '').replace('}', '').strip()
                                if finding_str:  # Only add non-empty findings
                                    formatted_answer += f"• {finding_str}\n"

                        # If we still have no content, use the raw response
                        if not formatted_answer or len(formatted_answer.strip()) < 20:
                            formatted_answer = response_text
                            print(f"  ⚠️ Formatted answer was empty, using raw response")

                        scores = [{
                            'question_text': questions_text[:200] + '...' if len(questions_text) > 200 else questions_text,
                            'main_answer': formatted_answer,
                            'score': float(analysis_data.get('overall_score', 0)),
                            'reasoning': analysis_data.get('analysis', analysis_data.get('reasoning', 'Analysis completed')),
                            'confidence': 0.85,
                            'category': 'ESG Analysis'
                        }]
                    else:
                        # Fallback: treat entire response as analysis
                        print(f"  📝 Using fallback: displaying raw response")
                        scores = [{
                            'question_text': questions_text[:200] + '...' if len(questions_text) > 200 else questions_text,
                            'main_answer': response_text if response_text else "No response generated",
                            'score': 0,
                            'reasoning': 'Raw LLM response (JSON parsing failed)',
                            'confidence': 0.5,
                            'category': 'ESG Analysis'
                        }]

                    print(f"  ✅ Created scores array with {len(scores)} item(s)")

                except Exception as api_error:
                    print(f"  ❌ LLM API Error: {api_error}")
                    raise

                results.append({
                    'filename': filename,
                    'scores': scores
                })

            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                import traceback
                error_trace = traceback.format_exc()
                print(error_trace)

                error_message = f"{type(e).__name__}: {str(e)}"
                results.append({
                    'filename': filename,
                    'scores': [],
                    'error': error_message,
                    'error_details': error_trace[:500]
                })
            finally:
                # Clean up saved file
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

        print(f"🎉 [Job {job_id}] Done! Processed {len(results)} documents")

        # Update job status
        with jobs_lock:
            analysis_jobs[job_id]['status'] = 'completed'
            analysis_jobs[job_id]['progress'] = 100
            analysis_jobs[job_id]['results'] = results

    except Exception as e:
        print(f"❌ [Job {job_id}] Fatal error: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)

        with jobs_lock:
            analysis_jobs[job_id]['status'] = 'failed'
            analysis_jobs[job_id]['error'] = str(e)
            analysis_jobs[job_id]['error_details'] = error_trace[:500]


@app.route('/api/chat-analyze', methods=['POST'])
def chat_analyze():
    """NON-BLOCKING analysis - returns job_id immediately, runs in background"""
    try:
        print("="*80)
        print("🚀 API ENDPOINT CALLED: /api/chat-analyze (ASYNC MODE)")
        print("="*80)

        # Get files, questions, and scoring criteria from form
        files = request.files.getlist('documents')
        questions_text = request.form.get('questions', '').strip()
        scoring_criteria = request.form.get('scoring_criteria', '').strip()

        print(f"⚡ ASYNC MODE: {len(files)} docs")
        print(f"📋 Questions: {questions_text[:100]}...")
        print(f"⭐ Scoring Criteria: {scoring_criteria[:100]}...")

        if not files:
            return jsonify({'success': False, 'error': 'Missing files'}), 400

        if not questions_text:
            return jsonify({'success': False, 'error': 'Missing questions'}), 400

        if not scoring_criteria:
            return jsonify({'success': False, 'error': 'Missing scoring criteria'}), 400

        # Get provider
        provider = os.getenv('LLM_PROVIDER', 'qwen')
        print(f"🤖 Using LLM Provider: {provider}")

        # Generate job ID
        job_id = str(uuid.uuid4())
        print(f"📋 Created job: {job_id}")

        # Save all files immediately
        saved_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue

            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
            file.save(pdf_path)
            saved_files.append((pdf_path, filename))

            # Log file size
            file_size = os.path.getsize(pdf_path) / 1024
            print(f"💾 Saved: {filename} ({file_size:.1f} KB) -> {pdf_path}")

        # Initialize job status
        with jobs_lock:
            analysis_jobs[job_id] = {
                'status': 'queued',
                'progress': 0,
                'created_at': datetime.now().isoformat(),
                'files': [f[1] for f in saved_files],
                'questions': questions_text,
                'scoring_criteria': scoring_criteria
            }

        # Start background thread
        thread = threading.Thread(
            target=run_analysis_in_background,
            args=(job_id, saved_files, questions_text, scoring_criteria, provider),
            daemon=True
        )
        thread.start()
        print(f"🚀 Started background thread for job {job_id}")

        # Return immediately with job_id
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'queued',
            'message': 'Analysis started in background. Use /api/status/{job_id} to check progress.'
        }), 202  # 202 Accepted

    except Exception as e:
        print(f"❌ Error starting analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Failed to start analysis: {str(e)}'}), 500




def split_document_into_chunks(document_text, pages_data, chunk_size=10):
    """Split document into chunks of approximately chunk_size pages"""
    chunks = []

    if not pages_data:
        # Fallback: split by character count
        max_chars = 30000  # ~10 pages worth
        for i in range(0, len(document_text), max_chars):
            chunk_text = document_text[i:i + max_chars]
            chunks.append({
                'chunk_id': len(chunks) + 1,
                'text': chunk_text,
                'pages': f"chars {i}-{i+len(chunk_text)}",
                'page_numbers': []
            })
        return chunks

    # Split by pages
    total_pages = len(pages_data)
    for i in range(0, total_pages, chunk_size):
        chunk_pages = pages_data[i:i + chunk_size]
        chunk_text = '\n\n'.join([p.get('markdown', '') for p in chunk_pages])

        page_numbers = [p.get('page_number', i+j+1) for j, p in enumerate(chunk_pages)]

        chunks.append({
            'chunk_id': len(chunks) + 1,
            'text': chunk_text,
            'pages': f"pages {page_numbers[0]}-{page_numbers[-1]}",
            'page_numbers': page_numbers,
            'page_count': len(chunk_pages)
        })

    return chunks


def analyze_document_in_chunks(document_text, pages_data, selected_questions, llm_client, filename, provider='qwen'):
    """Analyze document by splitting into chunks and processing each"""

    print(f"  📑 Splitting document into chunks...")

    # Determine chunk size based on document size
    # OPTIMIZED: Use larger chunks for faster processing
    total_pages = len(pages_data) if pages_data else len(document_text) // 3000

    # For Hugging Face: smaller chunks due to 32K token limit
    if provider == 'huggingface':
        if total_pages <= 30:
            chunk_size = 8  # Small doc: 8 pages per chunk
        elif total_pages <= 60:
            chunk_size = 10  # Medium doc: 10 pages per chunk
        else:
            chunk_size = 12  # Large doc: 12 pages per chunk (fits within HF token limits)
    else:
        # For other providers (Qwen local, OpenRouter, Gemini)
        if total_pages <= 30:
            chunk_size = 15  # Small doc: 15 pages per chunk
        elif total_pages <= 60:
            chunk_size = 25  # Medium doc: 25 pages per chunk
        else:
            chunk_size = 30  # Large doc: 30 pages per chunk

    chunks = split_document_into_chunks(document_text, pages_data, chunk_size)
    print(f"  📦 Created {len(chunks)} chunks (chunk size: {chunk_size} pages)")

    # Prepare questions JSON
    questions_json = []
    for i, q in enumerate(selected_questions, 1):
        questions_json.append({
            "question_number": i,
            "question_text": q['question_text'],
            "category": q.get('category', 'N/A'),
            "sub_questions": q['sub_questions'],
            "rubric": q['rubric']
        })

    # Helper function to process a single chunk
    def process_single_chunk(chunk):
        """Process one chunk and return results - used for parallel processing"""
        print(f"  🔍 Processing chunk {chunk['chunk_id']}/{len(chunks)} ({chunk['pages']})...")

        chunk_prompt = f"""You are a document analysis assistant for ESG reports.

IMPORTANT CONTEXT:
- You are analyzing a SECTION of a larger document (specifically: {chunk['pages']} of {filename})
- Work ONLY with the text section provided below
- If information is not in this section, state "Not found in this section"

ANSWER FORMATTING:
- Use **Bold Headers:** to separate sections (e.g., **Key Finding:**, **Evidence:**, **Gap:**)
- Keep paragraphs short (2-3 sentences)
- Use bullet points (•) for lists
- Be concise but specific

QUESTIONS TO ANALYZE:
{json.dumps(questions_json, indent=2)}

DOCUMENT SECTION ({chunk['pages']}):
{chunk['text']}

RESPONSE FORMAT - Return ONLY valid JSON:
{{
  "chunk_info": {{
    "chunk_id": {chunk['chunk_id']},
    "pages_analyzed": "{chunk['pages']}"
  }},
  "results": [
    {{
      "question_number": 1,
      "answer": "**Finding:** [Main point]\\n\\n**Evidence:**\\n• Point 1\\n• Point 2\\n\\n**Note:** [If incomplete]",
      "score": 3.5,
      "reasoning": "Brief explanation",
      "confidence": 0.8
    }}
  ]
}}

Remember: Use bold headers, bullet points, and short paragraphs in the answer. Return ONLY valid JSON."""

        try:
            if provider == 'huggingface':
                # Hugging Face uses InferenceClient
                # Reduced max_tokens to 1000 to allow more input tokens (HF limit: 32K total)
                chunk_model = config.LLM_MODELS['huggingface']['model']
                response = llm_client.chat_completion(
                    model=chunk_model,
                    messages=[{"role": "user", "content": chunk_prompt}],
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content
            elif provider in ['qwen', 'openrouter']:
                if provider == 'qwen':
                    chunk_model = config.LLM_MODELS['qwen']['model']
                else:  # openrouter
                    chunk_model = config.LLM_MODELS['openrouter']['model']
                response = llm_client.chat.completions.create(
                    model=chunk_model,
                    messages=[{"role": "user", "content": chunk_prompt}],
                    temperature=0.7
                )
                response_text = response.choices[0].message.content

                import re
                if '<think>' in response_text:
                    response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
            else:  # gemini
                response = llm_client.generate_content(chunk_prompt, request_options={"timeout": 45})
                response_text = response.text

            # Parse JSON
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned, re.DOTALL)
                if json_match:
                    cleaned = json_match.group(1)

            chunk_data = json.loads(cleaned)
            print(f"    ✅ Chunk {chunk['chunk_id']} analyzed successfully")
            return (chunk['chunk_id'], chunk_data, None)

        except Exception as e:
            print(f"    ❌ Error analyzing chunk {chunk['chunk_id']}: {e}")
            import traceback
            traceback.print_exc()

            error_result = {
                'chunk_info': {'chunk_id': chunk['chunk_id'], 'pages_analyzed': chunk['pages']},
                'results': [{
                    'question_number': i + 1,
                    'answer': f'Error processing this section: {str(e)[:100]}',
                    'score': 0.0,
                    'reasoning': 'Chunk processing failed',
                    'confidence': 0.0
                } for i in range(len(selected_questions))]
            }
            return (chunk['chunk_id'], error_result, e)

    # PARALLEL PROCESSING with ThreadPoolExecutor
    MAX_PARALLEL_CHUNKS = 3
    all_chunk_results = []
    print(f"  🚀 Parallel processing enabled ({MAX_PARALLEL_CHUNKS} chunks at a time)")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as executor:
        future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}

        completed = 0
        for future in as_completed(future_to_chunk):
            completed += 1
            chunk_id, chunk_data, error = future.result()
            all_chunk_results.append(chunk_data)

            if error:
                print(f"    ⚠️ Chunk {chunk_id} completed with errors ({completed}/{len(chunks)})")
            else:
                print(f"    📊 Progress: {completed}/{len(chunks)} chunks completed")



    # Aggregate results from all chunks
    print(f"  🔄 Aggregating results from {len(all_chunk_results)} chunks...")
    aggregated_scores = aggregate_chunk_results(all_chunk_results, selected_questions)

    return aggregated_scores


def aggregate_chunk_results(chunk_results, questions):
    """Combine results from multiple chunks into final answers"""
    aggregated = []

    # Check if we have any results at all
    if not chunk_results:
        print("  ⚠️ Warning: No chunk results to aggregate!")
        return [{
            'question_id': q['id'],
            'question_text': q['question_text'],
            'category': q.get('category'),
            'main_answer': 'Error: No chunks were successfully processed',
            'score': 0.0,
            'reasoning': 'All chunks failed to process'
        } for q in questions]

    for i, question in enumerate(questions):
        question_num = i + 1

        # Collect all answers for this question from all chunks
        chunk_answers = []
        chunk_scores = []
        chunk_reasonings = []
        error_chunks = []

        for chunk_data in chunk_results:
            results = chunk_data.get('results', [])
            for result in results:
                if result.get('question_number') == question_num:
                    answer = result.get('answer', '')

                    # Track errors separately
                    if 'error processing' in answer.lower():
                        error_chunks.append(chunk_data.get('chunk_info', {}).get('pages_analyzed', 'unknown'))
                        continue

                    # Only include valid answers (not "not found" messages)
                    if answer and 'not found' not in answer.lower():
                        chunk_answers.append({
                            'answer': answer,
                            'score': result.get('score', 0),
                            'reasoning': result.get('reasoning', ''),
                            'pages': chunk_data.get('chunk_info', {}).get('pages_analyzed', 'unknown')
                        })
                        chunk_scores.append(result.get('score', 0))
                        chunk_reasonings.append(result.get('reasoning', ''))

        # Aggregate
        if chunk_answers:
            # Combine answers
            combined_answer = "Information found across multiple sections:\n\n"
            for idx, ans in enumerate(chunk_answers, 1):
                combined_answer += f"• From {ans['pages']}: {ans['answer']}\n\n"

            # Add warning if some chunks had errors
            if error_chunks:
                combined_answer += f"\n⚠️ Note: {len(error_chunks)} section(s) failed to process: {', '.join(error_chunks[:3])}"

            # Average score (only from successful chunks)
            avg_score = sum(chunk_scores) / len(chunk_scores)
            avg_score = max(1.0, min(5.0, round(avg_score * 2) / 2))

            # Combine reasoning
            success_ratio = f"{len(chunk_answers)}/{len(chunk_results)}"
            combined_reasoning = f"Analysis from {success_ratio} chunks. " + "; ".join(chunk_reasonings)
            if error_chunks:
                combined_reasoning += f" (Warning: {len(error_chunks)} chunks failed)"

            aggregated.append({
                'question_id': question['id'],
                'question_text': question['question_text'],
                'category': question.get('category'),
                'main_answer': combined_answer.strip(),
                'score': avg_score,
                'reasoning': combined_reasoning
            })
        else:
            # No information found in any chunk
            status_msg = 'Information not found in document'
            reasoning_msg = 'No relevant information found across all document sections'

            if error_chunks:
                status_msg = f'Processing issues: {len(error_chunks)}/{len(chunk_results)} chunks failed'
                reasoning_msg = f'Failed chunks: {", ".join(error_chunks)}'

            aggregated.append({
                'question_id': question['id'],
                'question_text': question['question_text'],
                'category': question.get('category'),
                'main_answer': status_msg,
                'score': 1.0,
                'reasoning': reasoning_msg
            })

    return aggregated


def parse_json_response(response_text, questions):
    """Parse JSON response from Gemini - ROBUST VERSION WITH TRUNCATION HANDLING"""
    import re
    scores = []

    print(f"🔍 Parsing JSON response for {len(questions)} questions...")

    try:
        # Clean the response text
        cleaned = response_text.strip()

        # Remove control characters that can cause JSON parsing errors (especially from Hugging Face)
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)

        # Remove markdown code blocks if present - IMPROVED
        if '```' in cleaned:
            # Extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            else:
                # Remove all ``` markers
                cleaned = re.sub(r'```(?:json)?', '', cleaned)
                cleaned = re.sub(r'```', '', cleaned)

        # Try to extract JSON if there's extra text
        json_match = re.search(r'\{.*?"results".*?\[.*?\].*?\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        # Handle truncated JSON - try to fix incomplete strings
        if not cleaned.endswith('}'):
            # Find the last complete result object
            last_complete = cleaned.rfind('}')
            if last_complete > 0:
                # Try to close the JSON properly
                temp = cleaned[:last_complete + 1]
                if temp.count('[') > temp.count(']'):
                    temp += ']'
                if temp.count('{') > temp.count('}'):
                    temp += '}'
                cleaned = temp

        # Parse JSON
        data = json.loads(cleaned)

        # Extract results
        results = data.get('results', [])
        print(f"  ✅ Parsed JSON successfully: {len(results)} results found")

        # Map results to questions
        for i, question in enumerate(questions):
            question_num = i + 1

            # Find matching result
            result = None
            for r in results:
                if r.get('question_number') == question_num:
                    result = r
                    break

            if result:
                answer = result.get('answer', 'No answer provided')
                score = float(result.get('score', 0.0))
                reasoning = result.get('reasoning', 'No reasoning provided')

                # Validate score
                score = max(1.0, min(5.0, round(score * 2) / 2))

                print(f"  ✅ Q{question_num}: score={score}")

                scores.append({
                    'question_id': question['id'],
                    'question_text': question['question_text'],
                    'category': question.get('category'),
                    'main_answer': answer,
                    'score': score,
                    'reasoning': reasoning
                })
            else:
                print(f"  ⚠️ Q{question_num}: Not found in response")
                scores.append({
                    'question_id': question['id'],
                    'question_text': question['question_text'],
                    'category': question.get('category'),
                    'main_answer': 'No answer found in response',
                    'score': 0.0,
                    'reasoning': 'Question not answered in AI response'
                })

        return scores

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing failed: {e}")
        print(f"  🔄 Falling back to text-based parsing...")
        # Fallback to old parser
        return parse_multi_question_response(response_text, questions)

    except Exception as e:
        print(f"  ❌ Parsing error: {e}")
        import traceback
        traceback.print_exc()
        # Return error results
        return [{
            'question_id': q['id'],
            'question_text': q['question_text'],
            'category': q.get('category'),
            'main_answer': f'Error parsing response: {str(e)}',
            'score': 0.0,
            'reasoning': 'Failed to parse AI response'
        } for q in questions]


def parse_multi_question_response(response_text, questions):
    """Parse response containing multiple question answers - ROBUST VERSION"""
    import re
    scores = []

    print(f"🔍 Parsing response for {len(questions)} questions...")
    print(f"📊 Response length: {len(response_text)} characters")

    # Try multiple parsing strategies

    # Strategy 1: Split by "QUESTION X:" markers (case-insensitive, handles various formats)
    parts = re.split(r'QUESTION\s+(\d+)[\s:]*', response_text, flags=re.IGNORECASE)

    # parts will be: ['intro text', '1', 'content for Q1', '2', 'content for Q2', ...]
    question_sections = {}
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            q_num = int(parts[i])
            q_content = parts[i+1]
            question_sections[q_num] = q_content

    print(f"  📦 Found {len(question_sections)} question sections using strategy 1")

    for i, question in enumerate(questions):
        question_num = i + 1
        section = question_sections.get(question_num, '')

        # Initialize defaults
        answer = ''
        score = 0.0
        reasoning = ''

        if not section:
            print(f"  ⚠️  Question {question_num} not found in response")
            scores.append({
                'question_id': question['id'],
                'question_text': question['question_text'],
                'category': question.get('category'),
                'main_answer': 'No answer found in response',
                'score': 0.0,
                'reasoning': 'Could not locate this question in the AI response'
            })
            continue

        # Extract ANSWER (try multiple patterns)
        answer_patterns = [
            r'ANSWER:\s*(.+?)(?=SCORE:|REASONING:|QUESTION|\Z)',
            r'Answer:\s*(.+?)(?=Score:|Reasoning:|Question|\Z)',
            r'answer:\s*(.+?)(?=score:|reasoning:|question|\Z)',
            r'\*\*Answer:\*\*\s*(.+?)(?=\*\*Score:|\*\*Reasoning:|\Z)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up answer (remove extra whitespace)
                answer = ' '.join(answer.split())
                break

        # Extract SCORE (try multiple patterns)
        score_patterns = [
            r'SCORE:\s*(\d+\.?\d*)',
            r'Score:\s*(\d+\.?\d*)',
            r'score:\s*(\d+\.?\d*)',
            r'\*\*Score:\*\*\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*5',  # Matches "4.5/5" format
        ]

        for pattern in score_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Round to nearest 0.5
                    score = round(score * 2) / 2
                    # Clamp to 1.0-5.0
                    score = max(1.0, min(5.0, score))
                    break
                except (ValueError, IndexError):
                    continue

        # Extract REASONING (try multiple patterns)
        reasoning_patterns = [
            r'REASONING:\s*(.+?)(?=QUESTION|\Z)',
            r'Reasoning:\s*(.+?)(?=Question|\Z)',
            r'reasoning:\s*(.+?)(?=question|\Z)',
            r'\*\*Reasoning:\*\*\s*(.+?)(?=\*\*Question:|\Z)',
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                # Clean up reasoning
                reasoning = ' '.join(reasoning.split())
                break

        # Fallback: if no structured data found, try to extract something useful
        if not answer and not score and not reasoning:
            # Just take the whole section as the answer
            answer = ' '.join(section.split())[:500]  # First 500 chars
            reasoning = "Could not parse structured response, showing raw content"
            score = 2.5  # Default middle score

        print(f"  ✅ Q{question_num}: answer={len(answer)} chars, score={score}, reasoning={len(reasoning)} chars")

        scores.append({
            'question_id': question['id'],
            'question_text': question['question_text'],
            'category': question.get('category'),
            'main_answer': answer or 'No answer found',
            'score': score,
            'reasoning': reasoning or 'No reasoning provided'
        })

    return scores


@app.route('/api/chat-with-docs', methods=['POST'])
def chat_with_docs():
    """Chat with Qwen3-30B about analyzed documents - WITH CONVERSATION MEMORY"""
    try:
        from openai import OpenAI

        data = request.get_json()
        user_message = data.get('message')
        documents = data.get('documents', [])
        chat_history = data.get('chat_history', [])  # NEW: Get conversation history

        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400

        # Configure OpenRouter with Qwen3-30B
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),
            default_headers={
                "HTTP-Referer": "https://web-production-2ea5.up.railway.app",
                "X-Title": "ESG Retrieval & Evaluation"
            }
        )
        model_name = 'qwen/qwen3-30b-a3b:free'

        # Build context from analyzed documents
        context = "ANALYZED DOCUMENTS:\n\n"
        for doc in documents:
            context += f"**Document: {doc['filename']}**\n"
            if 'scores' in doc:
                for score in doc['scores']:
                    context += f"- **Q:** {score['question_text']}\n"
                    context += f"  **Score:** {score['score']}/5.0\n"
                    context += f"  **Answer:** {score['main_answer'][:200]}...\n"
            context += "\n"

        # Build conversation history
        conversation_context = ""
        if chat_history:
            conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
            for msg in chat_history[-6:]:  # Last 6 messages (3 exchanges)
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_context += f"{role.upper()}: {content}\n"
            conversation_context += "\n"

        # Create chat prompt with memory and formatting instructions
        chat_prompt = f"""{context}{conversation_context}
USER QUESTION: {user_message}

RESPONSE INSTRUCTIONS:
1. Use **Bold Headers** to organize your response into sections
2. Break complex responses into clearly labeled sections
3. Use bullet points (•) for lists
4. Reference specific documents, scores, and previous conversation when relevant
5. If referring to previous scores or analysis, explicitly mention them

FORMAT EXAMPLE:
**Summary:** [Main point]

**Key Details:**
• Point 1
• Point 2

**Recommendation:** [If applicable]

Provide a helpful, well-formatted answer based on the analyzed documents and conversation history above."""

        # Get response from Qwen3-30B
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": chat_prompt}],
            max_tokens=2500,
            temperature=0.7
        )

        return jsonify({
            'success': True,
            'response': response.choices[0].message.content
        })

    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Make sure database is initialized
    db.init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)
