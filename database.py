"""
Database models for ESG Scoring Platform
"""

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DATABASE_PATH = 'esg_platform.db'


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database with all tables"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Companies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ticker TEXT UNIQUE,
                sector TEXT,
                country TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Reports table - stores uploaded ESG/sustainability reports
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                report_year INTEGER NOT NULL,
                report_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_id TEXT UNIQUE NOT NULL,
                file_path TEXT,
                total_pages INTEGER,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id),
                UNIQUE(company_id, report_year, report_type)
            )
        ''')

        # ESG Scores table - stores calculated scores
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS esg_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                framework_version TEXT NOT NULL,

                -- Overall scores
                overall_score REAL,
                environmental_score REAL,
                social_score REAL,
                governance_score REAL,

                -- Detailed metrics (stored as JSON)
                detailed_scores TEXT,
                extracted_data TEXT,
                ai_reasoning TEXT,

                -- Metadata
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scored_by TEXT,

                FOREIGN KEY (report_id) REFERENCES reports (id)
            )
        ''')

        # ESG Framework table - stores scoring criteria
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS esg_framework (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                criteria TEXT NOT NULL,  -- JSON structure of scoring criteria
                weights TEXT NOT NULL,   -- JSON structure of weights
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Questionnaire Templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questionnaire_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                version TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Questions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_id INTEGER NOT NULL,
                question_number TEXT,
                question_text TEXT NOT NULL,
                category TEXT,
                pillar TEXT,
                question_type TEXT DEFAULT 'text',
                options TEXT,
                weight REAL DEFAULT 1.0,
                requires_evidence BOOLEAN DEFAULT 1,
                scoring_guidance TEXT,
                display_order INTEGER,
                FOREIGN KEY (template_id) REFERENCES questionnaire_templates (id)
            )
        ''')

        # Document Library - stores all documents for a company
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                document_type TEXT NOT NULL,
                document_year INTEGER,
                filename TEXT NOT NULL,
                upload_id TEXT UNIQUE NOT NULL,
                file_path TEXT,
                total_pages INTEGER,
                description TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        ''')

        # Evaluations table - links reports to questionnaire completion
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                company_id INTEGER NOT NULL,
                template_id INTEGER NOT NULL,
                evaluation_year INTEGER,
                status TEXT DEFAULT 'in_progress',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                completed_by TEXT,
                notes TEXT,
                FOREIGN KEY (report_id) REFERENCES reports (id),
                FOREIGN KEY (company_id) REFERENCES companies (id),
                FOREIGN KEY (template_id) REFERENCES questionnaire_templates (id)
            )
        ''')

        # Evaluation Documents - junction table linking evaluations to multiple documents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id),
                FOREIGN KEY (document_id) REFERENCES documents (id),
                UNIQUE(evaluation_id, document_id)
            )
        ''')

        # Answers table - stores answers to questions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                answer_text TEXT,
                score REAL,
                confidence REAL,
                evidence TEXT,
                page_references TEXT,
                source_documents TEXT,
                ai_generated BOOLEAN DEFAULT 1,
                human_verified BOOLEAN DEFAULT 0,
                human_edited BOOLEAN DEFAULT 0,
                notes TEXT,
                answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id),
                FOREIGN KEY (question_id) REFERENCES questions (id)
            )
        ''')

        # Chat history for evaluations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                question_id INTEGER,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id),
                FOREIGN KEY (question_id) REFERENCES questions (id)
            )
        ''')

        # Scoring Questions Library - customizable questions with rubrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scoring_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL,
                sub_questions TEXT NOT NULL,
                rubric TEXT NOT NULL,
                category TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Scoring Results - stores results from applying scoring questions to documents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scoring_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                company_id INTEGER,
                scoring_question_id INTEGER NOT NULL,
                main_answer TEXT,
                score REAL,
                reasoning TEXT,
                evidence TEXT,
                page_references TEXT,
                ai_generated BOOLEAN DEFAULT 1,
                human_verified BOOLEAN DEFAULT 0,
                notes TEXT,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id),
                FOREIGN KEY (company_id) REFERENCES companies (id),
                FOREIGN KEY (scoring_question_id) REFERENCES scoring_questions (id)
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reports_company ON reports(company_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reports_year ON reports(report_year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scores_report ON esg_scores(report_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_template ON questions(template_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_answers_evaluation ON answers(evaluation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_report ON evaluations(report_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluation_docs ON evaluation_documents(evaluation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scoring_results_document ON scoring_results(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scoring_results_company ON scoring_results(company_id)')

        print("✅ Database initialized successfully")


def add_company(name, ticker=None, sector=None, country=None):
    """Add a new company"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO companies (name, ticker, sector, country)
            VALUES (?, ?, ?, ?)
        ''', (name, ticker, sector, country))
        return cursor.lastrowid


def get_all_companies():
    """Get all companies"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.*,
                   COUNT(DISTINCT r.id) as report_count,
                   MAX(r.report_year) as latest_year
            FROM companies c
            LEFT JOIN reports r ON c.id = r.company_id
            GROUP BY c.id
            ORDER BY c.name
        ''')
        return [dict(row) for row in cursor.fetchall()]


def get_company(company_id):
    """Get a single company with all reports"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM companies WHERE id = ?', (company_id,))
        company = dict(cursor.fetchone()) if cursor.fetchone() else None

        if company:
            cursor.execute('''
                SELECT r.*,
                       s.overall_score,
                       s.environmental_score,
                       s.social_score,
                       s.governance_score
                FROM reports r
                LEFT JOIN esg_scores s ON r.id = s.report_id
                WHERE r.company_id = ?
                ORDER BY r.report_year DESC
            ''', (company_id,))
            company['reports'] = [dict(row) for row in cursor.fetchall()]

        return company


def add_report(company_id, report_year, report_type, filename, upload_id, file_path, total_pages):
    """Add a new report"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reports (company_id, report_year, report_type, filename, upload_id, file_path, total_pages)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (company_id, report_year, report_type, filename, upload_id, file_path, total_pages))
        return cursor.lastrowid


def save_esg_score(report_id, framework_version, overall_score, env_score, social_score, gov_score,
                   detailed_scores, extracted_data, ai_reasoning, scored_by='AI'):
    """Save ESG scores for a report"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO esg_scores
            (report_id, framework_version, overall_score, environmental_score, social_score,
             governance_score, detailed_scores, extracted_data, ai_reasoning, scored_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (report_id, framework_version, overall_score, env_score, social_score, gov_score,
              json.dumps(detailed_scores), json.dumps(extracted_data), ai_reasoning, scored_by))
        return cursor.lastrowid


def get_company_history(company_id):
    """Get all reports and scores for a company across years"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.report_year,
                   r.report_type,
                   r.filename,
                   s.overall_score,
                   s.environmental_score,
                   s.social_score,
                   s.governance_score,
                   s.scored_at
            FROM reports r
            LEFT JOIN esg_scores s ON r.id = s.report_id
            WHERE r.company_id = ?
            ORDER BY r.report_year DESC
        ''', (company_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_active_framework():
    """Get the currently active ESG framework"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM esg_framework WHERE is_active = 1 LIMIT 1')
        row = cursor.fetchone()
        if row:
            framework = dict(row)
            framework['criteria'] = json.loads(framework['criteria'])
            framework['weights'] = json.loads(framework['weights'])
            return framework
        return None


def save_framework(version, name, description, criteria, weights):
    """Save a new ESG framework"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO esg_framework (version, name, description, criteria, weights, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
        ''', (version, name, description, json.dumps(criteria), json.dumps(weights)))

        # Deactivate other frameworks
        cursor.execute('UPDATE esg_framework SET is_active = 0 WHERE version != ?', (version,))
        return cursor.lastrowid


# Questionnaire functions

def create_questionnaire_template(name, description, category, version):
    """Create a new questionnaire template"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO questionnaire_templates (name, description, category, version)
            VALUES (?, ?, ?, ?)
        ''', (name, description, category, version))
        return cursor.lastrowid


def add_question(template_id, question_text, question_number=None, category=None, pillar=None,
                 question_type='text', weight=1.0, scoring_guidance=None, display_order=None):
    """Add a question to a template"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO questions
            (template_id, question_number, question_text, category, pillar, question_type,
             weight, scoring_guidance, display_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (template_id, question_number, question_text, category, pillar, question_type,
              weight, scoring_guidance, display_order))
        return cursor.lastrowid


def get_active_template():
    """Get the active questionnaire template"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM questionnaire_templates WHERE is_active = 1 LIMIT 1')
        row = cursor.fetchone()
        return dict(row) if row else None


def get_template_questions(template_id):
    """Get all questions for a template"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM questions
            WHERE template_id = ?
            ORDER BY display_order, id
        ''', (template_id,))
        return [dict(row) for row in cursor.fetchall()]


def create_evaluation(report_id, template_id):
    """Create a new evaluation session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO evaluations (report_id, template_id, status)
            VALUES (?, ?, 'in_progress')
        ''', (report_id, template_id))
        return cursor.lastrowid


def save_answer(evaluation_id, question_id, answer_text, score=None, confidence=None,
                evidence=None, page_references=None, source_documents=None, ai_generated=True):
    """Save an answer to a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO answers
            (evaluation_id, question_id, answer_text, score, confidence, evidence,
             page_references, source_documents, ai_generated, human_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ''', (evaluation_id, question_id, answer_text, score, confidence, evidence,
              page_references, source_documents, ai_generated))
        return cursor.lastrowid


def get_evaluation_answers(evaluation_id):
    """Get all answers for an evaluation"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.*, q.question_text, q.question_number, q.category, q.pillar
            FROM answers a
            JOIN questions q ON a.question_id = q.id
            WHERE a.evaluation_id = ?
            ORDER BY q.display_order, q.id
        ''', (evaluation_id,))
        return [dict(row) for row in cursor.fetchall()]


def update_answer(answer_id, answer_text, score=None, human_verified=True, notes=None):
    """Update an answer (human edit)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE answers
            SET answer_text = ?, score = ?, human_verified = ?, human_edited = 1, notes = ?
            WHERE id = ?
        ''', (answer_text, score, human_verified, notes, answer_id))


def save_chat_message(evaluation_id, user_message, ai_response, question_id=None):
    """Save a chat message for an evaluation"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO evaluation_chat (evaluation_id, question_id, user_message, ai_response)
            VALUES (?, ?, ?, ?)
        ''', (evaluation_id, question_id, user_message, ai_response))
        return cursor.lastrowid


def get_evaluation_chat(evaluation_id):
    """Get chat history for an evaluation"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM evaluation_chat
            WHERE evaluation_id = ?
            ORDER BY created_at
        ''', (evaluation_id,))
        return [dict(row) for row in cursor.fetchall()]


def complete_evaluation(evaluation_id, completed_by='User'):
    """Mark an evaluation as complete"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE evaluations
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP, completed_by = ?
            WHERE id = ?
        ''', (completed_by, evaluation_id))


# Document Library functions

def add_document(company_id, document_type, filename, upload_id, file_path, total_pages, document_year=None, description=None):
    """Add a document to the company's document library"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (company_id, document_type, document_year, filename, upload_id, file_path, total_pages, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (company_id, document_type, document_year, filename, upload_id, file_path, total_pages, description))
        return cursor.lastrowid


def get_company_documents(company_id):
    """Get all documents for a company"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM documents
            WHERE company_id = ?
            ORDER BY document_year DESC, uploaded_at DESC
        ''', (company_id,))
        return [dict(row) for row in cursor.fetchall()]


def delete_document(document_id):
    """Delete a document from the library"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))


def create_evaluation_with_documents(company_id, template_id, document_ids, evaluation_year=None):
    """Create an evaluation linked to multiple documents"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Create evaluation (report_id can be null for document library based evaluations)
        cursor.execute('''
            INSERT INTO evaluations (report_id, company_id, template_id, evaluation_year, status)
            VALUES (NULL, ?, ?, ?, 'in_progress')
        ''', (company_id, template_id, evaluation_year))
        evaluation_id = cursor.lastrowid

        # Link documents to evaluation
        for doc_id in document_ids:
            cursor.execute('''
                INSERT INTO evaluation_documents (evaluation_id, document_id)
                VALUES (?, ?)
            ''', (evaluation_id, doc_id))

        return evaluation_id


def get_evaluation_documents(evaluation_id):
    """Get all documents linked to an evaluation"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.* FROM documents d
            JOIN evaluation_documents ed ON d.id = ed.document_id
            WHERE ed.evaluation_id = ?
        ''', (evaluation_id,))
        return [dict(row) for row in cursor.fetchall()]


# Scoring Questions Library functions

def add_scoring_question(question_text, sub_questions, rubric, category=None):
    """Add a new scoring question to the library"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scoring_questions (question_text, sub_questions, rubric, category)
            VALUES (?, ?, ?, ?)
        ''', (question_text, sub_questions, rubric, category))
        return cursor.lastrowid


def get_all_scoring_questions(active_only=True):
    """Get all scoring questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        if active_only:
            cursor.execute('SELECT * FROM scoring_questions WHERE is_active = 1 ORDER BY id')
        else:
            cursor.execute('SELECT * FROM scoring_questions ORDER BY id')
        return [dict(row) for row in cursor.fetchall()]


def get_scoring_question(question_id):
    """Get a single scoring question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM scoring_questions WHERE id = ?', (question_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_scoring_question(question_id, question_text, sub_questions, rubric, category=None):
    """Update a scoring question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE scoring_questions
            SET question_text = ?, sub_questions = ?, rubric = ?, category = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (question_text, sub_questions, rubric, category, question_id))


def delete_scoring_question(question_id):
    """Delete a scoring question (soft delete - mark as inactive)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE scoring_questions SET is_active = 0 WHERE id = ?', (question_id,))


def save_scoring_result(document_id, company_id, scoring_question_id, main_answer,
                       score, reasoning, evidence=None, page_references=None):
    """Save a scoring result"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scoring_results
            (document_id, company_id, scoring_question_id, main_answer, score,
             reasoning, evidence, page_references, ai_generated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', (document_id, company_id, scoring_question_id, main_answer, score,
              reasoning, evidence, page_references))
        return cursor.lastrowid


def get_document_scoring_results(document_id):
    """Get all scoring results for a document"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sr.*, sq.question_text, sq.category
            FROM scoring_results sr
            JOIN scoring_questions sq ON sr.scoring_question_id = sq.id
            WHERE sr.document_id = ?
            ORDER BY sr.scored_at DESC
        ''', (document_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_company_scoring_results(company_id):
    """Get all scoring results for a company across all documents"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sr.*, sq.question_text, sq.category, d.filename, d.document_type
            FROM scoring_results sr
            JOIN scoring_questions sq ON sr.scoring_question_id = sq.id
            LEFT JOIN documents d ON sr.document_id = d.id
            WHERE sr.company_id = ?
            ORDER BY sr.scored_at DESC
        ''', (company_id,))
        return [dict(row) for row in cursor.fetchall()]


if __name__ == '__main__':
    init_db()
