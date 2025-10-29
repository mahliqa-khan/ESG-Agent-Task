#!/usr/bin/env python3
"""
Convert PDF OCR JSON output to a nicely formatted HTML page
Usage: python json_to_html.py <input_json> <output_html>
"""

import sys
import json
import re
import markdown2


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


def clean_ocr_tags(text):
    """Remove OCR reference and detection tags from markdown"""
    # Remove <|ref|>...<|/ref|> tags
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    # Remove <|det|>...<|/det|> tags
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\n\n\n+', '\n\n', text)

    # Fix LaTeX syntax
    text = fix_latex_syntax(text)

    return text.strip()


def get_html_template():
    """Return the HTML template without f-string processing"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Document</title>

    <!-- MathJax for rendering equations -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .metadata {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.9;
        }

        .metadata-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .page-nav {
            background-color: #f8f9fa;
            padding: 15px 30px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .page-nav select {
            padding: 8px 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
            background-color: white;
        }

        .page-nav button {
            padding: 8px 20px;
            border: 1px solid #667eea;
            background-color: #667eea;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }

        .page-nav button:hover {
            background-color: #5568d3;
        }

        .page-nav button:disabled {
            background-color: #ccc;
            border-color: #ccc;
            cursor: not-allowed;
        }

        .content {
            padding: 40px;
        }

        .page {
            display: none;
            animation: fadeIn 0.3s;
        }

        .page.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .page-number {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 600;
        }

        h1 {
            font-size: 2em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        h2 {
            font-size: 1.6em;
            color: #667eea;
        }

        p {
            margin-bottom: 15px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        table thead tr {
            background-color: #667eea;
            color: white;
            text-align: left;
        }

        table th,
        table td {
            padding: 12px 15px;
            border: 1px solid #ddd;
        }

        table tbody tr {
            border-bottom: 1px solid #ddd;
        }

        table tbody tr:nth-of-type(even) {
            background-color: #f8f9fa;
        }

        table tbody tr:hover {
            background-color: #e9ecef;
        }

        strong {
            color: #2c3e50;
            font-weight: 600;
        }

        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }

        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
        }

        blockquote {
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
        }

        /* Math equation styling */
        .MathJax {
            font-size: 1.1em !important;
        }

        mjx-container {
            overflow-x: auto;
            overflow-y: hidden;
        }

        /* Display math (block equations) */
        mjx-container[display="true"] {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .footer {
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }

        @media print {
            body {
                background-color: white;
                padding: 0;
            }

            .container {
                box-shadow: none;
            }

            .page-nav {
                display: none;
            }

            .page {
                display: block !important;
                page-break-after: always;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 PDF Document</h1>
            <div class="metadata">
                <div class="metadata-item">
                    <span>📑</span>
                    <span>%%TOTAL_PAGES%% pages</span>
                </div>
                <div class="metadata-item">
                    <span>⏱️</span>
                    <span>Processed in %%PROCESSING_TIME%%s</span>
                </div>
            </div>
        </div>

        <div class="page-nav">
            <button id="prevBtn" onclick="changePage(-1)">← Previous</button>
            <div>
                <label for="pageSelect">Jump to page: </label>
                <select id="pageSelect" onchange="goToPage(this.value)">
                    %%PAGE_OPTIONS%%
                </select>
            </div>
            <button id="nextBtn" onclick="changePage(1)">Next →</button>
        </div>

        <div class="content">
            %%PAGES_HTML%%
        </div>

        <div class="footer">
            Generated from PDF OCR API • %%TOTAL_PAGES%% pages • %%PROCESSING_TIME%%s processing time
        </div>
    </div>

    <script>
        let currentPage = 1;
        const totalPages = %%TOTAL_PAGES%%;

        function showPage(pageNum) {
            // Hide all pages
            document.querySelectorAll('.page').forEach(page => {
                page.classList.remove('active');
            });

            // Show selected page
            const page = document.getElementById('page-' + pageNum);
            if (page) {
                page.classList.add('active');
                currentPage = pageNum;

                // Update navigation
                document.getElementById('prevBtn').disabled = (currentPage === 1);
                document.getElementById('nextBtn').disabled = (currentPage === totalPages);
                document.getElementById('pageSelect').value = currentPage;

                // Scroll to top
                window.scrollTo(0, 0);
            }
        }

        function changePage(delta) {
            const newPage = currentPage + delta;
            if (newPage >= 1 && newPage <= totalPages) {
                showPage(newPage);
            }
        }

        function goToPage(pageNum) {
            showPage(parseInt(pageNum));
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                changePage(-1);
            } else if (e.key === 'ArrowRight') {
                changePage(1);
            }
        });

        // Show first page on load
        showPage(1);
    </script>
</body>
</html>'''


def convert_to_html(json_path, html_path):
    """Convert JSON output to formatted HTML"""

    # Read JSON file
    print(f"📖 Reading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data.get('success'):
        print("❌ JSON indicates processing was not successful")
        return

    pages = data.get('pages', [])
    total_pages = data.get('total_pages', len(pages))
    processing_time = data.get('processing_time', 0)

    print(f"📄 Processing {total_pages} pages...")

    # Generate page options for dropdown
    page_options = '\n'.join([f'<option value="{i+1}">Page {i+1}</option>'
                              for i in range(total_pages)])

    # Process each page
    pages_html = []
    for page_data in pages:
        page_num = page_data.get('page', 0)
        markdown_content = page_data.get('markdown', '')

        # Clean OCR tags
        clean_markdown = clean_ocr_tags(markdown_content)

        # IMPORTANT: Protect math content from markdown processing
        # Store math expressions with placeholders
        math_expressions = []
        def store_math_expr(match):
            math_expressions.append(match.group(0))
            # Use a placeholder unlikely to appear in text
            # Format: XMATHX + index + XENDX (X won't be touched by markdown)
            return f'XMATHX{len(math_expressions)-1}XENDX'

        # Protect display math $$...$$ first (non-greedy to get individual equations)
        protected_markdown = re.sub(r'\$\$.+?\$\$', store_math_expr, clean_markdown, flags=re.DOTALL)
        # Then protect inline math $...$
        protected_markdown = re.sub(r'\$[^\$\n]+?\$', store_math_expr, protected_markdown)

        # Convert markdown to HTML
        html_content = markdown2.markdown(protected_markdown, extras=[
            'tables',
            'fenced-code-blocks',
            'header-ids',
            'break-on-newline'
        ])

        # Restore math expressions
        for i, expr in enumerate(math_expressions):
            html_content = html_content.replace(f'XMATHX{i}XENDX', expr)

        # Wrap in page div
        page_html = f'''
        <div id="page-{page_num}" class="page">
            <div class="page-number">Page {page_num} of {total_pages}</div>
            {html_content}
        </div>
        '''
        pages_html.append(page_html)

    # Get template and replace placeholders
    html_template = get_html_template()
    html_template = html_template.replace('%%TOTAL_PAGES%%', str(total_pages))
    html_template = html_template.replace('%%PROCESSING_TIME%%', f'{processing_time:.2f}')
    html_template = html_template.replace('%%PAGE_OPTIONS%%', page_options)
    html_template = html_template.replace('%%PAGES_HTML%%', '\n'.join(pages_html))

    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"✅ HTML page created: {html_path}")
    print(f"🌐 Open in browser: file://{html_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_html.py <input_json> [output_html]")
        print("Example: python json_to_html.py my_output.json my_output.html")
        sys.exit(1)

    input_json = sys.argv[1]
    output_html = sys.argv[2] if len(sys.argv) > 2 else input_json.replace('.json', '.html')

    convert_to_html(input_json, output_html)
