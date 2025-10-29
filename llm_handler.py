"""
LLM Handler - Abstraction layer for different LLM providers
"""

import config
import google.generativeai as genai
from openai import OpenAI


class LLMHandler:
    """Unified interface for different LLM providers"""

    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self.model_config = config.CURRENT_MODEL_CONFIG
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == 'gemini':
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set. Please set it in environment variables.")
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.client = genai.GenerativeModel(self.model_config['model'])

        elif self.provider == 'qwen':
            # Qwen model via OpenAI-compatible API (no auth required for this endpoint)
            self.client = OpenAI(
                base_url=config.QWEN_API_URL,
                api_key="not-needed",  # This endpoint doesn't require auth
            )

        elif self.provider == 'openrouter':
            if not config.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY not set. Please set it in environment variables.")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

        elif self.provider == 'openai':
            # OpenAI setup (can be added later)
            raise NotImplementedError("OpenAI support coming soon!")

        elif self.provider == 'anthropic':
            # Anthropic setup (can be added later)
            raise NotImplementedError("Anthropic support coming soon!")

        elif self.provider == 'groq':
            # Groq setup (can be added later)
            raise NotImplementedError("Groq support coming soon!")

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def generate_summary(self, document_text):
        """Generate a TL;DR summary of the document"""

        prompt = f"""You are a document summarization expert. Please provide a concise TL;DR summary of the following document.

Focus on:
- Main topics and themes
- Key findings or results
- Important numbers or metrics
- Critical dates or events

Keep the summary to 3-5 bullet points, each being 1-2 sentences max.

Document:
{document_text[:15000]}

Provide ONLY the bullet points, starting each with a dash (-). Be concise and specific."""

        try:
            if self.provider == 'gemini':
                response = self.client.generate_content(prompt)
                return response.text.strip()

            elif self.provider in ['openrouter', 'qwen']:
                response = self.client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config['temperature']
                )
                return response.choices[0].message.content.strip()

            else:
                return "Summary generation not available for this provider yet."
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def chat(self, document_text, user_question, chat_history=None):
        """Answer questions about the document"""

        context = f"""You are a helpful AI assistant that answers questions about documents.
You have access to the following document:

{document_text[:15000]}

Answer the user's question based on the document content. Be specific and cite relevant information from the document."""

        if chat_history:
            context += "\n\nPrevious conversation:\n" + "\n".join(
                [f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history]
            )

        prompt = f"""{context}

User Question: {user_question}

Please provide a clear, helpful answer based on the document content."""

        try:
            if self.provider == 'gemini':
                response = self.client.generate_content(prompt)
                return response.text.strip()

            elif self.provider in ['openrouter', 'qwen']:
                response = self.client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=self.model_config['temperature']
                )
                return response.choices[0].message.content.strip()

            else:
                return "Chat not available for this provider yet."
        except Exception as e:
            return f"Error: {str(e)}"


# Global instance
llm = None


def get_llm():
    """Get or create LLM handler instance"""
    global llm
    if llm is None:
        llm = LLMHandler()
    return llm
