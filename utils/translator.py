# utils/translator.py  (add this to your project)

from deep_translator import GoogleTranslator
from langdetect import detect
from config import PRESERVE_TERMS

def translate_query_to_english(query: str) -> tuple[str, str]:
    """
    Detects language and translates to English if needed.
    Returns (translated_query, original_language_code)
    
    PRESERVE_TERMS like 'IPC', 'FIR', 'RTI' are never translated.
    """
    lang = detect(query)          # 'hi', 'en', 'mr', etc.
    
    if lang == 'en':
        return query, 'en'
    
    # Mask legal terms before translation
    masked, mapping = _mask_terms(query)
    translated = GoogleTranslator(source='auto', target='en').translate(masked)
    restored = _restore_terms(translated, mapping)
    
    return restored, lang


def _mask_terms(text: str) -> tuple[str, dict]:
    """Replace legal terms with placeholders so translator ignores them."""
    mapping = {}
    for i, term in enumerate(PRESERVE_TERMS):
        placeholder = f"__TERM{i}__"
        if term in text:
            mapping[placeholder] = term
            text = text.replace(term, placeholder)
    return text, mapping


def _restore_terms(text: str, mapping: dict) -> str:
    for placeholder, term in mapping.items():
        text = text.replace(placeholder, term)
    return text