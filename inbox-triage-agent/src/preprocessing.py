"""
Text Preprocessing Module for Inbox Triage Agent.
Cleans and standardizes email text (subject and body) using NLTK and regex.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK data dependencies are loaded safely
try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    try:
        _STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        _STOP_WORDS = set([
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
            "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
            "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
            "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
            "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
            "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
            "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
            "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
            "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
            "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
            "yourselves"
        ])

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

try:
    _STEMMER = PorterStemmer()
except Exception:
    _STEMMER = None


def clean_text(text: str, remove_stopwords: bool = True, stem: bool = True) -> str:
    """
    Cleans raw email text:
      1. Converts to lowercase
      2. Removes HTML tags and email headers
      3. Normalizes URLs to 'http_url' token and emails to 'email_addr' token
      4. Strips punctuation and digits
      5. Tokenizes via NLTK
      6. Removes stopwords
      7. Applies Porter Stemming
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Remove HTML tags if present
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # 3. Normalize URLs and Email addresses
    cleaned = re.sub(r"https?://\S+|www\.\S+", "url_token", cleaned)
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_token", cleaned)

    # 4. Remove special characters and digits, retaining letters and whitespace
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)

    # 5. Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        # Fallback regex tokenizer if NLTK tokenizer fails
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", cleaned)

    # 6. Stopwords removal and length filtering
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in _STOP_WORDS and len(tok) > 1]
    else:
        tokens = [tok for tok in tokens if len(tok) > 1]

    # 7. Stemming
    if stem and _STEMMER is not None:
        tokens = [_STEMMER.stem(tok) for tok in tokens]

    return " ".join(tokens)


def extract_url_count(text: str) -> int:
    """Counts URLs in the email text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE))


def extract_email_address_count(text: str) -> int:
    """Counts embedded email addresses in the email text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text))
