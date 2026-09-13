"""
Anonymizer module for text and PDF files before training a machine learning model.
 This module provides functions to anonymize sensitive information in text and PDF files, ensuring that personal data is protected during the training process.
"""

from __future__ import annotations

import re

_CNPJ_ = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
_CPF_ = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")

_MONEY_ = re.compile(r"(R\$\s*)?\d{1,3}(?:\.\d{3})*,\d{2}")
_DATE_ = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

def anonymize(text: str) -> str:
    """
    Anonymizes sensitive information in the given text.

    Args:
        text (str): The input text to be anonymized.
    Returns:
        str: The anonymized text.
    """
    text = _CNPJ_.sub("[CNPJ]", text)
    text = _CPF_.sub("[CPF]", text)
    text = _MONEY_.sub("[MONEY]", text)
    text = _DATE_.sub("[DATE]", text)
    return text