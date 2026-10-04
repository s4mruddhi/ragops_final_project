"""
app/validation/text_cleaner.py
================================
Cleans extracted text before chunking.

Operations (in order):
  1. Strip leading/trailing whitespace
  2. Normalise unicode characters (NFKC)
  3. Replace multiple consecutive blank lines with a single blank line
  4. Replace non-breaking spaces and other whitespace variants with regular space
  5. Remove form-feed (\f) and other control characters (keep \n, \t)
  6. Collapse multiple spaces on a single line to one space
  7. Remove lines that are pure horizontal rules (----, ====, ****) > 4 chars
  8. Strip trailing whitespace from each line

The cleaner does NOT:
  - Remove sentences or semantic content
  - Lowercase text (embeddings handle case sensitivity)
  - Remove punctuation
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans extracted document text.

    Usage:
        cleaner = TextCleaner()
        clean = cleaner.clean(raw_text)
    """

    # Patterns compiled once for efficiency
    _MULTI_BLANK_LINES = re.compile(r"\n{3,}")
    _MULTI_SPACES = re.compile(r"[ \t]{2,}")
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _HORIZONTAL_RULE = re.compile(r"^[-=*#_]{4,}$")
    _TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to text."""
        if not text:
            return ""

        # 1. NFKC unicode normalisation
        text = unicodedata.normalize("NFKC", text)

        # 2. Replace non-breaking spaces and zero-width chars
        text = text.replace("\u00a0", " ")   # non-breaking space
        text = text.replace("\u200b", "")    # zero-width space
        text = text.replace("\u200c", "")    # zero-width non-joiner
        text = text.replace("\ufeff", "")    # BOM

        # 3. Remove control characters (keep \n and \t)
        text = self._CONTROL_CHARS.sub("", text)

        # 4. Strip trailing whitespace per line
        text = self._TRAILING_SPACE.sub("", text)

        # 5. Remove pure horizontal-rule lines
        lines = text.split("\n")
        lines = [
            line for line in lines
            if not self._HORIZONTAL_RULE.match(line.strip())
        ]
        text = "\n".join(lines)

        # 6. Collapse multiple spaces on a single line
        text = self._MULTI_SPACES.sub(" ", text)

        # 7. Collapse 3+ consecutive blank lines → 2
        text = self._MULTI_BLANK_LINES.sub("\n\n", text)

        # 8. Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean_batch(self, extracted_docs: list[dict]) -> list[dict]:
        """
        Clean text in a batch of extracted document dicts.
        Each dict must have a "text" key.
        Returns the same list with "text" replaced by cleaned version
        and "cleaned_char_count" added.
        """
        results = []
        for doc in extracted_docs:
            raw = doc.get("text", "")
            cleaned = self.clean(raw)
            if not cleaned:
                logger.warning(
                    f"Empty text after cleaning: {doc.get('document_id', 'unknown')}"
                )
                continue
            doc_out = dict(doc)
            doc_out["text"] = cleaned
            doc_out["cleaned_char_count"] = len(cleaned)
            doc_out["original_char_count"] = len(raw)
            results.append(doc_out)
            logger.debug(
                f"Cleaned {doc.get('document_id', '?')}: "
                f"{len(raw)} -> {len(cleaned)} chars"
            )
        return results
