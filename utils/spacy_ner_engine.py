'''
 Responsibilities:

    Load a spaCy model (e.g., en_core_web_trf or en_core_web_sm)

    Accept plain OCR text (as string)

    Extract named entities (person, address, date, money, etc.)

    Return entities in structured format for downstream mapping

Sample Output

 [
  {"text": "Paula M. Byrens", "label": "PERSON"},
  {"text": "March 27, 2003", "label": "DATE"},
  {"text": "$698.50", "label": "MONEY"},
  {"text": "Walnut Creek", "label": "GPE"},
]
'''
# ner/spacy_ner_engine.py

import spacy
from typing import List, Dict, Tuple

class SpacyNEREngine:
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the spaCy NER model.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(f"spaCy model '{model}' not found. Run: python -m spacy download {model}")

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Run NER on input text and extract entities.

        Parameters:
            text (str): The input text (typically combined OCR output)

        Returns:
            List of dictionaries with entity data:
            [{'text': 'Paula M. Byrens', 'label': 'PERSON'}, ...]
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_
            })
        return entities
