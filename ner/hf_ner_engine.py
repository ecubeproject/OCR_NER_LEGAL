'''
 Responsibilities:

    Load a Hugging face model (e.g., dslim/bert-base-NER or nlpaueb/legal-bert-base-uncased)

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

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class HuggingFaceNEREngine:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple"
        )

    def extract_entities(self, text):
        raw_entities = self.pipe(text)
        return self._normalize_entities(raw_entities)

    def _normalize_entities(self, entities):
        """
        Convert Hugging Face pipeline output to:
        [{"text": ..., "label": ...}]
        """
        normalized = []
        for ent in entities:
            label = ent.get("entity_group") or ent.get("label") or "UNKNOWN"
            text = ent.get("word") or ent.get("text")
            if text:
                normalized.append({
                    "text": text.strip(),
                    "label": label.strip()
                })
        return normalized

