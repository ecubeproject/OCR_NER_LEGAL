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

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class HuggingFaceNEREngine:
    def __init__(self, model_name="dslim/bert-base-NER"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace NER model '{model_name}': {e}")

    def extract_entities(self, text):
        raw_entities = self.pipe(text)
        return self._normalize_entities(raw_entities)

    def _normalize_entities(self, raw_entities):
        results = []
        for ent in raw_entities:
            results.append({
                "text": ent.get("word") or ent.get("text"),
                "label": ent.get("entity_group") or ent.get("entity"),
                "score": float(ent["score"]),
                "start": ent.get("start", -1),
                "end": ent.get("end", -1)
            })
        return results


