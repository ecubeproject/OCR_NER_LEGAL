from transformers import pipeline
import re
from typing import List, Dict, Tuple

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def run_ner_bert(text: str):
    return ner_pipeline(text)


def extract_grantor_grantee_with_ner(text: str, person_entities: List[Dict]) -> Tuple[str, str]:
    text_upper = text.upper()
    text_relevant = text_upper.split("EXHIBIT A")[0] if "EXHIBIT A" in text_upper else text

    pattern = r"hereby\s+GRANT\(S\)\s+to"
    match = re.search(pattern, text_relevant, re.IGNORECASE)
    if not match:
        return "", ""

    grant_keyword_pos = match.start()

    grantors = [ent["word"] for ent in person_entities if ent["start"] < grant_keyword_pos]
    grantees = [ent["word"] for ent in person_entities if ent["start"] >= grant_keyword_pos]

    return ", ".join(grantors), ", ".join(grantees)
