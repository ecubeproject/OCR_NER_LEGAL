# utils/ner_engine_factory.py

from utils.spacy_ner_engine import SpacyNEREngine
from utils.hf_ner_engine import HuggingFaceNEREngine

def get_ner_engine(model_choice: str):
    """
    Returns the appropriate NER engine instance based on selected model.
    """
    if model_choice == "spaCy (large)":
        return SpacyNEREngine("en_core_web_trf")
    elif model_choice == "spaCy (small)":
        return SpacyNEREngine("en_core_web_sm")
    elif model_choice == "BERT (legal)":
        return HuggingFaceNEREngine("nlpaueb/legal-bert-base-uncased")
    elif model_choice == "BERT (general)":
        return HuggingFaceNEREngine("dslim/bert-base-NER")
    else:
        raise ValueError(f"Unsupported NER model selection: {model_choice}")
