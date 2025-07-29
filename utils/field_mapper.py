'''
Responsibilities:

    Take list of {"text": ..., "label": ...} entities

    Map to required fields:

        Buyer Name

        Seller Name

        Property Address

        Date of Agreement / Registration

        Sale Consideration

        Property ID / Survey Number

        Registrar Office / Jurisdiction

        Witness Names (optional)

    Apply fallback rules or heuristics for missing fields
'''
# utils/field_mapper.py

from typing import List, Dict
import re

# Define mapping synonyms
BUYER_KEYWORDS = ["grantee", "buyer", "purchaser", "vendee"]
SELLER_KEYWORDS = ["grantor", "seller", "vendor"]
ADDRESS_LABELS = ["GPE", "LOC", "FAC", "ADDRESS"]
MONEY_LABELS = ["MONEY", "AMOUNT"]
DATE_LABELS = ["DATE", "REG_DATE"]
PERSON_LABELS = ["PERSON", "NAME"]

def clean_text(text):
    """Remove subword artifacts, punctuation, and surrounding junk."""
    text = text.strip()
    text = re.sub(r"^##", "", text)
    text = re.sub(r"[^\w\s.,/-]", "", text)  # allow some address symbols
    return text.strip()

def map_entities_to_fields(entities: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Map NER entities to structured fields using rules and heuristics.
    """

    fields = {
        "Buyer Name": "",
        "Seller Name": "",
        "Property Address": "",
        "Date of Agreement": "",
        "Sale Consideration": "",
        "Property ID / Survey Number": "",
        "Registrar Office": "",
        "Witness Names": ""
    }

    # Buffers
    buyer_candidates = []
    seller_candidates = []
    address_parts = []
    money_vals = []
    dates = []
    witnesses = []
    others = []

    for ent in entities:
        label = ent.get("label") or ent.get("entity") or ent.get("entity_group") or ent.get("label_")
        text = ent.get("text") or ent.get("word") or ""
        if not label or not text:
            continue

        text = clean_text(text)
        label = label.strip().upper()

        label_lower = label.lower()
        text_lower = text.lower()

        # Map by keyword hint
        if any(k in text_lower for k in BUYER_KEYWORDS):
            buyer_candidates.append(text)
        elif any(k in text_lower for k in SELLER_KEYWORDS):
            seller_candidates.append(text)
        elif label in PERSON_LABELS:
            others.append(text)  # fallback people
        elif label in ADDRESS_LABELS:
            address_parts.append(text)
        elif label in MONEY_LABELS:
            money_vals.append(text)
        elif label in DATE_LABELS:
            dates.append(text)
        elif "witness" in text_lower:
            witnesses.append(text)
        else:
            others.append(text)

    # Assign fields
    if buyer_candidates:
        fields["Buyer Name"] = ", ".join(buyer_candidates)
    elif others:
        fields["Buyer Name"] = others[0]

    if seller_candidates:
        fields["Seller Name"] = ", ".join(seller_candidates)
    elif len(others) > 1:
        fields["Seller Name"] = others[1]

    if address_parts:
        fields["Property Address"] = ", ".join(address_parts[:3])
    if dates:
        fields["Date of Agreement"] = dates[0]
    if money_vals:
        fields["Sale Consideration"] = money_vals[0]
    if witnesses:
        fields["Witness Names"] = ", ".join(witnesses)

    return fields

