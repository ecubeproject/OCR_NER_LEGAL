# OCR + NER Legal Document Extraction App

This project is a Gradio-based application for extracting structured information from **unstructured legal documents** (primarily `.tif` scanned images). It uses **OCR + Named Entity Recognition (NER)** pipelines and rule-based heuristics to extract key fields such as:

- Grantor (Seller) Name  
- Grantee (Buyer) Name  
- Property Address  
- Agreement/Registration Date  
- Sale Consideration (Amount)  
- Property ID / Survey Number  
- Registrar Office / Jurisdiction  
- Witness Names (if available)

---

## Project Features

### Core Functionality

- OCR using **PaddleOCR** (CPU only)
- NER using selectable models via dropdown:
  - `spaCy` Large: `en_core_web_trf`
  - `spaCy` Small: `en_core_web_sm`
  - `BERT` General: `dslim/bert-base-NER`
  - `BERT` Legal: `nlpaueb/legal-bert-base-uncased`
- Custom **postprocessing heuristics** to improve Grantor/Grantee extraction using patterns like `hereby GRANT(S) to`.
- Support for `.tif` input with conversion to searchable `.pdf`.
- Confidence filtering and field-wise score export in `.csv` and `.json`.

---

## Directory Structure
```
ocr_ner_project/
├── main.py # Main Gradio app logic
├── ner/
│ └── hf_ner_engine.py
| |__ ner_utils.py
| |__ spacy_ner_engine.py
├── ocr/
│ └── paddle_ocr_runner.py # PaddleOCR engine wrapper
├── utils/
│ └── field_mapper.py # Maps raw NER entities to form fields
| |__ hf_ner_engine.py
| |__ ner_engine_factory.py
| |__ pdf_utils.py
| |__ spacy_ner_engine.py
├── sample_input/ # Sample .tif and .pdf files
├── output/ # Output CSV/JSON exports
└── requirements.txt # Cleaned requirements for deployment
```
---

##  How to Run  ( Video Demo:   https://youtu.be/5nPYdkucLh4)

### Option 1: Local (for demo/testing)

```bash
```
# Step 1: Create virtual environment
conda create -n ocr_venv python=3.10
conda activate ocr_venv
--- 
# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
python main.py
---

 Future Work

 Model tuning using legal-specific datasets.

 Improve location/address extraction via spatial layout features.

 Integrate layout-aware parsing (e.g., layoutparser, DocLayNet).

Author

Built by Tejas Desai
For queries/added features : [aimldstejas@gmail.com]
