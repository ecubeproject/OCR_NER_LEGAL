'''
Responsibilities:

    Accept input .tif or .pdf path

    Convert .pdf to images (if needed)

    Feed each image to:

        PaddleOCRRunner → OCR

        Combine OCR text

        SpacyNEREngine → NER

        map_entities_to_fields() → field mapping

    Print or save structured output


Key Features of gradio UI:

    Upload .pdf or .tif

    Select output format: .json (default) or .csv

    Display extracted fields

    Save downloadable .json and .csv versions to output/

    Outputs

    Structured results are shown in browser

    Files saved to: output/{filename}_{timestamp}.json or .csv
'''
# main.py
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from PIL import Image
from pdf2image import convert_from_path
import gradio as gr
from ocr.paddle_ocr_runner import PaddleOCRRunner
from ner.hf_ner_engine import HuggingFaceNEREngine
from utils.field_mapper import map_entities_to_fields
from ner.ner_utils import run_ner_bert, extract_grantor_grantee_with_ner
from utils.ner_engine_factory import get_ner_engine
from utils.pdf_utils import convert_tif_to_searchable_pdf

import pandas as pd

DEFAULT_CONFIDENCE_THRESHOLD = 0.6

def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    return convert_from_path(pdf_path, dpi=dpi)

def process_image(image: Image.Image, ocr_engine, ner_engine) -> Dict:
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)
    ocr_result = ocr_engine.run_ocr(temp_image_path)
    os.remove(temp_image_path)

    text_lines = [item[1] for item in ocr_result if item[2] > DEFAULT_CONFIDENCE_THRESHOLD]
    full_text = "\n".join(text_lines)
    avg_confidence = sum(item[2] for item in ocr_result) / len(ocr_result) if ocr_result else 0

    person_entities = run_ner_bert(full_text)
    grantor, grantee = extract_grantor_grantee_with_ner(full_text, person_entities)
    entities = ner_engine.extract_entities(full_text)
    mapped_fields = map_entities_to_fields(entities)

    if grantor:
        mapped_fields['Seller Name'] = grantor
    if grantee:
        mapped_fields['Buyer Name'] = grantee

    return {
        "fields": mapped_fields,
        "confidence_avg": round(avg_confidence, 4)
    }
    
def process_single_file(file_path: str, ocr_engine, ner_engine) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()
    results = []

    if ext == ".pdf":
        images = convert_pdf_to_images(file_path)
        for i, img in enumerate(images):
            page_result = process_image(img, ocr_engine, ner_engine)
            page_result["page"] = i + 1
            results.append(page_result)
    elif ext in [".tif", ".tiff", ".jpg", ".png"]:
        img = Image.open(file_path)
        page_result = process_image(img, ocr_engine, ner_engine)
        page_result["page"] = 1
        results.append(page_result)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return results

def run_pipeline(files: List, output_format: str, export_dir: str, selected_model: str):
    if not files:
        return "No files uploaded.", None

    export_dir = export_dir or "output"
    os.makedirs(export_dir, exist_ok=True)

    ocr_engine = PaddleOCRRunner()
    ner_engine = get_ner_engine(selected_model)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = f"results_{timestamp}"

    for file in files:
        input_path = file if isinstance(file, str) else file.name
        filename = os.path.splitext(os.path.basename(input_path))[0]
        file_results = process_single_file(input_path, ocr_engine, ner_engine)
        for page in file_results:
            page["source_file"] = filename
            page["confidence_threshold"] = DEFAULT_CONFIDENCE_THRESHOLD
        all_results.extend(file_results)

    output_path = os.path.join(export_dir, combined_filename + (".json" if output_format == "json" else ".csv"))

    if output_format == "json":
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        flat_rows = []
        for row in all_results:
            flat = {
                "source_file": row["source_file"],
                "page": row["page"],
                "confidence_avg": row["confidence_avg"],
                "confidence_threshold": row["confidence_threshold"]
            }
            flat.update(row["fields"])
            flat_rows.append(flat)

        if flat_rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
                writer.writeheader()
                writer.writerows(flat_rows)
        else:
            with open(output_path, "w") as f:
                f.write("No results to export.\n")

    return all_results, output_path



if __name__ == "__main__":
    with gr.Blocks(title="OCR + NER Extractor") as demo:
        gr.Markdown("## OCR + NER Extractor for Legal Documents")

        with gr.Row():
            file_input = gr.File(label="Upload PDF or Image", file_types=[".pdf", ".tif", ".tiff", ".jpg", ".png"], file_count="multiple")
            output_format = gr.Radio(choices=["json", "csv"], value="json", label="Select Export Format")
            export_folder = gr.Textbox(label="Output Folder", placeholder="Leave blank for default 'output'")

        ner_model_selector = gr.Dropdown(
            choices=[
                "spaCy (large)",
                "spaCy (small)",
                "BERT (general)",
                "BERT (legal)"
            ],
            value="spaCy (large)",
            label="Select NER Model"
        )

        run_button = gr.Button("Run Extraction")

        output_path = gr.Textbox(label="Saved Output File Path")
        result_container = gr.Column()
        result_table = gr.Dataframe(visible=False)
        json_outputs = []

        def handle_run(files, fmt, folder, selected_model):
            folder = folder or "output/pdf_temp"  # <-- FIX: fallback if folder is empty
            file_paths = [f.name for f in files if f and hasattr(f, "name") and os.path.exists(f.name)]
        
            # Convert TIF files to searchable PDFs
            final_inputs = []
            for file_path in file_paths:
                if file_path.lower().endswith(".tif"):
                    try:
                        converted_pdf = convert_tif_to_searchable_pdf(file_path, folder)
                        final_inputs.append(converted_pdf)
                    except Exception as e:
                        raise RuntimeError(f"TIF to PDF conversion failed: {str(e)}")
                else:
                    final_inputs.append(file_path)
        
            results, out_path = run_pipeline(final_inputs, fmt, folder, selected_model)
            # Step 3: Output display
            if fmt == "json":
                for comp in json_outputs:
                    comp.visible = False
                json_outputs.clear()
        
                with result_container:
                    for page in results:
                        comp = gr.JSON(label=f"{page['source_file']} - Page {page['page']}", value=page)
                        json_outputs.append(comp)
                return None, out_path
        
            else:
                flat_rows = []
                for row in results:
                    flat = {
                        "source_file": row["source_file"],
                        "page": row["page"],
                        "confidence_avg": row["confidence_avg"],
                        "confidence_threshold": row["confidence_threshold"]
                    }
                    flat.update(row["fields"])
                    flat_rows.append(flat)
        
                return gr.Dataframe(value=flat_rows), out_path
        
        run_button.click(
            fn=handle_run,
            inputs=[file_input, output_format, export_folder, ner_model_selector],
            outputs=[result_table, output_path]
        )

    demo.launch()



