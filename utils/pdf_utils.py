import os
import uuid
from PIL import Image
import ocrmypdf

def convert_tif_to_searchable_pdf(tif_path: str, export_folder: str) -> str:
    """
    Converts a multi-page .tif file to searchable .pdf using ocrmypdf.
    Returns output PDF path.
    """
    os.makedirs(export_folder, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    temp_pdf = os.path.join(export_folder, f"temp_{unique_id}.pdf")
    output_pdf = os.path.join(export_folder, f"converted_{unique_id}.pdf")

    # Convert TIFF to PDF first
    with Image.open(tif_path) as img:
        img.save(temp_pdf, "PDF", resolution=300.0, save_all=True)

    # Apply OCR to make it searchable
    ocrmypdf.ocr(temp_pdf, output_pdf, force_ocr=True, deskew=True, rotate_pages=True)
    os.remove(temp_pdf)
    return output_pdf
