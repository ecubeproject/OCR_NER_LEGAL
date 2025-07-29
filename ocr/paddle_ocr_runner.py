# ocr/paddle_ocr_runner.py
'''
This module:

    Initializes PaddleOCR in CPU mode

    Accepts an image path

    Returns list of text blocks with bounding boxes and confidence

Sample Output (one line)
[
  ([[80, 45], [200, 45], [200, 70], [80, 70]], "Paula M. Byrens", 0.9851),
  ...
]

'''

from paddleocr import PaddleOCR
from PIL import Image
from typing import List, Tuple

class PaddleOCRRunner:
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        """
        Initialize the OCR engine with desired configurations.
        """
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def run_ocr(self, image_path: str) -> List[Tuple[List[Tuple[int, int]], str, float]]:
        """
        Run OCR on the given image path.
        
        Parameters:
            image_path (str): Path to the image (.tif, .png, etc.)

        Returns:
            List of tuples: [ (bounding_box, text, confidence_score), ... ]
        """
        result = self.ocr.ocr(image_path, cls=True)
        output = []
        for line in result[0]:
            box, (text, score) = line
            output.append((box, text, score))
        return output
