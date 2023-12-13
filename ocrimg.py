import os
from pathlib import Path

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR, PPStructure, draw_ocr, save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import (
    convert_info_docx, sorted_layout_boxes)
from PIL import Image

ocr_engine_ch = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False) # need to run only once to download and load model into memory
ocr_engine_en = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) # need to run only once to download and load model into memory
structure_engine = PPStructure(table=False, ocr=False, show_log=False)

def center_y(elem):
    return (elem[0][0][1]+elem[0][2][1])/2
def height(row):
    return elem[0][2][1] - elem[0][0][1]
def bound_x(row):
    return (row[0][0][0][0], row[-1][0][2][0])
def bound_y(row):
    return (row[0][0][0][1], row[-1][0][2][1])
def height(row):
    top, bottom = bound_y(row)
    return bottom - top

def write_ocr_result(ocr_results, output_path: str, offset: int = 5, mode: str = "w", encoding: str = "utf-8"):
    # 按照 y中点 坐标排序
    sorted_by_y = sorted(ocr_results, key=lambda x: center_y(x))
    results = []
    temp_row = [sorted_by_y[0]]
    for i in range(1, len(sorted_by_y)):
        # 如果和前一个元素的 y 坐标差值小于偏移量，则视为同一行
        if abs(center_y(sorted_by_y[i]) - center_y(sorted_by_y[i-1])) < offset:
            temp_row.append(sorted_by_y[i])
        else:
            # 按照 x 坐标排序，将同一行的元素按照 x 坐标排序
            temp_row = sorted(temp_row, key=lambda x: x[0][0])
            # 将同一行的元素添加到结果列表中
            results.append(temp_row)
            temp_row = [sorted_by_y[i]]
    # 将最后一行的元素添加到结果列表中
    temp_row = sorted(temp_row, key=lambda x: x[0][0])
    results.append(temp_row)
    with open(output_path, mode=mode, encoding=encoding) as f:
        paragraph = ""
        lb, rb = bound_x(results[0])
        for row in results:
            line = ""
            clb, crb = bound_x(row)
            for item in row:
                pos, (text, prob) = item
                line += f"{text} "
            line = line.rstrip()
            paragraph += f"{line}"
            if abs(crb - rb) > 3*offset:
                f.write(f"{paragraph}\n")
                paragraph = ""
            lb, rb = clb, crb
        f.write(f"{paragraph}")

def ocr_from_image(input_path: str, lang: str = 'ch', output_path: str = None, offset: float = 5., use_double_columns: bool = False, show_log: bool = False):
    try:
        p = Path(input_path)
        if output_path is None:
            output_dir = p.parent / f"{p.stem}-OCR识别"
        else:
            output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        text_output_path = str(output_dir / f"{p.stem}-ocr.txt")
        if lang == 'ch':
            ocr_engine = ocr_engine_ch
        elif lang == 'en':
            ocr_engine = ocr_engine_en
        else:
            raise ValueError("不支持的语言")
        result = ocr_engine.ocr(input_path, cls=False)[0]
        write_ocr_result(result, text_output_path, offset)
    except:
