"""
Reusable inference logic for the fine-tuned LayoutLMv3 form-understanding model.

This module intentionally contains NO training code and NO notebook/Colab-only
commands (`!pip install`, `!apt-get`, etc.) — it is meant to be imported by the
FastAPI service (`app/api.py`) or any other Python process, and is the
single source of truth for "how do I get a prediction out of this model"
so the notebook and the API can't silently drift apart.

Unlike the raw FUNSD dataset (which ships pre-tokenized words + bounding boxes),
real documents arrive as plain images. `ocr_words_and_boxes` runs Tesseract OCR
to reconstruct words + normalized boxes before handing them to the model, which
is what makes this usable on documents the model has never seen annotations for.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import TypedDict

import pytesseract
import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODEL_DIR = os.environ.get("MODEL_DIR", "./model")

LABEL_LIST = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

ENTITY_COLORS = {
    "HEADER": "#2563eb",    # blue
    "QUESTION": "#dc2626",  # red
    "ANSWER": "#16a34a",    # green
}


class WordPrediction(TypedDict):
    word: str
    box: list[int]          # [x0, y0, x1, y1], normalized 0-1000
    label: str               # e.g. "B-ANSWER"
    entity: str               # e.g. "ANSWER" (BIO prefix stripped, "O" if none)
    color: str
    confidence: float


@lru_cache(maxsize=1)
def load_model():
    """
    Load the fine-tuned model + processor once per process.

    Raises a clear error (instead of a confusing HuggingFace stack trace) if the
    model hasn't been exported yet, since that's the single most likely setup
    mistake for anyone cloning this repo.
    """
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"MODEL_DIR '{MODEL_DIR}' not found. Train the model and export it "
            f"first — see notebook.ipynb Section 12 ('Save Model for Serving'), "
            f"or set the MODEL_DIR environment variable to point at an existing "
            f"checkpoint."
        )
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, processor


def ocr_words_and_boxes(pil_image: Image.Image) -> tuple[list[str], list[list[int]]]:
    """Run Tesseract OCR and return (words, normalized_boxes) in LayoutLMv3's 0-1000 format."""
    w, h = pil_image.size
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [
            int(1000 * x / w), int(1000 * y / h),
            int(1000 * (x + bw) / w), int(1000 * (y + bh) / h),
        ]
        words.append(text)
        boxes.append(box)
    return words, boxes


def extract_key_value_pairs(results: list[WordPrediction]) -> list[dict]:
    """Groups consecutive QUESTION/ANSWER words and pairs them spatially."""
    questions, answers = [], []
    current_chunk = []
    current_entity = None
    
    # Group contiguous tokens
    for r in results:
        if r["entity"] in ["QUESTION", "ANSWER"]:
            if r["entity"] == current_entity:
                current_chunk.append(r)
            else:
                if current_chunk:
                    (questions if current_entity == "QUESTION" else answers).append(current_chunk)
                current_chunk = [r]
                current_entity = r["entity"]
        else:
            if current_chunk:
                (questions if current_entity == "QUESTION" else answers).append(current_chunk)
                current_chunk = []
                current_entity = None
                
    if current_chunk:
        (questions if current_entity == "QUESTION" else answers).append(current_chunk)

    pairs = []
    def get_chunk_info(chunk):
        text = " ".join([c["word"] for c in chunk])
        x0 = min([c["box"][0] for c in chunk])
        y0 = min([c["box"][1] for c in chunk])
        x1 = max([c["box"][2] for c in chunk])
        y1 = max([c["box"][3] for c in chunk])
        conf = sum([c["confidence"] for c in chunk]) / len(chunk)
        return text, (x0, y0, x1, y1), conf

    used_answers = set()

    for q_chunk in questions:
        q_text, q_box, q_conf = get_chunk_info(q_chunk)
        best_dist = float('inf')
        best_a = None
        best_a_idx = -1
        
        for i, a_chunk in enumerate(answers):
            if i in used_answers:
                continue
                
            a_text, a_box, a_conf = get_chunk_info(a_chunk)
            q_cx = (q_box[0] + q_box[2]) / 2
            q_cy = (q_box[1] + q_box[3]) / 2
            a_cx = (a_box[0] + a_box[2]) / 2
            a_cy = (a_box[1] + a_box[3]) / 2
            
            # Heuristic: Answer should generally be to the right or directly below
            if a_cx > q_cx - 100 and a_cy > q_cy - 50:
                dist = ((a_cx - q_cx)**2 + (a_cy - q_cy)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_a = (a_text, a_conf)
                    best_a_idx = i
                    
        if best_a:
            used_answers.add(best_a_idx)
            pairs.append({
                "question": q_text,
                "answer": best_a[0],
                "confidence": round((q_conf + best_a[1]) / 2, 4)
            })
        else:
            pairs.append({
                "question": q_text,
                "answer": "",
                "confidence": round(q_conf, 4)
            })
            
    return pairs


def predict(pil_image: Image.Image) -> tuple[list[WordPrediction], list[dict]]:
    """
    Run OCR + LayoutLMv3 on an arbitrary document image and return one entry
    per detected word, along with paired key-value structured data.
    """
    model, processor = load_model()
    pil_image = pil_image.convert("RGB")

    words, boxes = ocr_words_and_boxes(pil_image)
    if not words:
        return [], []

    encoding = processor(pil_image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512, is_split_into_words=True)
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v for k, v in encoding.items() if k != "word_ids"}

    with torch.no_grad():
        logits = model(**inputs).logits
        
    probs = torch.softmax(logits, dim=-1)
    confidences, pred_ids = probs.max(dim=-1)
    
    pred_ids = pred_ids.squeeze().tolist()
    confidences = confidences.squeeze().tolist()

    word_to_pred: dict[int, int] = {}
    word_to_conf: dict[int, float] = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in word_to_pred:
            word_to_pred[word_idx] = pred_ids[token_idx]
            word_to_conf[word_idx] = confidences[token_idx]

    results: list[WordPrediction] = []
    for i, (word, box) in enumerate(zip(words, boxes)):
        label = ID2LABEL.get(word_to_pred.get(i, 0), "O")
        entity = label.split("-", 1)[1] if "-" in label else "O"
        results.append(
            WordPrediction(
                word=word,
                box=box,
                label=label,
                entity=entity,
                color=ENTITY_COLORS.get(entity, "#6b7280"),
                confidence=round(word_to_conf.get(i, 0.0), 4)
            )
        )
        
    structured_data = extract_key_value_pairs(results)
    
    return results, structured_data
