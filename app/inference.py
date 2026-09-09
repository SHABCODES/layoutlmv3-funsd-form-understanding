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


def predict(pil_image: Image.Image) -> list[WordPrediction]:
    """
    Run OCR + LayoutLMv3 on an arbitrary document image and return one entry
    per detected word, correctly aligned via `word_ids()` (each word's label
    is taken from its first subword token — see notebook Section 13 for why
    this matters and what breaks if you skip it).
    """
    model, processor = load_model()
    pil_image = pil_image.convert("RGB")

    words, boxes = ocr_words_and_boxes(pil_image)
    if not words:
        return []

    encoding = processor(pil_image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512, is_split_into_words=True)
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v for k, v in encoding.items() if k != "word_ids"}

    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = logits.argmax(-1).squeeze().tolist()

    word_to_pred: dict[int, int] = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in word_to_pred:
            word_to_pred[word_idx] = pred_ids[token_idx]

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
            )
        )
    return results
