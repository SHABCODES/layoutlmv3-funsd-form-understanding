# Layout-Aware Form Understanding with LayoutLMv3

Fine-tuning Microsoft's LayoutLMv3 on the FUNSD dataset for named entity
recognition across form fields — headers, questions, and answers —
using spatial layout embeddings and bounding box coordinates.

## Results

| Entity   | Precision | Recall | F1   |
|----------|-----------|--------|------|
| ANSWER   | 0.90      | 0.91   | 0.91 |
| QUESTION | 0.88      | 0.91   | 0.89 |
| HEADER   | 0.57      | 0.61   | 0.59 |
| **Overall** | **0.87** | **0.89** | **0.8791** |

## Prediction Visualization

![Predictions](predictions_visualized.png)

Blue = Header | Red = Question | Green = Answer

## Dataset

FUNSD (Form Understanding in Noisy Scanned Documents)
- 149 training documents, 50 test documents
- Source: [nielsr/funsd-layoutlmv3](https://huggingface.co/datasets/nielsr/funsd-layoutlmv3)

## Model

LayoutLMv3-base (Microsoft) fine-tuned for token classification
- Paper: [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
- 10 epochs | Learning rate: 1e-5 | Batch size: 2

## How to Run
```bash
# Install dependencies
pip install transformers datasets seqeval pillow torch torchvision

# Run notebook
jupyter notebook layoutlmv3_funsd.ipynb
```

## Tech Stack

PyTorch · HuggingFace Transformers · LayoutLMv3 · OCR · Docker · seqeval
