"""
FastAPI service exposing the fine-tuned LayoutLMv3 form-understanding model.

Run:
    uvicorn app.api:app --reload --port 8000

Then either:
  - open web/index.html in a browser (it calls this API directly), or
  - POST an image to /predict yourself, e.g.:
        curl -F "file=@form.png" http://localhost:8000/predict
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.inference import load_model, predict

app = FastAPI(
    title="LayoutLMv3 Form Understanding API",
    description="Upload a scanned form image; get back HEADER / QUESTION / ANSWER "
                 "entities with bounding boxes.",
    version="1.0.0",
)

# Wide-open CORS is fine for a local portfolio demo; tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up_model() -> None:
    """Load the model once at startup instead of on the first request, so the
    first real user doesn't eat a multi-second cold-start latency spike."""
    try:
        load_model()
    except RuntimeError as exc:
        # Don't crash the whole server if the model isn't exported yet —
        # surface a clear error on the first request instead.
        print(f"[startup warning] {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    try:
        results, structured_data = predict(image)
    except RuntimeError as exc:
        # Most likely cause: MODEL_DIR isn't populated yet (see app/inference.py).
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse(
        {
            "image_width": image.width,
            "image_height": image.height,
            "words": results,
            "structured_data": structured_data,
        }
    )

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="web", html=True), name="web")
