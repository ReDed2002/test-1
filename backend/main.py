from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database import HistoryDatabase
from backend.recognition import SignRecognizer
from manipuri_translation import get_translation_details, translate_to_manipuri

app = FastAPI(title="ISL to Meiteilon Translator")
recognizer = None
database = HistoryDatabase(ROOT / "recognition_history.db")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


class HistoryEntry(BaseModel):
    sign: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)


@app.on_event("startup")
def startup():
    global recognizer
    recognizer = SignRecognizer(ROOT)


@app.on_event("shutdown")
def shutdown():
    if recognizer is not None:
        recognizer.close()


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/status")
def status():
    return {"status": "running", "recognition_model_loaded": recognizer is not None}


@app.post("/api/predict")
async def predict(
    frame: UploadFile = File(...),
    detect_body: bool = Form(True),
    detect_face: bool = Form(True)
):
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognition model is not loaded.")
    content = await frame.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty camera frame.")
    try:
        result = recognizer.predict(content, detect_body=detect_body, detect_face=detect_face)
        if result.get("recognized"):
            result.update(get_translation_details(result["sign"]))
        else:
            result["translation"] = ""
            result["audio_url"] = ""
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/history")
def save_history(entry: HistoryEntry):
    translation = translate_to_manipuri(entry.sign)
    if not translation:
        raise HTTPException(status_code=400, detail="No Meiteilon translation is configured for this sign.")
    record_id = database.add(entry.sign, translation, entry.confidence)
    return {"saved": True, "id": record_id}


@app.get("/api/history")
def get_history():
    return database.recent()


@app.delete("/api/history")
def clear_history():
    database.clear()
    return {"cleared": True}
