
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "sign_translations.json"
AUDIO_DIR = ROOT / "frontend" / "audio"


def _load_dictionary():
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_translation_details(sign):
    entry = _load_dictionary().get(str(sign), {})
    translation = entry.get("translation", "")
    audio_name = entry.get("audio", "")
    audio_url = ""
    if audio_name:
        # Only serve files placed within frontend/audio.
        relative = Path(audio_name)
        if not relative.is_absolute() and ".." not in relative.parts:
            audio_path = (ROOT / "frontend" / relative).resolve()
            frontend_root = (ROOT / "frontend").resolve()
            if frontend_root in audio_path.parents and audio_path.is_file():
                audio_url = "/static/" + relative.as_posix()
    return {"translation": translation, "audio_url": audio_url}


def translate_to_manipuri(sign):
    return get_translation_details(sign)["translation"]
