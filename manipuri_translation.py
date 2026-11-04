
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "sign_translations.json"
SENTENCE_PATH = ROOT / "sentence_translations.json"
AUDIO_DIR = ROOT / "frontend" / "audio"


def _load_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_dictionary():
    return _load_json(CONFIG_PATH, {})


def _load_sentences():
    data = _load_json(SENTENCE_PATH, {})
    return data.get("sentences", []) if isinstance(data, dict) else []


def _audio_url(audio_name):
    """Return a /static/... URL for a clip inside frontend/, or "" if missing/unsafe."""
    if not audio_name:
        return ""
    # Only serve files placed within frontend/.
    relative = Path(audio_name)
    if relative.is_absolute() or ".." in relative.parts:
        return ""
    audio_path = (ROOT / "frontend" / relative).resolve()
    frontend_root = (ROOT / "frontend").resolve()
    if frontend_root in audio_path.parents and audio_path.is_file():
        return "/static/" + relative.as_posix()
    return ""


def get_translation_details(sign):
    entry = _load_dictionary().get(str(sign), {})
    return {
        "translation": entry.get("translation", ""),
        "audio_url": _audio_url(entry.get("audio", "")),
    }


def translate_to_manipuri(sign):
    return get_translation_details(sign)["translation"]


def _normalize(signs):
    return [str(s).strip().upper() for s in signs if str(s).strip()]


def get_sentence_details(signs):
    """Look up a full sentence from an ordered list of recognized signs.

    Returns:
      matched      - True if the exact sign order is listed in sentence_translations.json
      translation  - the sentence (matched) or the word translations joined (fallback)
      audio_url    - one recorded clip for the whole sentence ("" if none)
      verified     - whether the sentence has been checked by a native speaker
      word_audio_urls - per-word clips, used when there is no sentence clip
    """
    wanted = _normalize(signs)

    words = [get_translation_details(sign) for sign in wanted]
    word_audio_urls = [w["audio_url"] for w in words if w["audio_url"]]
    joined = " ".join(w["translation"] for w in words if w["translation"])

    for entry in _load_sentences():
        if _normalize(entry.get("signs", [])) == wanted:
            return {
                "matched": True,
                "translation": entry.get("translation") or joined,
                "audio_url": _audio_url(entry.get("audio", "")),
                "verified": bool(entry.get("verified", False)),
                "word_audio_urls": word_audio_urls,
            }

    return {
        "matched": False,
        "translation": joined,
        "audio_url": "",
        "verified": False,
        "word_audio_urls": word_audio_urls,
    }
