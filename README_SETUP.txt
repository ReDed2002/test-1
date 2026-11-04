UPDATED LOCAL WEBSITE

Replace the backend/ and frontend/ folders in your project with the folders
from this package. Keep models/ and manipuri_translation.py unchanged.

Install packages if not already installed:
  pip install fastapi uvicorn python-multipart

Run from the project root:
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

Open http://127.0.0.1:8000

Changes:
- Browser sends frames at a 70ms target interval instead of 180ms.
- JPEG quality is reduced slightly to make frame uploads smaller.
- The 30-frame LSTM sequence length is NOT changed, preserving compatibility.
- Hand, pose/body, and face landmarks are returned by the backend.
- Three independent checkboxes turn hand, body and face reference points on/off.
- Landmark points are drawn as an overlay on the browser video.
- Requires all three task files: hand_landmarker.task, pose_landmarker_lite.task,
  face_landmarker.task, plus the existing ISL model and labels.

If recognition is still slow, the actual limiting factor may be CPU time spent
running three MediaPipe detectors per frame. This version runs pose and face
detection too because their reference-point overlays were requested. For the
fastest recognition-only mode, we can make pose/face detection conditional
on the corresponding checkboxes so they do not run when switched off.


LATEST UPDATE
-------------
1. Landmark alignment: corrected the horizontal coordinate mapping so the
   hand/body/face reference points line up with the mirrored camera preview.
2. Performance: when Body or Face reference points are unchecked, the backend
   now skips that detector. Keep Hand enabled because the sign classifier uses
   hand landmarks.
3. Voice buttons:
   - Speak English uses the browser's installed English speech voice.
   - Speak Meiteilon checks for an installed Manipuri/Meiteilon (mni) speech
     voice. If none is available, the site explains that a dedicated Meiteilon
     TTS voice model is needed; it will not pretend to speak it in English.

After replacing backend and frontend folders, restart the server. Refresh the
browser page after the server starts.


LATEST UPDATE: DIRECT MEITEILON TRANSLATION + RECORDED AUDIO
-----------------------------------------------------------
- The backend now returns the Meiteilon text and the corresponding recorded
  audio URL for each recognized sign.
- After the same sign is recognized consistently for 3 frames, its recorded
  Meiteilon audio plays automatically (when a recording is configured).
- The Replay Meiteilon audio button replays the current sign's recording.
- Browser speech synthesis is no longer used for Meiteilon, so playback does
  not depend on an installed Manipuri voice.
- Configure exact model class labels and translations in
  sign_translations.json. The labels must match models/isl/labels.npy.
- Put recordings in frontend/audio/ and set each relative filename in the
  matching JSON entry. Example: "audio": "audio/hello.mp3".
- The two entries in the supplied JSON are examples/placeholders only. Replace
  them with your actual recognized labels, correct Meiteilon translations,
  and audio filenames. Do not leave placeholder values.
- Audio recording files are not included; add your own recordings.

IMPORTANT:
This archive does not include the trained ISL classifier files or the existing
model's class labels. Keep your models/ directory when copying these changes.
The translation JSON is a template because correct Meiteilon translations and
the exact model labels must come from your project; they are not guessed here.

Run:
  pip install fastapi uvicorn python-multipart
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
