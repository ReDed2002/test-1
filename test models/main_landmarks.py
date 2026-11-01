import cv2
import mediapipe as mp
import os
import numpy as np
import torch
import torch.nn as nn
from manipuri_translation import translate_to_manipuri
from collections import deque
from PIL import Image, ImageDraw, ImageFont



# ============================================================
# BASE DIRECTORY
# ============================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_DIR = os.path.join(BASE_DIR, "fonts")

ttf_files = [
    f for f in os.listdir(FONT_DIR)
    if f.lower().endswith(".ttf")
]

if not ttf_files:
    raise FileNotFoundError(
        "No .ttf font found inside the fonts folder."
    )

FONT_PATH = os.path.join(FONT_DIR, ttf_files[0])

print("Font found:")
print(FONT_PATH)

manipuri_font = ImageFont.truetype(FONT_PATH, 32)

print("Font loaded successfully.")

# ============================================================
# MEDIAPIPE MODEL PATHS
# ============================================================

HAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)

POSE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pose_landmarker_lite.task"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


# ============================================================
# ISL MODEL PATHS
# ============================================================

ISL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isl",
    "sign_lstm_torch.pth"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isl",
    "labels.npy"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.75


# ============================================================
# CHECK MODEL FILES
# ============================================================

print("Checking model files...")

if not os.path.exists(HAND_MODEL_PATH):
    print("ERROR: Hand model not found:")
    print(HAND_MODEL_PATH)
    exit()

if not os.path.exists(POSE_MODEL_PATH):
    print("ERROR: Pose model not found:")
    print(POSE_MODEL_PATH)
    exit()

if not os.path.exists(FACE_MODEL_PATH):
    print("ERROR: Face model not found:")
    print(FACE_MODEL_PATH)
    exit()

if not os.path.exists(ISL_MODEL_PATH):
    print("ERROR: ISL model not found:")
    print(ISL_MODEL_PATH)
    exit()

if not os.path.exists(LABELS_PATH):
    print("ERROR: Labels file not found:")
    print(LABELS_PATH)
    exit()


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=42,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        _, (hn, _) = self.lstm(x)

        return self.fc(hn[-1])


# ============================================================
# LOAD ISL LABELS
# ============================================================

labels = np.load(
    LABELS_PATH,
    allow_pickle=True
)


# ============================================================
# LOAD ISL LSTM MODEL
# ============================================================

print()
print("Loading ISL model...")

isl_model = LSTMModel(
    len(labels)
)

isl_model.load_state_dict(
    torch.load(
        ISL_MODEL_PATH,
        map_location="cpu"
    )
)

isl_model.eval()

print()
print("================================")
print("ISL MODEL LOADED SUCCESSFULLY")
print("================================")

print()
print("Available signs:")

for label in labels:
    print(" -", label)


# ============================================================
# EXTRACT HAND FEATURES
# ============================================================

def extract_hand_features(hand):

    # Wrist = landmark 0

    base_x = hand[0].x
    base_y = hand[0].y

    data = []

    # 21 landmarks
    for landmark in hand:

        data.append(
            landmark.x - base_x
        )

        data.append(
            landmark.y - base_y
        )

    return data


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ------------------------------------------------------------
# HAND
# ------------------------------------------------------------

hand_options = mp.tasks.vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=2
)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
    hand_options
)


# ------------------------------------------------------------
# POSE
# ------------------------------------------------------------

pose_options = mp.tasks.vision.PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_poses=1
)

pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
    pose_options
)


# ------------------------------------------------------------
# FACE
# ------------------------------------------------------------

face_options = mp.tasks.vision.FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_faces=1
)

face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(
    face_options
)





# ============================================================
# DISPLAY SETTINGS
# ============================================================

show_hand = True
show_body = True
show_face = True


# ============================================================
# ISL SEQUENCE
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

last_prediction = ""

last_confidence = 0.0

last_manipuri_text = ""


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


# ============================================================
# FRAME NUMBER
# ============================================================

frame_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read frame.")

        break


    # --------------------------------------------------------
    # FLIP CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # CONVERT FRAME FOR MEDIAPIPE
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms = frame_number * 33

    frame_number += 1


    # ========================================================
    # HAND DETECTION
    # ========================================================

    hand_result = hand_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # POSE DETECTION
    # ========================================================

    pose_result = pose_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_result = face_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # HAND LANDMARKS + ISL FEATURES
    # ========================================================

    if show_hand and hand_result.hand_landmarks:

        # ----------------------------------------------------
        # USE FIRST HAND FOR EXISTING ISL MODEL
        # ----------------------------------------------------

        hand = hand_result.hand_landmarks[0]


        # ----------------------------------------------------
        # EXTRACT 42 FEATURES
        # ----------------------------------------------------

        features = extract_hand_features(
            hand
        )


        # ----------------------------------------------------
        # ADD FRAME TO SEQUENCE
        # ----------------------------------------------------

        sequence.append(
            features
        )


        # ----------------------------------------------------
        # DRAW HAND LANDMARKS
        # ----------------------------------------------------

        for landmark in hand:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


    else:

        # No hand detected
        sequence.clear()


    # ========================================================
    # ISL PREDICTION
    # ========================================================

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(
            sequence,
            dtype=np.float32
        )


        # Shape:
        # (30, 42)


        input_tensor = torch.tensor(
            input_data,
            dtype=torch.float32
        ).unsqueeze(0)


        # Shape:
        # (1, 30, 42)


        with torch.no_grad():

            output = isl_model(
                input_tensor
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )[0]


            index = torch.argmax(
                probabilities
            ).item()


            confidence = probabilities[
                index
            ].item()


            prediction = labels[
                index
            ]
            manipuri_text = translate_to_manipuri( str(prediction)
            )


        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        if confidence >= CONFIDENCE_THRESHOLD:

            last_prediction = str(
                prediction
            )

            last_confidence = confidence

            last_manipuri_text = manipuri_text

        else:

            last_prediction = "Unknown"

            last_confidence = confidence

            last_manipuri_text = "Unknown"


    # ========================================================
    # DRAW BODY LANDMARKS
    # ========================================================

    if show_body and pose_result.pose_landmarks:

        for pose_landmarks in pose_result.pose_landmarks:

            for landmark in pose_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (255, 0, 0),
                    -1
                )


    # ========================================================
    # DRAW FACE LANDMARKS
    # ========================================================

    if show_face and face_result.face_landmarks:

        for face_landmarks in face_result.face_landmarks:

            for landmark in face_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 255),
                    -1
                )


    # ========================================================
    # ISL RESULT BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 20),
        (430, 105),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "ISL: " + last_prediction,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Confidence: {:.1f}%".format(
            last_confidence * 100
        ),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    # ========================================================
    # MANIPURI TRANSLATION
    # ========================================================
           
    pil_frame = Image.fromarray(
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(pil_frame)

    draw.text(
    (20, 120),
    "Manipuri: " + last_manipuri_text,
    font=manipuri_font,
    fill=(255, 255, 0)
)

    frame = cv2.cvtColor(
    np.array(pil_frame),
    cv2.COLOR_RGB2BGR
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "ISL - Hand + Body + Face + Recognition",
        frame
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # TOGGLE HAND
    # --------------------------------------------------------

    elif key == ord("h"):

        show_hand = not show_hand

        if not show_hand:

            sequence.clear()


    # --------------------------------------------------------
    # TOGGLE BODY
    # --------------------------------------------------------

    elif key == ord("b"):

        show_body = not show_body


    # --------------------------------------------------------
    # TOGGLE FACE
    # --------------------------------------------------------

    elif key == ord("f"):

        show_face = not show_face


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hand_detector.close()
pose_detector.close()
face_detector.close()