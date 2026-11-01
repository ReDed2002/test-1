import cv2
import mediapipe as mp
import os


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)

print("Model path:")
print(MODEL_PATH)


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Face model not found!")
    print(MODEL_PATH)
    exit()

print("Face model found!")


# --------------------------------------------------
# CREATE FACE LANDMARKER
# --------------------------------------------------

options = mp.tasks.vision.FaceLandmarkerOptions(

    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=mp.tasks.vision.RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


face_detector = (
    mp.tasks.vision.FaceLandmarker
    .create_from_options(options)
)


# --------------------------------------------------
# WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    exit()

print("Webcam started.")
print("Show your face to the camera.")
print("Press Q to quit.")


frame_number = 0


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:

        print("Could not read webcam.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # Image dimensions
    h, w, _ = frame.shape


    # BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------

    result = face_detector.detect_for_video(
        mp_image,
        frame_number
    )

    frame_number += 1


    # --------------------------------------------------
    # DRAW FACE LANDMARKS
    # --------------------------------------------------

    if result.face_landmarks:

        for face in result.face_landmarks:

            # Draw every 10th landmark
            # so the screen does not become too crowded.

            for i in range(
                0,
                len(face),
                10
            ):

                landmark = face[i]

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 255),
                    -1
                )


        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    cv2.imshow(
        "MediaPipe Face Landmarks",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

face_detector.close()

print("Program ended.")