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
    "pose_landmarker_lite.task"
)

print("Model path:")
print(MODEL_PATH)


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Pose model not found!")
    print(MODEL_PATH)
    exit()

print("Pose model found!")


# --------------------------------------------------
# CREATE POSE LANDMARKER
# --------------------------------------------------

options = mp.tasks.vision.PoseLandmarkerOptions(

    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=mp.tasks.vision.RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


pose_detector = (
    mp.tasks.vision.PoseLandmarker
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
print("Stand in front of the camera.")
print("Press Q to quit.")


frame_number = 0


# --------------------------------------------------
# POSE CONNECTIONS
# --------------------------------------------------

POSE_CONNECTIONS = [

    # Face/head
    (0,1),
    (1,2),
    (2,3),
    (3,7),

    (0,4),
    (4,5),
    (5,6),
    (6,8),

    # Shoulders
    (11,12),

    # Left arm
    (11,13),
    (13,15),

    # Right arm
    (12,14),
    (14,16),

    # Left hand
    (15,17),
    (15,19),
    (15,21),

    # Right hand
    (16,18),
    (16,20),
    (16,22),

    # Body
    (11,23),
    (12,24),
    (23,24),

    # Left leg
    (23,25),
    (25,27),
    (27,29),
    (29,31),

    # Right leg
    (24,26),
    (26,28),
    (28,30),
    (30,32)
]


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


    # Get dimensions
    h, w, _ = frame.shape


    # BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------
    # POSE DETECTION
    # --------------------------------------------------

    result = pose_detector.detect_for_video(
        mp_image,
        frame_number
    )

    frame_number += 1


    # --------------------------------------------------
    # DRAW POSE
    # --------------------------------------------------

    if result.pose_landmarks:

        for pose in result.pose_landmarks:

            # Draw points
            for i, landmark in enumerate(pose):

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 255),
                    -1
                )

                # Landmark number
                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1
                )


            # Draw connections
            for start, end in POSE_CONNECTIONS:

                x1 = int(
                    pose[start].x * w
                )

                y1 = int(
                    pose[start].y * h
                )

                x2 = int(
                    pose[end].x * w
                )

                y2 = int(
                    pose[end].y * h
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2
                )


    else:

        cv2.putText(
            frame,
            "No body detected",
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
        "MediaPipe Pose - Lite Model",
        frame
    )


    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

pose_detector.close()

print("Program ended.")