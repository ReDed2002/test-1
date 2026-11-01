import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"
# Create options
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

# Create detector
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened")
    exit()

print("Webcam started.")
print("Show your hand to the camera.")
print("Press Q to quit.")

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read webcam frame")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect
    result = detector.detect_for_video(
        mp_image,
        frame_number
    )

    frame_number += 1

    # If hand detected
    if len(result.hand_landmarks) > 0:

        print("HAND DETECTED!")

        for hand in result.hand_landmarks:

            # Draw all 21 landmarks
            for i, landmark in enumerate(hand):

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Point
                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )

                # Number
                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )

            # Connect landmarks
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]

            for a, b in connections:

                x1 = int(hand[a].x * w)
                y1 = int(hand[a].y * h)

                x2 = int(hand[b].x * w)
                y2 = int(hand[b].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    else:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show window
    cv2.imshow(
        "ISL Hand Landmark Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()

print("Program ended.")