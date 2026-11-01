import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=42, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class SignRecognizer:
    def __init__(self, project_root):
        self.sequence_length = 30
        self.confidence_threshold = 0.75
        self.sequence = deque(maxlen=self.sequence_length)
        self.last_timestamp_ms = 0

        model_dir = os.path.join(project_root, "models")
        hand_path = os.path.join(model_dir, "hand_landmarker.task")
        pose_path = os.path.join(model_dir, "pose_landmarker_lite.task")
        face_path = os.path.join(model_dir, "face_landmarker.task")
        model_path = os.path.join(model_dir, "isl", "sign_lstm_torch.pth")
        labels_path = os.path.join(model_dir, "isl", "labels.npy")

        missing = [p for p in (hand_path, pose_path, face_path, model_path, labels_path)
                   if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))

        self.labels = np.load(labels_path, allow_pickle=True)
        self.model = LSTMModel(len(self.labels))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        video_mode = vision.RunningMode.VIDEO

        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=hand_path),
                running_mode=video_mode,
                num_hands=2
            )
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=pose_path),
                running_mode=video_mode,
                num_poses=1
            )
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=face_path),
                running_mode=video_mode,
                num_faces=1
            )
        )

    @staticmethod
    def extract_hand_features(hand):
        wrist_x, wrist_y = hand[0].x, hand[0].y
        features = []
        for landmark in hand:
            features.extend([landmark.x - wrist_x, landmark.y - wrist_y])
        return features

    @staticmethod
    def points(landmarks, every=1):
        return [{"x": float(p.x), "y": float(p.y)}
                for p in landmarks[::every]]

    def predict(self, image_bytes, detect_body=True, detect_face=True):
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"recognized": False, "sign": "", "confidence": 0.0,
                    "message": "Could not decode camera frame.",
                    "landmarks": {"hand": [], "body": [], "face": []}}

        # Keep the same mirrored-camera convention as the desktop prototype.
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.monotonic_ns() / 1_000_000),
                           self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = (self.pose_detector.detect_for_video(mp_image, timestamp_ms)
                       if detect_body else None)
        face_result = (self.face_detector.detect_for_video(mp_image, timestamp_ms)
                       if detect_face else None)

        hand_points = []
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            hand_points = self.points(hand)
            self.sequence.append(self.extract_hand_features(hand))
        else:
            self.sequence.clear()

        body_points = []
        if pose_result and pose_result.pose_landmarks:
            body_points = self.points(pose_result.pose_landmarks[0])

        face_points = []
        if face_result and face_result.face_landmarks:
            # Display a sample of the face mesh rather than all 468/478 points.
            face_points = self.points(face_result.face_landmarks[0], every=10)

        landmarks = {"hand": hand_points, "body": body_points, "face": face_points}

        if not hand_result.hand_landmarks:
            return {"recognized": False, "sign": "", "confidence": 0.0,
                    "message": "Show your hand clearly to the camera.",
                    "landmarks": landmarks}

        if len(self.sequence) < self.sequence_length:
            return {"recognized": False, "sign": "", "confidence": 0.0,
                    "message": f"Collecting frames: {len(self.sequence)}/{self.sequence_length}",
                    "landmarks": landmarks}

        tensor = torch.tensor(np.asarray(self.sequence, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[index].item())

        sign = str(self.labels[index])
        if confidence < self.confidence_threshold:
            return {"recognized": False, "sign": "Unknown", "confidence": confidence,
                    "message": "Confidence is below the recognition threshold.",
                    "landmarks": landmarks}

        return {"recognized": True, "sign": sign, "confidence": confidence,
                "message": "Sign recognized.", "landmarks": landmarks}

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()
        self.face_detector.close()
