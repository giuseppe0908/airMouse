"""Validated hand tracking wrapper around MediaPipe Hand Landmarker."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import hypot, isfinite
from pathlib import Path
import time
from typing import Any

import config
from security_utils import (
    ensure_within_project,
    install_network_guards,
    prepare_runtime_environment,
)

install_network_guards()
prepare_runtime_environment(config.LOG_DIR_NAME)

import cv2
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)


LOGGER = logging.getLogger("airmouse.hand_tracker")

_NORMALIZED_COORD_TOLERANCE = 0.20

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

NormalizedPoint = tuple[float, float]
NormalizedLandmark3D = tuple[float, float, float]


@dataclass(slots=True)
class HandState:
    """Per-frame hand state extracted from validated landmarks."""

    index_tip: NormalizedPoint
    thumb_tip: NormalizedPoint
    middle_tip: NormalizedPoint
    pinch_thumb_index: float
    pinch_thumb_middle: float
    is_open_hand: bool
    is_resume_gesture: bool
    is_fist: bool
    finger_states: dict[str, bool]
    extended_finger_count: int
    non_thumb_extended_count: int
    handedness: str
    landmarks: list[NormalizedLandmark3D]
    raw_landmarks: list[NormalizedPoint]
    clamped_landmark_count: int


class HandTracker:
    """Tracks a single hand and exposes gesture-friendly measurements."""

    def __init__(self) -> None:
        model_path = ensure_within_project(
            Path(config.PROJECT_ROOT) / config.HAND_LANDMARKER_MODEL_PATH
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. "
                "Expected a local project asset."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0
        self._closed = False
        self._logged_projection_clamp_warning = False

    def process(self, frame: Any) -> HandState | None:
        """Process one BGR frame and return the current hand state."""

        if not self._is_valid_frame(frame):
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        self._last_timestamp_ms = max(
            self._last_timestamp_ms + 1,
            int(time.monotonic() * 1000),
        )
        results = self._hand_landmarker.detect_for_video(
            mp_image,
            self._last_timestamp_ms,
        )

        if not results.hand_landmarks:
            return None

        validated = self._extract_validated_landmarks(results.hand_landmarks[0])
        if validated is None:
            return None

        hand_landmarks, raw_landmarks, clamped_landmark_count = validated
        handedness = self._extract_handedness(results)

        thumb_tip = self._xy(hand_landmarks[4])
        index_tip = self._xy(hand_landmarks[8])
        middle_tip = self._xy(hand_landmarks[12])

        pinch_thumb_index = self._distance(thumb_tip, index_tip)
        pinch_thumb_middle = self._distance(thumb_tip, middle_tip)

        finger_states = self._get_finger_states(hand_landmarks, handedness)
        is_open_hand = all(finger_states.values())
        extended_finger_count = sum(finger_states.values())
        non_thumb_extended_count = sum(
            finger_states[name] for name in ("index", "middle", "ring", "pinky")
        )
        is_resume_gesture = self._is_resume_gesture(
            hand_landmarks,
            finger_states,
            handedness,
            pinch_thumb_index,
            pinch_thumb_middle,
        )
        is_fist = self._is_fist(hand_landmarks, finger_states)

        return HandState(
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            middle_tip=middle_tip,
            pinch_thumb_index=pinch_thumb_index,
            pinch_thumb_middle=pinch_thumb_middle,
            is_open_hand=is_open_hand,
            is_resume_gesture=is_resume_gesture,
            is_fist=is_fist,
            finger_states=finger_states,
            extended_finger_count=extended_finger_count,
            non_thumb_extended_count=non_thumb_extended_count,
            handedness=handedness,
            landmarks=hand_landmarks,
            raw_landmarks=raw_landmarks,
            clamped_landmark_count=clamped_landmark_count,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self._closed:
            return
        self._hand_landmarker.close()
        self._closed = True

    @staticmethod
    def _extract_handedness(results: Any) -> str:
        if results.handedness and results.handedness[0]:
            top_handedness = results.handedness[0][0]
            return top_handedness.category_name or top_handedness.display_name or "Unknown"
        return "Unknown"

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        if not isinstance(frame, np.ndarray):
            LOGGER.warning("Skipping non-array frame.")
            return False
        if frame.ndim != 3:
            LOGGER.warning("Skipping frame with invalid rank.")
            return False
        height, width, channels = frame.shape
        if height <= 0 or width <= 0 or channels != 3:
            LOGGER.warning("Skipping frame with invalid shape.")
            return False
        return True

    def _extract_validated_landmarks(
        self,
        detected_landmarks: list[Any],
    ) -> tuple[list[NormalizedLandmark3D], list[NormalizedPoint], int] | None:
        if len(detected_landmarks) != 21:
            LOGGER.warning("Skipping frame with unexpected landmark count.")
            return None

        validated_landmarks: list[NormalizedLandmark3D] = []
        raw_landmarks: list[NormalizedPoint] = []
        clamped_landmark_count = 0

        for landmark in detected_landmarks:
            x = getattr(landmark, "x", None)
            y = getattr(landmark, "y", None)
            validated_x, x_was_clamped = self._sanitize_normalized_value(x)
            validated_y, y_was_clamped = self._sanitize_normalized_value(y)
            if validated_x is None:
                LOGGER.warning("Skipping frame with invalid landmark x coordinate.")
                return None
            if validated_y is None:
                LOGGER.warning("Skipping frame with invalid landmark y coordinate.")
                return None
            if x_was_clamped or y_was_clamped:
                clamped_landmark_count += int(x_was_clamped) + int(y_was_clamped)
            validated_landmarks.append((validated_x, validated_y, 0.0))
            raw_landmarks.append((validated_x, validated_y))

        if clamped_landmark_count and not self._logged_projection_clamp_warning:
            LOGGER.warning(
                "Clamping slightly out-of-range MediaPipe landmark coordinates. "
                "Tracking continues, but this suggests projection overshoot near the frame edge."
            )
            self._logged_projection_clamp_warning = True

        return validated_landmarks, raw_landmarks, clamped_landmark_count

    @staticmethod
    def _sanitize_normalized_value(value: Any) -> tuple[float | None, bool]:
        if value is None:
            return None, False
        if not isinstance(value, (int, float)):
            return None, False
        numeric = float(value)
        if not isfinite(numeric):
            return None, False
        if numeric < -_NORMALIZED_COORD_TOLERANCE or numeric > 1.0 + _NORMALIZED_COORD_TOLERANCE:
            LOGGER.debug(
                "Discarding out-of-range landmark value %.4f (tolerance ±%.2f).",
                numeric,
                _NORMALIZED_COORD_TOLERANCE,
            )
            return None, False

        clamped = min(max(numeric, 0.0), 1.0)
        return clamped, clamped != numeric

    @staticmethod
    def _xy(point: NormalizedLandmark3D) -> NormalizedPoint:
        return point[0], point[1]

    @staticmethod
    def _distance(first: NormalizedPoint, second: NormalizedPoint) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])

    def _get_finger_states(
        self,
        landmarks: list[NormalizedLandmark3D],
        handedness: str,
    ) -> dict[str, bool]:
        margin = config.FINGER_UP_Y_MARGIN
        fingers = {
            "index": self._is_finger_extended(landmarks, 8, 6, 5, margin),
            "middle": self._is_finger_extended(landmarks, 12, 10, 9, margin),
            "ring": self._is_finger_extended(landmarks, 16, 14, 13, margin),
            "pinky": self._is_finger_extended(landmarks, 20, 18, 17, margin),
        }
        fingers["thumb"] = self._is_thumb_extended(landmarks, handedness)
        return fingers

    @staticmethod
    def _is_finger_extended(
        landmarks: list[NormalizedLandmark3D],
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int,
        margin: float,
    ) -> bool:
        tip_y = landmarks[tip_idx][1]
        pip_y = landmarks[pip_idx][1]
        mcp_y = landmarks[mcp_idx][1]
        return tip_y < pip_y - margin and pip_y < mcp_y

    @staticmethod
    def _is_thumb_extended(
        landmarks: list[NormalizedLandmark3D],
        handedness: str,
    ) -> bool:
        tip_x = landmarks[4][0]
        ip_x = landmarks[3][0]
        mcp_x = landmarks[2][0]
        margin = config.THUMB_OPEN_X_MARGIN

        if handedness == "Right":
            return tip_x < min(ip_x, mcp_x) - margin
        if handedness == "Left":
            return tip_x > max(ip_x, mcp_x) + margin

        return abs(tip_x - ip_x) > margin and abs(tip_x - mcp_x) > margin

    def _is_resume_gesture(
        self,
        landmarks: list[NormalizedLandmark3D],
        finger_states: dict[str, bool],
        handedness: str,
        pinch_thumb_index: float,
        pinch_thumb_middle: float,
    ) -> bool:
        non_thumb_extended = sum(
            finger_states[name] for name in ("index", "middle", "ring", "pinky")
        )
        if non_thumb_extended < 3:
            return False

        return (
            pinch_thumb_index > config.LEFT_PINCH_THRESHOLD * 1.45
            and pinch_thumb_middle > config.RIGHT_PINCH_THRESHOLD * 1.30
        )

    def _is_fist(
        self,
        landmarks: list[NormalizedLandmark3D],
        finger_states: dict[str, bool],
    ) -> bool:
        wrist = self._xy(landmarks[0])
        palm_width = self._distance(self._xy(landmarks[5]), self._xy(landmarks[17]))
        if palm_width <= 0.0:
            return False

        index_tip_to_mcp = self._distance(self._xy(landmarks[8]), self._xy(landmarks[5]))
        if (
            finger_states["index"]
            or index_tip_to_mcp / palm_width >= config.FIST_INDEX_EXTENSION_BLOCK_RATIO
        ):
            return False

        non_thumb_tip_to_mcp = (
            self._distance(self._xy(landmarks[8]), self._xy(landmarks[5])),
            self._distance(self._xy(landmarks[12]), self._xy(landmarks[9])),
            self._distance(self._xy(landmarks[16]), self._xy(landmarks[13])),
            self._distance(self._xy(landmarks[20]), self._xy(landmarks[17])),
        )
        if any(
            distance / palm_width > config.FIST_MAX_NON_THUMB_TIP_TO_MCP_RATIO
            for distance in non_thumb_tip_to_mcp
        ):
            return False

        tip_indices = (4, 8, 12, 16, 20)
        avg_tip_distance = sum(
            self._distance(self._xy(landmarks[index]), wrist) for index in tip_indices
        ) / len(tip_indices)

        curled_non_thumb = sum(
            not finger_states[name] for name in ("index", "middle", "ring", "pinky")
        )
        thumb_to_index_mcp = self._distance(self._xy(landmarks[4]), self._xy(landmarks[5]))
        thumb_curled = (
            not finger_states["thumb"]
            or thumb_to_index_mcp <= config.FIST_MAX_THUMB_TO_INDEX_MCP_DISTANCE
        )

        return (
            curled_non_thumb == 4
            and thumb_curled
            and avg_tip_distance <= config.FIST_MAX_AVERAGE_TIP_DISTANCE
        )
