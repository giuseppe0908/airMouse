"""Central configuration and validation for the AirMouse project."""

from __future__ import annotations

from pathlib import Path
import sys


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parent
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}

# Runtime mode and logging
PRODUCTION_MODE = True
ENABLE_FILE_LOGGING = False
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "airmouse.log"
LOG_FILE_MAX_BYTES = 1_000_000
LOG_BACKUP_COUNT = 1
LOG_LEVEL = "INFO"

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FLIP_HORIZONTAL = True
WINDOW_NAME = "AirMouse"
QUIT_KEY = ord("q")
ESC_KEY = 27

# MediaPipe hand tracking
HAND_LANDMARKER_MODEL_PATH = "models/hand_landmarker.task"
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.70
MIN_HAND_PRESENCE_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.65

# Active cursor area inside the webcam frame.
# Values are normalized percentages of the frame size.
ACTIVE_AREA_MARGIN_X = 0.12
ACTIVE_AREA_MARGIN_Y = 0.15

# Gesture thresholds and timings (normalized landmark distances and seconds)
LEFT_PINCH_THRESHOLD = 0.055
LEFT_PINCH_RELEASE_THRESHOLD = 0.075
LEFT_PINCH_RELEASE_GRACE_SECONDS = 0.150
RIGHT_PINCH_THRESHOLD = 0.065
PINCH_DEBOUNCE_SECONDS = 0.150
PINCH_CLICK_MIN_HOLD_SECONDS = 0.700
LEFT_PINCH_CLICK_MAX_SECONDS = 2.500
PAUSE_FIST_HOLD_SECONDS = 0.600
RESUME_OPEN_HAND_HOLD_SECONDS = 0.180

# Navigation swipe gestures (normalized landmark movement and seconds)
NAVIGATION_SWIPE_HISTORY_SIZE = 10
NAVIGATION_SWIPE_MIN_WINDOW_SECONDS = 0.050
NAVIGATION_SWIPE_MAX_WINDOW_SECONDS = 0.450
NAVIGATION_SWIPE_MIN_DISPLACEMENT = 0.075
NAVIGATION_SWIPE_MAX_ORTHOGONAL_DRIFT = 0.085
NAVIGATION_FINGER_ALIGNMENT_TOLERANCE = 0.135
NAVIGATION_FINGER_EXTENSION_MIN_RATIO = 1.05
NAVIGATION_SWIPE_SHAPE_GRACE_SECONDS = 0.080
NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS = 3
NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS = 4
NAVIGATION_SPACES_EXTENDED_FINGERS = 3
NAVIGATION_GESTURE_COOLDOWN_SECONDS = 0.850
NAVIGATION_SWIPE_SHORTCUTS = {
    "up": ("ctrl", "up"),
    "down": ("ctrl", "down"),
    "left": ("ctrl", "left"),
    "right": ("ctrl", "right"),
}

# Hand-shape heuristics
FINGER_UP_Y_MARGIN = 0.020
THUMB_OPEN_X_MARGIN = 0.040
FIST_MAX_AVERAGE_TIP_DISTANCE = 0.185
FIST_MAX_THUMB_TO_INDEX_MCP_DISTANCE = 0.165
FIST_INDEX_EXTENSION_BLOCK_RATIO = 1.05
FIST_MAX_NON_THUMB_TIP_TO_MCP_RATIO = 1.10

# Cursor smoothing.
# Alpha is the blend factor toward the target: lower values smooth more.
SMOOTHING_ALPHA_SLOW = 0.18
SMOOTHING_ALPHA_FAST = 0.72
SMOOTHING_SPEED_SLOW = 8.0
SMOOTHING_SPEED_FAST = 120.0

# pyautogui behavior
PYAUTOGUI_FAILSAFE = True
PYAUTOGUI_PAUSE_SECONDS = 0.0
SCREEN_EDGE_SAFETY_MARGIN = 8

# FPS smoothing for the debug HUD
FPS_SMOOTHING = 0.85

# HUD toggles
HUD_SHOW_LANDMARKS = True
HUD_SHOW_ACTIVE_AREA = True
HUD_SHOW_TEXT = True

# HUD layout
HUD_MARGIN = 16
HUD_LINE_HEIGHT = 26
HUD_FONT_SCALE = 0.65
HUD_FONT_THICKNESS = 2
HUD_PANEL_WIDTH = 520
HUD_PANEL_HEIGHT = 300
HUD_PANEL_OPACITY = 0.35

# BGR colors for OpenCV drawing
COLOR_PRIMARY = (40, 220, 140)
COLOR_WARNING = (0, 190, 255)
COLOR_DANGER = (0, 90, 255)
COLOR_TEXT = (240, 245, 250)
COLOR_PANEL = (20, 20, 20)
COLOR_ACTIVE_AREA = (255, 180, 40)


def _validate_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}.")


def _validate_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}.")


def _validate_float(name: str, value: object, minimum: float, maximum: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a float, got {type(value).__name__}.")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {numeric}.")


def _validate_str(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_shortcut_map(name: str, value: object) -> None:
    expected_directions = {"up", "down", "left", "right"}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict.")
    if set(value.keys()) != expected_directions:
        raise ValueError(
            f"{name} must define exactly {sorted(expected_directions)}, "
            f"got {sorted(value.keys())}."
        )

    for direction, shortcut in value.items():
        if not isinstance(shortcut, (tuple, list)) or not shortcut:
            raise ValueError(f"{name}[{direction!r}] must be a non-empty tuple/list.")
        for index, key_name in enumerate(shortcut):
            if not isinstance(key_name, str) or not key_name.strip():
                raise ValueError(
                    f"{name}[{direction!r}][{index}] must be a non-empty string."
                )


def validate_config() -> None:
    """Validate all runtime configuration before the app starts."""

    _validate_bool("PRODUCTION_MODE", PRODUCTION_MODE)
    _validate_bool("ENABLE_FILE_LOGGING", ENABLE_FILE_LOGGING)
    _validate_bool("FRAME_FLIP_HORIZONTAL", FRAME_FLIP_HORIZONTAL)
    _validate_bool("PYAUTOGUI_FAILSAFE", PYAUTOGUI_FAILSAFE)
    _validate_bool("HUD_SHOW_LANDMARKS", HUD_SHOW_LANDMARKS)
    _validate_bool("HUD_SHOW_ACTIVE_AREA", HUD_SHOW_ACTIVE_AREA)
    _validate_bool("HUD_SHOW_TEXT", HUD_SHOW_TEXT)

    _validate_str("WINDOW_NAME", WINDOW_NAME)
    _validate_str("HAND_LANDMARKER_MODEL_PATH", HAND_LANDMARKER_MODEL_PATH)
    _validate_str("LOG_DIR_NAME", LOG_DIR_NAME)
    _validate_str("LOG_FILE_NAME", LOG_FILE_NAME)
    _validate_str("LOG_LEVEL", LOG_LEVEL)
    if LOG_LEVEL.upper() not in VALID_LOG_LEVELS:
        raise ValueError(
            f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}, got {LOG_LEVEL!r}."
        )

    _validate_int("CAMERA_INDEX", CAMERA_INDEX, 0, 10)
    _validate_int("FRAME_WIDTH", FRAME_WIDTH, 320, 3840)
    _validate_int("FRAME_HEIGHT", FRAME_HEIGHT, 320, 3840)
    _validate_int("QUIT_KEY", QUIT_KEY, 0, 255)
    _validate_int("ESC_KEY", ESC_KEY, 0, 255)
    _validate_int("MAX_NUM_HANDS", MAX_NUM_HANDS, 1, 2)
    _validate_int("LOG_FILE_MAX_BYTES", LOG_FILE_MAX_BYTES, 1_024, 1_000_000)
    _validate_int("LOG_BACKUP_COUNT", LOG_BACKUP_COUNT, 0, 10)
    _validate_int("HUD_MARGIN", HUD_MARGIN, 0, 200)
    _validate_int("HUD_LINE_HEIGHT", HUD_LINE_HEIGHT, 12, 60)
    _validate_int("HUD_FONT_THICKNESS", HUD_FONT_THICKNESS, 1, 4)
    _validate_int("HUD_PANEL_WIDTH", HUD_PANEL_WIDTH, 200, 800)
    _validate_int("HUD_PANEL_HEIGHT", HUD_PANEL_HEIGHT, 120, 600)
    _validate_int("SCREEN_EDGE_SAFETY_MARGIN", SCREEN_EDGE_SAFETY_MARGIN, 0, 100)
    _validate_int("NAVIGATION_SWIPE_HISTORY_SIZE", NAVIGATION_SWIPE_HISTORY_SIZE, 3, 60)
    _validate_int(
        "NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS",
        NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS,
        1,
        4,
    )
    _validate_int(
        "NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS",
        NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS,
        1,
        4,
    )
    _validate_int(
        "NAVIGATION_SPACES_EXTENDED_FINGERS",
        NAVIGATION_SPACES_EXTENDED_FINGERS,
        1,
        4,
    )

    _validate_float("MIN_DETECTION_CONFIDENCE", MIN_DETECTION_CONFIDENCE, 0.10, 1.0)
    _validate_float(
        "MIN_HAND_PRESENCE_CONFIDENCE",
        MIN_HAND_PRESENCE_CONFIDENCE,
        0.10,
        1.0,
    )
    _validate_float("MIN_TRACKING_CONFIDENCE", MIN_TRACKING_CONFIDENCE, 0.10, 1.0)
    _validate_float("ACTIVE_AREA_MARGIN_X", ACTIVE_AREA_MARGIN_X, 0.0, 0.45)
    _validate_float("ACTIVE_AREA_MARGIN_Y", ACTIVE_AREA_MARGIN_Y, 0.0, 0.45)
    _validate_float("LEFT_PINCH_THRESHOLD", LEFT_PINCH_THRESHOLD, 0.01, 0.20)
    _validate_float("LEFT_PINCH_RELEASE_THRESHOLD", LEFT_PINCH_RELEASE_THRESHOLD, 0.01, 0.25)
    _validate_float(
        "LEFT_PINCH_RELEASE_GRACE_SECONDS",
        LEFT_PINCH_RELEASE_GRACE_SECONDS,
        0.0,
        0.30,
    )
    _validate_float("RIGHT_PINCH_THRESHOLD", RIGHT_PINCH_THRESHOLD, 0.01, 0.20)
    _validate_float("PINCH_DEBOUNCE_SECONDS", PINCH_DEBOUNCE_SECONDS, 0.05, 1.0)
    _validate_float(
        "PINCH_CLICK_MIN_HOLD_SECONDS",
        PINCH_CLICK_MIN_HOLD_SECONDS,
        0.05,
        1.0,
    )
    _validate_float(
        "LEFT_PINCH_CLICK_MAX_SECONDS",
        LEFT_PINCH_CLICK_MAX_SECONDS,
        0.05,
        5.0,
    )
    _validate_float("PAUSE_FIST_HOLD_SECONDS", PAUSE_FIST_HOLD_SECONDS, 0.05, 2.0)
    _validate_float("RESUME_OPEN_HAND_HOLD_SECONDS", RESUME_OPEN_HAND_HOLD_SECONDS, 0.05, 2.0)
    _validate_float(
        "NAVIGATION_SWIPE_MIN_WINDOW_SECONDS",
        NAVIGATION_SWIPE_MIN_WINDOW_SECONDS,
        0.0,
        1.0,
    )
    _validate_float(
        "NAVIGATION_SWIPE_MAX_WINDOW_SECONDS",
        NAVIGATION_SWIPE_MAX_WINDOW_SECONDS,
        0.05,
        2.0,
    )
    _validate_float(
        "NAVIGATION_SWIPE_MIN_DISPLACEMENT",
        NAVIGATION_SWIPE_MIN_DISPLACEMENT,
        0.01,
        1.0,
    )
    _validate_float(
        "NAVIGATION_SWIPE_MAX_ORTHOGONAL_DRIFT",
        NAVIGATION_SWIPE_MAX_ORTHOGONAL_DRIFT,
        0.0,
        1.0,
    )
    _validate_float(
        "NAVIGATION_FINGER_ALIGNMENT_TOLERANCE",
        NAVIGATION_FINGER_ALIGNMENT_TOLERANCE,
        0.0,
        1.0,
    )
    _validate_float(
        "NAVIGATION_FINGER_EXTENSION_MIN_RATIO",
        NAVIGATION_FINGER_EXTENSION_MIN_RATIO,
        0.50,
        3.0,
    )
    _validate_float(
        "NAVIGATION_SWIPE_SHAPE_GRACE_SECONDS",
        NAVIGATION_SWIPE_SHAPE_GRACE_SECONDS,
        0.0,
        0.25,
    )
    _validate_float(
        "NAVIGATION_GESTURE_COOLDOWN_SECONDS",
        NAVIGATION_GESTURE_COOLDOWN_SECONDS,
        0.10,
        5.0,
    )
    _validate_float("FINGER_UP_Y_MARGIN", FINGER_UP_Y_MARGIN, 0.0, 0.20)
    _validate_float("THUMB_OPEN_X_MARGIN", THUMB_OPEN_X_MARGIN, 0.0, 0.20)
    _validate_float(
        "FIST_MAX_AVERAGE_TIP_DISTANCE",
        FIST_MAX_AVERAGE_TIP_DISTANCE,
        0.01,
        1.0,
    )
    _validate_float(
        "FIST_MAX_THUMB_TO_INDEX_MCP_DISTANCE",
        FIST_MAX_THUMB_TO_INDEX_MCP_DISTANCE,
        0.01,
        1.0,
    )
    _validate_float(
        "FIST_INDEX_EXTENSION_BLOCK_RATIO",
        FIST_INDEX_EXTENSION_BLOCK_RATIO,
        0.10,
        3.0,
    )
    _validate_float(
        "FIST_MAX_NON_THUMB_TIP_TO_MCP_RATIO",
        FIST_MAX_NON_THUMB_TIP_TO_MCP_RATIO,
        0.10,
        3.0,
    )
    _validate_float("SMOOTHING_ALPHA_SLOW", SMOOTHING_ALPHA_SLOW, 0.01, 1.0)
    _validate_float("SMOOTHING_ALPHA_FAST", SMOOTHING_ALPHA_FAST, 0.01, 1.0)
    _validate_float("SMOOTHING_SPEED_SLOW", SMOOTHING_SPEED_SLOW, 1.0, 10.0)
    _validate_float("SMOOTHING_SPEED_FAST", SMOOTHING_SPEED_FAST, 10.0, 1000.0)
    _validate_float("PYAUTOGUI_PAUSE_SECONDS", PYAUTOGUI_PAUSE_SECONDS, 0.0, 1.0)
    _validate_float("FPS_SMOOTHING", FPS_SMOOTHING, 0.0, 0.99)
    _validate_float("HUD_FONT_SCALE", HUD_FONT_SCALE, 0.30, 2.0)
    _validate_float("HUD_PANEL_OPACITY", HUD_PANEL_OPACITY, 0.0, 1.0)
    _validate_shortcut_map("NAVIGATION_SWIPE_SHORTCUTS", NAVIGATION_SWIPE_SHORTCUTS)

    if not PYAUTOGUI_FAILSAFE:
        raise ValueError("PYAUTOGUI_FAILSAFE must remain True in production.")

    if PINCH_CLICK_MIN_HOLD_SECONDS < PINCH_DEBOUNCE_SECONDS:
        raise ValueError(
            "PINCH_CLICK_MIN_HOLD_SECONDS must be greater than or equal to "
            "PINCH_DEBOUNCE_SECONDS."
        )

    if LEFT_PINCH_RELEASE_THRESHOLD <= LEFT_PINCH_THRESHOLD:
        raise ValueError(
            "LEFT_PINCH_RELEASE_THRESHOLD must be greater than LEFT_PINCH_THRESHOLD."
        )

    if LEFT_PINCH_CLICK_MAX_SECONDS <= PINCH_CLICK_MIN_HOLD_SECONDS:
        raise ValueError(
            "LEFT_PINCH_CLICK_MAX_SECONDS must be greater than "
            "PINCH_CLICK_MIN_HOLD_SECONDS."
        )

    if NAVIGATION_SWIPE_MIN_WINDOW_SECONDS >= NAVIGATION_SWIPE_MAX_WINDOW_SECONDS:
        raise ValueError(
            "NAVIGATION_SWIPE_MIN_WINDOW_SECONDS must be less than "
            "NAVIGATION_SWIPE_MAX_WINDOW_SECONDS."
        )

    if NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS > NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS:
        raise ValueError(
            "NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS must be less than or equal to "
            "NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS."
        )

    if not (
        NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS
        <= NAVIGATION_SPACES_EXTENDED_FINGERS
        <= NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS
    ):
        raise ValueError(
            "NAVIGATION_SPACES_EXTENDED_FINGERS must be within the configured "
            "navigation swipe finger range."
        )

    if SMOOTHING_ALPHA_SLOW > SMOOTHING_ALPHA_FAST:
        raise ValueError(
            "SMOOTHING_ALPHA_SLOW must be less than or equal to SMOOTHING_ALPHA_FAST."
        )

    if SMOOTHING_SPEED_SLOW >= SMOOTHING_SPEED_FAST:
        raise ValueError(
            "SMOOTHING_SPEED_SLOW must be strictly less than SMOOTHING_SPEED_FAST."
        )

    margin_x_pixels = int(round(ACTIVE_AREA_MARGIN_X * FRAME_WIDTH))
    margin_y_pixels = int(round(ACTIVE_AREA_MARGIN_Y * FRAME_HEIGHT))
    if not 0 <= margin_x_pixels <= 400:
        raise ValueError(
            f"ACTIVE_AREA_MARGIN_X resolves to {margin_x_pixels}px; it must stay between 0 and 400px."
        )
    if not 0 <= margin_y_pixels <= 400:
        raise ValueError(
            f"ACTIVE_AREA_MARGIN_Y resolves to {margin_y_pixels}px; it must stay between 0 and 400px."
        )

    model_path = (PROJECT_ROOT / HAND_LANDMARKER_MODEL_PATH).resolve()
    if not model_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("HAND_LANDMARKER_MODEL_PATH must stay inside the project directory.")
