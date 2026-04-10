"""End-to-end hand gesture to mouse control state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from math import hypot
import time
from typing import Any

import config
from hand_tracker import HandState, HandTracker
from mouse_controller import MouseControlError, MouseControlSafetyStop, MouseController


LOGGER = logging.getLogger("airmouse.controller")

NO_HAND = "NO_HAND"
ACTIVE_TRACKING = "ACTIVE_TRACKING"
TRACKING_PAUSED = "TRACKING_PAUSED"
LEFT_PINCH_PENDING = "LEFT_PINCH_PENDING"
LEFT_CLICK = "LEFT_CLICK"
DRAGGING = "DRAGGING"
RIGHT_PINCH_PENDING = "RIGHT_PINCH_PENDING"
RIGHT_CLICK = "RIGHT_CLICK"
FIST_PENDING = "FIST_PENDING"
RESUME_PENDING = "RESUME_PENDING"
NAVIGATION_SWIPE_UP = "NAVIGATION_SWIPE_UP"
NAVIGATION_SWIPE_DOWN = "NAVIGATION_SWIPE_DOWN"
NAVIGATION_SWIPE_LEFT = "NAVIGATION_SWIPE_LEFT"
NAVIGATION_SWIPE_RIGHT = "NAVIGATION_SWIPE_RIGHT"
PINCH_LOG_MARGIN = 0.020

_NAVIGATION_STATE_BY_DIRECTION = {
    "up": NAVIGATION_SWIPE_UP,
    "down": NAVIGATION_SWIPE_DOWN,
    "left": NAVIGATION_SWIPE_LEFT,
    "right": NAVIGATION_SWIPE_RIGHT,
}

_NON_THUMB_TIP_INDICES = {
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
_NON_THUMB_MCP_INDICES = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}
_NON_THUMB_PIP_INDICES = {
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}
_PALM_CENTER_INDICES = (0, 5, 9, 13, 17)


@dataclass(slots=True)
class _NavigationGestureSample:
    timestamp: float
    center: tuple[float, float]


class NavigationSwipeDetector:
    """High-confidence, one-shot navigation swipe detector."""

    def __init__(self) -> None:
        self._history: deque[_NavigationGestureSample] = deque(
            maxlen=config.NAVIGATION_SWIPE_HISTORY_SIZE
        )
        self._cooldown_until = 0.0
        self._last_candidate = "NONE"
        self._last_detected: str | None = None
        self._last_delta = (0.0, 0.0)
        self._last_finger_alignment: float | None = None
        self._last_alignment_axis = "--"
        self._last_extended_fingers = 0
        self._last_navigation_finger_states: dict[str, bool] = {}
        self._last_history_size = 0
        self._last_history_duration = 0.0
        self._last_vertical_shape = False
        self._last_horizontal_shape = False
        self._last_shape_seen_at: float | None = None
        self._pending = False

    def reset(self) -> None:
        self._history.clear()
        self._last_candidate = "NONE"
        self._last_detected = None
        self._last_delta = (0.0, 0.0)
        self._last_finger_alignment = None
        self._last_alignment_axis = "--"
        self._last_extended_fingers = 0
        self._last_navigation_finger_states = {}
        self._last_history_size = 0
        self._last_history_duration = 0.0
        self._last_vertical_shape = False
        self._last_horizontal_shape = False
        self._last_shape_seen_at = None
        self._pending = False

    def update(self, now: float, hand_state: HandState) -> str | None:
        self._last_detected = None
        self._last_delta = (0.0, 0.0)
        navigation_fingers = self._navigation_finger_states(hand_state)
        self._last_navigation_finger_states = navigation_fingers
        self._last_extended_fingers = sum(navigation_fingers.values())
        self._last_finger_alignment, self._last_alignment_axis = (
            self._finger_alignment_span(hand_state, navigation_fingers)
        )
        self._pending = False

        if self._cooldown_remaining(now) > 0.0:
            self._history.clear()
            self._last_candidate = "COOLDOWN"
            self._last_history_size = 0
            self._last_history_duration = 0.0
            return None

        vertical_shape = self._is_vertical_swipe_shape(navigation_fingers)
        horizontal_shape = self._is_horizontal_swipe_shape(navigation_fingers)
        self._last_vertical_shape = vertical_shape
        self._last_horizontal_shape = horizontal_shape
        if not vertical_shape and not horizontal_shape:
            if self._within_shape_grace(now):
                self._pending = bool(self._history)
                self._last_candidate = "SHAPE_GAP"
                self._update_history_debug(now)
                return None

            self._clear_history_debug()
            self._last_candidate = "NONE"
            return None

        self._last_shape_seen_at = now
        center = self._palm_center(hand_state)
        self._history.append(_NavigationGestureSample(now, center))
        self._trim_history(now)
        self._update_history_debug(now)
        self._pending = True
        self._last_candidate = "SHAPE_OK"

        if len(self._history) < 2:
            return None

        first_sample = self._history[0]
        duration = now - first_sample.timestamp
        dx = center[0] - first_sample.center[0]
        dy = center[1] - first_sample.center[1]
        self._last_delta = (dx, dy)
        self._last_candidate = self._candidate_from_delta(dx, dy, vertical_shape, horizontal_shape)

        if duration < config.NAVIGATION_SWIPE_MIN_WINDOW_SECONDS:
            return None

        direction = self._detect_direction(dx, dy, vertical_shape, horizontal_shape)
        if direction is None:
            return None

        detected = f"SWIPE_{direction.upper()}"
        self._cooldown_until = now + config.NAVIGATION_GESTURE_COOLDOWN_SECONDS
        self._clear_history_debug()
        self._pending = False
        self._last_candidate = detected
        self._last_detected = detected
        return direction

    def is_motion_candidate_active(self) -> bool:
        if not self._pending:
            return False
        dx, dy = self._last_delta
        return max(abs(dx), abs(dy)) >= config.NAVIGATION_SWIPE_MIN_DISPLACEMENT * 0.35

    def get_debug_state(self, now: float) -> dict[str, Any]:
        cooldown_remaining = self._cooldown_remaining(now)
        candidate = self._last_candidate
        if cooldown_remaining > 0.0 and candidate == "NONE":
            candidate = "COOLDOWN"
        return {
            "navigation_candidate": candidate,
            "navigation_pending": self._pending,
            "navigation_detected": (
                self._last_detected if cooldown_remaining > 0.0 else None
            ),
            "navigation_cooldown_remaining": cooldown_remaining,
            "navigation_delta": self._last_delta,
            "navigation_finger_alignment": self._last_finger_alignment,
            "navigation_alignment_axis": self._last_alignment_axis,
            "navigation_extended_fingers": self._last_extended_fingers,
            "navigation_finger_states": dict(self._last_navigation_finger_states),
            "navigation_history_size": self._last_history_size,
            "navigation_history_duration": self._last_history_duration,
            "navigation_vertical_shape": self._last_vertical_shape,
            "navigation_horizontal_shape": self._last_horizontal_shape,
        }

    def _cooldown_remaining(self, now: float) -> float:
        return max(0.0, self._cooldown_until - now)

    def _trim_history(self, now: float) -> None:
        while (
            self._history
            and now - self._history[0].timestamp
            > config.NAVIGATION_SWIPE_MAX_WINDOW_SECONDS
        ):
            self._history.popleft()

    def _within_shape_grace(self, now: float) -> bool:
        return (
            self._history
            and self._last_shape_seen_at is not None
            and now - self._last_shape_seen_at
            <= config.NAVIGATION_SWIPE_SHAPE_GRACE_SECONDS
        )

    def _clear_history_debug(self) -> None:
        self._history.clear()
        self._last_history_size = 0
        self._last_history_duration = 0.0

    def _update_history_debug(self, now: float) -> None:
        self._last_history_size = len(self._history)
        if not self._history:
            self._last_history_duration = 0.0
            return
        self._last_history_duration = now - self._history[0].timestamp

    def _is_vertical_swipe_shape(self, navigation_fingers: dict[str, bool]) -> bool:
        return (
            config.NAVIGATION_SWIPE_MIN_EXTENDED_FINGERS
            <= sum(navigation_fingers.values())
            <= config.NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS
            and self._has_primary_three_fingers(navigation_fingers)
            and self._is_aligned()
        )

    def _is_horizontal_swipe_shape(self, navigation_fingers: dict[str, bool]) -> bool:
        return (
            config.NAVIGATION_SPACES_EXTENDED_FINGERS
            <= sum(navigation_fingers.values())
            <= config.NAVIGATION_SWIPE_MAX_EXTENDED_FINGERS
            and self._has_primary_three_fingers(navigation_fingers)
            and self._is_aligned()
        )

    @staticmethod
    def _has_primary_three_fingers(navigation_fingers: dict[str, bool]) -> bool:
        return all(navigation_fingers[name] for name in ("index", "middle", "ring"))

    def _is_aligned(self) -> bool:
        return (
            self._last_finger_alignment is not None
            and self._last_finger_alignment <= config.NAVIGATION_FINGER_ALIGNMENT_TOLERANCE
        )

    def _finger_alignment_span(
        self,
        hand_state: HandState,
        navigation_fingers: dict[str, bool],
    ) -> tuple[float, str]:
        fingertip_x_values = [
            hand_state.landmarks[tip_index][0]
            for finger_name, tip_index in _NON_THUMB_TIP_INDICES.items()
            if navigation_fingers[finger_name]
        ]
        fingertip_y_values = [
            hand_state.landmarks[tip_index][1]
            for finger_name, tip_index in _NON_THUMB_TIP_INDICES.items()
            if navigation_fingers[finger_name]
        ]
        if len(fingertip_y_values) < 2:
            return 1.0, "--"

        x_span = max(fingertip_x_values) - min(fingertip_x_values)
        y_span = max(fingertip_y_values) - min(fingertip_y_values)
        if y_span <= x_span:
            return y_span, "y"
        return x_span, "x"

    def _navigation_finger_states(self, hand_state: HandState) -> dict[str, bool]:
        palm_width = self._distance(
            self._xy(hand_state.landmarks[5]),
            self._xy(hand_state.landmarks[17]),
        )
        if palm_width <= 0.0:
            return {
                name: hand_state.finger_states[name]
                for name in _NON_THUMB_TIP_INDICES
            }

        navigation_fingers: dict[str, bool] = {}
        for name, tip_index in _NON_THUMB_TIP_INDICES.items():
            mcp_index = _NON_THUMB_MCP_INDICES[name]
            pip_index = _NON_THUMB_PIP_INDICES[name]
            tip_to_mcp = self._distance(
                self._xy(hand_state.landmarks[tip_index]),
                self._xy(hand_state.landmarks[mcp_index]),
            )
            pip_to_mcp = self._distance(
                self._xy(hand_state.landmarks[pip_index]),
                self._xy(hand_state.landmarks[mcp_index]),
            )
            navigation_fingers[name] = (
                hand_state.finger_states[name]
                or (
                    tip_to_mcp / palm_width
                    >= config.NAVIGATION_FINGER_EXTENSION_MIN_RATIO
                    and tip_to_mcp > pip_to_mcp * 1.20
                )
            )

        return navigation_fingers

    @staticmethod
    def _palm_center(hand_state: HandState) -> tuple[float, float]:
        x = sum(hand_state.landmarks[index][0] for index in _PALM_CENTER_INDICES)
        y = sum(hand_state.landmarks[index][1] for index in _PALM_CENTER_INDICES)
        return x / len(_PALM_CENTER_INDICES), y / len(_PALM_CENTER_INDICES)

    @staticmethod
    def _xy(point: tuple[float, float, float]) -> tuple[float, float]:
        return point[0], point[1]

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _candidate_from_delta(
        dx: float,
        dy: float,
        vertical_shape: bool,
        horizontal_shape: bool,
    ) -> str:
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return "SHAPE_OK"
        if vertical_shape and abs(dy) >= abs(dx):
            return "UP_PENDING" if dy < 0.0 else "DOWN_PENDING"
        if horizontal_shape:
            return "LEFT_PENDING" if dx < 0.0 else "RIGHT_PENDING"
        return "SHAPE_OK"

    @staticmethod
    def _detect_direction(
        dx: float,
        dy: float,
        vertical_shape: bool,
        horizontal_shape: bool,
    ) -> str | None:
        min_displacement = config.NAVIGATION_SWIPE_MIN_DISPLACEMENT
        max_drift = config.NAVIGATION_SWIPE_MAX_ORTHOGONAL_DRIFT

        if (
            vertical_shape
            and abs(dy) >= min_displacement
            and abs(dx) <= max_drift
        ):
            return "up" if dy < 0.0 else "down"

        if (
            horizontal_shape
            and abs(dx) >= min_displacement
            and abs(dy) <= max_drift
        ):
            return "left" if dx < 0.0 else "right"

        return None


class HandMouseController:
    """Processes webcam frames and turns gestures into mouse actions."""

    def __init__(self) -> None:
        self.hand_tracker = HandTracker()
        self.mouse_controller = MouseController()

        self.latest_hand_state: HandState | None = None
        self._gesture_state = TRACKING_PAUSED
        self._tracking_active = False
        self._mouse_down = False

        self._left_pinch_started_at: float | None = None
        self._left_pinch_gap_started_at: float | None = None
        self._right_pinch_started_at: float | None = None
        self._right_click_latched = False
        self._fist_started_at: float | None = None
        self._resume_started_at: float | None = None
        self._last_left_pinching = False
        self._last_right_pinching = False
        self._left_click_fired = False
        self._last_navigation_shortcut = None
        self._last_navigation_shortcut_fired = False
        self._left_pinch_near_logged = False
        self._right_pinch_near_logged = False

        self._navigation_detector = NavigationSwipeDetector()
        self._cursor_position = self.mouse_controller.get_position()
        self._fps = 0.0
        self._last_update_at = time.perf_counter()
        self._debug_state: dict[str, Any] = {}
        self._refresh_debug_state()

    def update(self, frame: Any) -> None:
        """Process a frame end to end."""

        now = time.perf_counter()
        self._update_fps(now)
        self._left_click_fired = False
        hand_state = self.hand_tracker.process(frame)
        self.latest_hand_state = hand_state

        if hand_state is None:
            self._handle_missing_hand()
            self._refresh_debug_state()
            return

        self._log_pinch_proximity(hand_state)

        if not self._tracking_active:
            self._handle_paused_state(now, hand_state)
            self._refresh_debug_state()
            return

        self._handle_active_state(now, hand_state)
        self._refresh_debug_state()

    def _handle_missing_hand(self) -> None:
        if self._left_pinch_started_at is not None or self._right_pinch_started_at is not None:
            LOGGER.info(
                "Pinch cancelled by tracking loss: left_pending=%s right_pending=%s drag=%s",
                self._left_pinch_started_at is not None,
                self._right_pinch_started_at is not None,
                self._mouse_down,
            )

        self._left_pinch_near_logged = False
        self._right_pinch_near_logged = False
        self._last_left_pinching = False
        self._last_right_pinching = False
        self._fist_started_at = None
        self._navigation_detector.reset()

        if not self._tracking_active and self._gesture_state == RESUME_PENDING:
            self._gesture_state = RESUME_PENDING
        elif self._tracking_active:
            self._resume_started_at = None
            self._gesture_state = NO_HAND
        else:
            self._resume_started_at = None
            self._gesture_state = TRACKING_PAUSED

        self._cancel_pending_actions(release_mouse=True)

    def _handle_paused_state(self, now: float, hand_state: HandState) -> None:
        """In paused mode, only a stable open-hand system gesture can resume."""

        self._fist_started_at = None
        self._cancel_pending_actions(release_mouse=True)
        self._navigation_detector.reset()
        self._last_left_pinching = False
        self._last_right_pinching = False

        if not self._is_resume_signal(hand_state):
            self._resume_started_at = None
            self._gesture_state = TRACKING_PAUSED
            return

        if self._resume_started_at is None:
            self._resume_started_at = now
            self._gesture_state = RESUME_PENDING
            LOGGER.info("Resume gesture detected: hold open hand to activate tracking.")
            return

        if now - self._resume_started_at < config.RESUME_OPEN_HAND_HOLD_SECONDS:
            self._gesture_state = RESUME_PENDING
            return

        self._tracking_active = True
        self._resume_started_at = None
        self._gesture_state = ACTIVE_TRACKING
        LOGGER.info("Tracking activated by open-hand gesture.")

    def _handle_active_state(self, now: float, hand_state: HandState) -> None:
        """In active mode, point with index finger; only a stable fist pauses."""

        self._resume_started_at = None

        if hand_state.is_fist:
            self._navigation_detector.reset()
            if self._fist_started_at is None:
                self._fist_started_at = now
                self._gesture_state = FIST_PENDING
                return

            if now - self._fist_started_at < config.PAUSE_FIST_HOLD_SECONDS:
                self._gesture_state = FIST_PENDING
                return

            self._tracking_active = False
            self._cancel_pending_actions(release_mouse=True)
            self._gesture_state = TRACKING_PAUSED
            LOGGER.info("Tracking paused by fist gesture.")
            return

        self._fist_started_at = None

        try:
            self._left_click_fired = False
            self._cursor_position = self.mouse_controller.move(*hand_state.index_tip)
            left_pinching = self._is_left_pinching(hand_state)
            right_pinching = hand_state.pinch_thumb_middle <= config.RIGHT_PINCH_THRESHOLD
            self._last_left_pinching = left_pinching
            self._last_right_pinching = right_pinching

            pinch_state_active = (
                self._mouse_down
                or self._left_pinch_started_at is not None
                or left_pinching
                or self._right_pinch_started_at is not None
                or self._right_click_latched
            )

            if pinch_state_active:
                self._navigation_detector.reset()
            else:
                navigation_direction = self._navigation_detector.update(now, hand_state)
                if navigation_direction is not None:
                    self._dispatch_navigation_gesture(navigation_direction)
                    self._gesture_state = _NAVIGATION_STATE_BY_DIRECTION[navigation_direction]
                    return

            navigation_motion_candidate = self._navigation_detector.is_motion_candidate_active()
            if navigation_motion_candidate and not pinch_state_active:
                self._gesture_state = ACTIVE_TRACKING
                return

            if self._mouse_down or self._left_pinch_started_at is not None or left_pinching:
                self._handle_left_pinch(now, left_pinching)
            elif (
                self._right_pinch_started_at is not None
                or self._right_click_latched
                or right_pinching
            ):
                self._handle_right_pinch(now, right_pinching)
            else:
                self._gesture_state = ACTIVE_TRACKING
        except MouseControlSafetyStop:
            self._tracking_active = False
            self._gesture_state = TRACKING_PAUSED
            self._cancel_pending_actions(release_mouse=True)
            self._refresh_debug_state()
            raise
        except MouseControlError:
            self._tracking_active = False
            self._gesture_state = TRACKING_PAUSED
            self._cancel_pending_actions(release_mouse=True)
            self._refresh_debug_state()
            raise

    def _dispatch_navigation_gesture(self, direction: str) -> None:
        shortcut = tuple(config.NAVIGATION_SWIPE_SHORTCUTS[direction])
        LOGGER.info(
            "Navigation swipe detected: direction=%s shortcut=%s",
            direction,
            "+".join(shortcut),
        )
        self._last_navigation_shortcut = "+".join(shortcut)
        self._last_navigation_shortcut_fired = self.mouse_controller.keyboard_shortcut(shortcut)

    def _is_left_pinching(self, hand_state: HandState) -> bool:
        return hand_state.pinch_thumb_index <= self._left_pinch_active_threshold()

    def _left_pinch_active_threshold(self) -> float:
        if self._mouse_down or self._left_pinch_started_at is not None:
            return config.LEFT_PINCH_RELEASE_THRESHOLD
        return config.LEFT_PINCH_THRESHOLD

    @staticmethod
    def _is_resume_signal(hand_state: HandState) -> bool:
        return hand_state.is_open_hand or hand_state.is_resume_gesture

    def get_cursor_position(self) -> tuple[int, int]:
        return self._cursor_position

    def is_clicking(self) -> bool:
        return self._mouse_down

    def get_debug_state(self) -> dict[str, Any]:
        return dict(self._debug_state)

    def release_if_needed(self) -> None:
        """Release any held mouse button and close tracker resources."""

        self._cancel_pending_actions(release_mouse=True)
        self.hand_tracker.close()
        self._refresh_debug_state()

    def _handle_left_pinch(self, now: float, left_pinching: bool) -> None:
        self._right_pinch_started_at = None
        self._right_click_latched = False

        if left_pinching:
            self._left_pinch_gap_started_at = None
            if self._left_pinch_started_at is None:
                self._left_pinch_started_at = now
                self._gesture_state = LEFT_PINCH_PENDING
                LOGGER.info(
                    "Left pinch started: pinch=%.3f cursor=%s",
                    self._pinch_metric("left"),
                    self._cursor_position,
                )
                return

            hold_duration = now - self._left_pinch_started_at
            if hold_duration >= config.LEFT_PINCH_CLICK_MAX_SECONDS:
                if not self._mouse_down:
                    LOGGER.info(
                        "Left drag start: hold=%.3fs pinch=%.3f cursor=%s",
                        hold_duration,
                        self._pinch_metric("left"),
                        self._cursor_position,
                    )
                    self.mouse_controller.mouse_down()
                    self._mouse_down = True
                self._gesture_state = DRAGGING
            else:
                self._gesture_state = LEFT_PINCH_PENDING
            return

        if self._mouse_down:
            LOGGER.info(
                "Left drag end: cursor=%s",
                self._cursor_position,
            )
            self.mouse_controller.mouse_up()
            self._mouse_down = False
            self._gesture_state = ACTIVE_TRACKING
        elif self._left_pinch_started_at is not None:
            if self._left_pinch_gap_started_at is None:
                self._left_pinch_gap_started_at = now
                self._gesture_state = LEFT_PINCH_PENDING
                return

            if now - self._left_pinch_gap_started_at < config.LEFT_PINCH_RELEASE_GRACE_SECONDS:
                self._gesture_state = LEFT_PINCH_PENDING
                return

            hold_duration = self._left_pinch_gap_started_at - self._left_pinch_started_at
            if (
                config.PINCH_CLICK_MIN_HOLD_SECONDS
                <= hold_duration
                < config.LEFT_PINCH_CLICK_MAX_SECONDS
            ):
                LOGGER.info(
                    "Left click requested: hold=%.3fs pinch=%.3f cursor=%s",
                    hold_duration,
                    self._pinch_metric("left"),
                    self._cursor_position,
                )
                self.mouse_controller.left_click()
                self._left_click_fired = True
                self._gesture_state = LEFT_CLICK
            else:
                LOGGER.info(
                    "Left pinch cancelled: hold=%.3fs pinch=%.3f cursor=%s",
                    hold_duration,
                    self._pinch_metric("left"),
                    self._cursor_position,
                )
                self._gesture_state = ACTIVE_TRACKING
        else:
            self._gesture_state = ACTIVE_TRACKING

        self._left_pinch_started_at = None
        self._left_pinch_gap_started_at = None

    def _handle_right_pinch(self, now: float, right_pinching: bool) -> None:
        self._left_pinch_started_at = None
        self._left_pinch_gap_started_at = None

        if right_pinching:
            if self._right_click_latched:
                self._gesture_state = RIGHT_CLICK
                return

            if self._right_pinch_started_at is None:
                self._right_pinch_started_at = now
                self._gesture_state = RIGHT_PINCH_PENDING
                LOGGER.info(
                    "Right pinch started: pinch=%.3f cursor=%s",
                    self._pinch_metric("right"),
                    self._cursor_position,
                )
                return

            hold_duration = now - self._right_pinch_started_at
            if hold_duration >= config.PINCH_CLICK_MIN_HOLD_SECONDS:
                LOGGER.info(
                    "Right click requested: hold=%.3fs pinch=%.3f cursor=%s",
                    hold_duration,
                    self._pinch_metric("right"),
                    self._cursor_position,
                )
                self.mouse_controller.right_click()
                self._right_click_latched = True
                self._gesture_state = RIGHT_CLICK
            else:
                self._gesture_state = RIGHT_PINCH_PENDING
            return

        if self._right_pinch_started_at is not None and not self._right_click_latched:
            LOGGER.info(
                "Right pinch cancelled: hold=%.3fs pinch=%.3f cursor=%s",
                now - self._right_pinch_started_at,
                self._pinch_metric("right"),
                self._cursor_position,
            )
        self._right_pinch_started_at = None
        self._right_click_latched = False
        self._gesture_state = ACTIVE_TRACKING

    def _cancel_pending_actions(self, release_mouse: bool) -> None:
        self._left_pinch_started_at = None
        self._left_pinch_gap_started_at = None
        self._right_pinch_started_at = None
        self._right_click_latched = False

        if release_mouse and self._mouse_down:
            try:
                self.mouse_controller.mouse_up()
            except (MouseControlSafetyStop, MouseControlError):
                self.mouse_controller.emergency_release()
                LOGGER.warning("Forced emergency mouse release during cleanup.")
            self._mouse_down = False

    def _update_fps(self, now: float) -> None:
        dt = now - self._last_update_at
        self._last_update_at = now

        if dt <= 0:
            return

        instantaneous_fps = 1.0 / dt
        if self._fps == 0.0:
            self._fps = instantaneous_fps
            return

        self._fps = (
            config.FPS_SMOOTHING * self._fps
            + (1.0 - config.FPS_SMOOTHING) * instantaneous_fps
        )

    def _refresh_debug_state(self) -> None:
        now = time.perf_counter()
        left_pinch_gap_elapsed = (
            None
            if self._left_pinch_gap_started_at is None
            else now - self._left_pinch_gap_started_at
        )
        left_pinch_measure_at = self._left_pinch_gap_started_at or now
        left_pinch_hold_elapsed = (
            None
            if self._left_pinch_started_at is None
            else left_pinch_measure_at - self._left_pinch_started_at
        )
        left_pinch_click_eligible = (
            left_pinch_hold_elapsed is not None
            and config.PINCH_CLICK_MIN_HOLD_SECONDS
            <= left_pinch_hold_elapsed
            < config.LEFT_PINCH_CLICK_MAX_SECONDS
        )

        debug_state = {
            "gesture": self._gesture_state,
            "tracking_state": "ACTIVE" if self._tracking_active else "PAUSED",
            "mouse_state": "DOWN" if self._mouse_down else "UP",
            "index_tip": (
                None if self.latest_hand_state is None else self.latest_hand_state.index_tip
            ),
            "handedness": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.handedness
            ),
            "resume_gesture": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.is_resume_gesture
            ),
            "finger_states": (
                {}
                if self.latest_hand_state is None
                else dict(self.latest_hand_state.finger_states)
            ),
            "extended_finger_count": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.extended_finger_count
            ),
            "non_thumb_extended_count": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.non_thumb_extended_count
            ),
            "clamped_landmark_count": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.clamped_landmark_count
            ),
            "pinch_thumb_index": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.pinch_thumb_index
            ),
            "pinch_thumb_middle": (
                None
                if self.latest_hand_state is None
                else self.latest_hand_state.pinch_thumb_middle
            ),
            "left_pinching": (
                None if self.latest_hand_state is None else self._last_left_pinching
            ),
            "left_pinch_active": (
                None if self.latest_hand_state is None else self._last_left_pinching
            ),
            "right_pinching": (
                None if self.latest_hand_state is None else self._last_right_pinching
            ),
            "left_pinch_hold_elapsed": left_pinch_hold_elapsed,
            "left_pinch_gap_elapsed": left_pinch_gap_elapsed,
            "left_pinch_release_grace_remaining": (
                None
                if left_pinch_gap_elapsed is None
                else max(0.0, config.LEFT_PINCH_RELEASE_GRACE_SECONDS - left_pinch_gap_elapsed)
            ),
            "left_pinch_click_eligible": left_pinch_click_eligible,
            "left_pinch_click_armed": left_pinch_click_eligible,
            "left_pinch_click_fired": self._left_click_fired,
            "left_pinch_drag_ready": (
                left_pinch_hold_elapsed is not None
                and left_pinch_hold_elapsed >= config.LEFT_PINCH_CLICK_MAX_SECONDS
            ),
            "left_pinch_active_threshold": self._left_pinch_active_threshold(),
            "navigation_shortcut": self._last_navigation_shortcut,
            "navigation_shortcut_fired": self._last_navigation_shortcut_fired,
            "fps": self._fps,
            **self._navigation_detector.get_debug_state(now),
            **self.mouse_controller.get_debug_state(),
        }

        if not config.PRODUCTION_MODE:
            debug_state["cursor_position"] = self._cursor_position

        self._debug_state = debug_state

    def _pinch_metric(self, side: str) -> float:
        if self.latest_hand_state is None:
            return float("nan")
        if side == "left":
            return self.latest_hand_state.pinch_thumb_index
        return self.latest_hand_state.pinch_thumb_middle

    def _log_pinch_proximity(self, hand_state: HandState) -> None:
        left_metric = hand_state.pinch_thumb_index
        right_metric = hand_state.pinch_thumb_middle

        left_near = left_metric <= config.LEFT_PINCH_THRESHOLD + PINCH_LOG_MARGIN
        right_near = right_metric <= config.RIGHT_PINCH_THRESHOLD + PINCH_LOG_MARGIN

        if left_near and not self._left_pinch_near_logged:
            LOGGER.info(
                "Left pinch near threshold: pinch=%.3f threshold=%.3f cursor=%s",
                left_metric,
                config.LEFT_PINCH_THRESHOLD,
                self._cursor_position,
            )
            self._left_pinch_near_logged = True
        elif not left_near:
            self._left_pinch_near_logged = False

        if right_near and not self._right_pinch_near_logged:
            LOGGER.info(
                "Right pinch near threshold: pinch=%.3f threshold=%.3f cursor=%s",
                right_metric,
                config.RIGHT_PINCH_THRESHOLD,
                self._cursor_position,
            )
            self._right_pinch_near_logged = True
        elif not right_near:
            self._right_pinch_near_logged = False
