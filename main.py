"""Application entry point for the hardened AirMouse controller."""

from __future__ import annotations

import signal
import sys
from types import FrameType
from typing import Any

sys.dont_write_bytecode = True

import config
from security_utils import (
    NetworkAccessBlockedError,
    configure_logging,
    enforce_directory_file_size_cap,
    install_network_guards,
    prepare_runtime_environment,
)

install_network_guards()
prepare_runtime_environment(config.LOG_DIR_NAME)

import cv2
import numpy as np

from controller import HandMouseController
from hand_tracker import HAND_CONNECTIONS
from mouse_controller import MouseControlError, MouseControlSafetyStop


LOGGER = configure_logging(
    production_mode=config.PRODUCTION_MODE,
    enable_file_logging=config.ENABLE_FILE_LOGGING,
    log_dir_name=config.LOG_DIR_NAME,
    log_file_name=config.LOG_FILE_NAME,
    log_file_max_bytes=config.LOG_FILE_MAX_BYTES,
    log_backup_count=config.LOG_BACKUP_COUNT,
    log_level=config.LOG_LEVEL,
)
enforce_directory_file_size_cap(
    config.PROJECT_ROOT / config.LOG_DIR_NAME,
    config.LOG_FILE_MAX_BYTES,
)


def get_status_banner_lines(debug_state: dict[str, object]) -> tuple[list[str], tuple[int, int, int]]:
    if debug_state["control_mode"] == "TRACKING_ONLY":
        return (
            [
                "TRACKING ONLY: macOS Accessibility permission missing.",
                "Hand tracking works, but real cursor movement and clicks are blocked.",
            ],
            config.COLOR_DANGER,
        )

    if (
        debug_state["command_sent"]
        and not debug_state["movement_applied"]
        and not debug_state["mouse_events_blocked"]
    ):
        return (
            [
                "MOVE NOT APPLIED: mapped coordinates updated, but the system cursor stayed put.",
                "Check landmark status, projection accuracy, and macOS event injection.",
            ],
            config.COLOR_WARNING,
        )

    return [], config.COLOR_WARNING


def normalized_to_pixel(
    point: tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    x = min(max(int(round(point[0] * (frame_width - 1))), 0), frame_width - 1)
    y = min(max(int(round(point[1] * (frame_height - 1))), 0), frame_height - 1)
    return x, y


def draw_landmarks(frame: Any, hand_state: Any) -> None:
    if not config.HUD_SHOW_LANDMARKS or hand_state is None:
        return

    frame_height, frame_width = frame.shape[:2]
    points = [
        normalized_to_pixel(point, frame_width, frame_height)
        for point in hand_state.raw_landmarks
    ]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_idx],
            points[end_idx],
            config.COLOR_PRIMARY,
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(frame, point, 4, config.COLOR_TEXT, thickness=-1)
        cv2.circle(frame, point, 3, config.COLOR_PRIMARY, thickness=-1)


def draw_active_area(frame: Any, controller: HandMouseController) -> None:
    if not config.HUD_SHOW_ACTIVE_AREA:
        return

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = controller.mouse_controller.get_active_area_rect(
        frame_width,
        frame_height,
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_ACTIVE_AREA, 2)
    cv2.putText(
        frame,
        "Active area",
        (x1, max(config.HUD_MARGIN, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.COLOR_ACTIVE_AREA,
        2,
        cv2.LINE_AA,
    )


def draw_status_banner(frame: Any, controller: HandMouseController) -> None:
    debug_state = controller.get_debug_state()
    banner_lines, banner_color = get_status_banner_lines(debug_state)
    if not banner_lines:
        return

    banner_height = 68
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], banner_height),
        banner_color,
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    for index, line in enumerate(banner_lines):
        text_y = 24 + index * 24
        cv2.putText(
            frame,
            line,
            (12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            config.COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )


def draw_debug_panel(frame: Any, controller: HandMouseController) -> None:
    if not config.HUD_SHOW_TEXT:
        return

    debug_state = controller.get_debug_state()
    x = config.HUD_MARGIN
    y = config.HUD_MARGIN
    if get_status_banner_lines(debug_state)[0]:
        y += 60
    panel_w = config.HUD_PANEL_WIDTH

    lines = [
        f"Mode: {debug_state['control_mode']}",
        f"Gesture: {debug_state['gesture']}",
        f"Nav: {format_navigation_state(debug_state)}",
        f"Nav dbg: {format_navigation_debug(debug_state)}",
        f"Tracking: {debug_state['tracking_state']}",
        f"Mouse: {debug_state['mouse_state']}",
        f"Handedness: {debug_state['handedness'] or '--'}",
        f"Fingers: {format_finger_state_summary(debug_state)}",
        f"L pinch: {format_left_pinch_debug(debug_state)}",
        f"Resume palm: {'YES' if debug_state['resume_gesture'] else 'NO'}",
        f"Landmarks: {format_landmark_status(debug_state['clamped_landmark_count'])}",
        (
            f"Index tip: {format_normalized_point(debug_state['index_tip'])} "
            f"{'OK' if debug_state['input_in_bounds'] else 'OUT'}"
        ),
        f"Target: {format_point(debug_state['target_position'])}",
        (
            f"Move cmd: "
            f"{'BLOCKED' if debug_state['mouse_events_blocked'] else ('SENT' if debug_state['command_sent'] else 'IDLE')} "
            f"via {debug_state['dispatch_backend']}"
        ),
        f"Move applied: {'YES' if debug_state['movement_applied'] else 'NO'}",
        f"Accessibility: {'OK' if debug_state['accessibility_trusted'] else 'MISSING'}",
        f"FPS: {debug_state['fps']:.1f}",
        "Controls: q or ESC to quit",
    ]

    if not config.PRODUCTION_MODE:
        lines.insert(
            9,
            f"Thumb-Middle pinch: {format_metric(debug_state['pinch_thumb_middle'])}",
        )
        lines.insert(
            9,
            f"Thumb-Index pinch: {format_metric(debug_state['pinch_thumb_index'])}",
        )
        if "cursor_position" in debug_state:
            cursor_x, cursor_y = debug_state["cursor_position"]
            lines.insert(8, f"Cursor: ({cursor_x}, {cursor_y})")
        if debug_state["shutdown_requested"]:
            lines.insert(4, "Safety stop requested")

    panel_h = max(config.HUD_PANEL_HEIGHT, 24 + len(lines) * config.HUD_LINE_HEIGHT)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_w, y + panel_h),
        config.COLOR_PANEL,
        thickness=-1,
    )
    cv2.addWeighted(
        overlay,
        config.HUD_PANEL_OPACITY,
        frame,
        1.0 - config.HUD_PANEL_OPACITY,
        0,
        frame,
    )

    for index, line in enumerate(lines):
        text_y = y + 34 + index * config.HUD_LINE_HEIGHT
        cv2.putText(
            frame,
            line,
            (x + 14, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.HUD_FONT_SCALE,
            config.COLOR_TEXT,
            config.HUD_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def format_metric(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def format_navigation_state(debug_state: dict[str, object]) -> str:
    candidate = str(debug_state.get("navigation_candidate") or "NONE")
    detected = debug_state.get("navigation_detected")
    cooldown = float(debug_state.get("navigation_cooldown_remaining") or 0.0)
    pending = bool(debug_state.get("navigation_pending"))

    if cooldown > 0.0:
        active = str(detected or candidate)
        return f"{active} cooldown {cooldown:.1f}s"

    if candidate == "NONE":
        return "NONE"

    delta = debug_state.get("navigation_delta")
    suffix = " pending" if pending else ""
    return f"{candidate}{suffix} d={format_motion_delta(delta)}"


def format_navigation_debug(debug_state: dict[str, object]) -> str:
    extended_count = debug_state.get("navigation_extended_fingers")
    alignment_axis = debug_state.get("navigation_alignment_axis") or "--"
    alignment = format_metric(debug_state.get("navigation_finger_alignment"))
    history_size = debug_state.get("navigation_history_size") or 0
    history_duration = format_seconds(debug_state.get("navigation_history_duration"))
    fired = "YES" if debug_state.get("navigation_shortcut_fired") else "NO"
    shortcut = debug_state.get("navigation_shortcut") or "--"
    vertical = "V" if debug_state.get("navigation_vertical_shape") else "-"
    horizontal = "H" if debug_state.get("navigation_horizontal_shape") else "-"
    return (
        f"ext={extended_count} align {alignment_axis}:{alignment} "
        f"hist={history_size}/{history_duration} shape={vertical}{horizontal} "
        f"fired={fired} {shortcut}"
    )


def format_finger_state_summary(debug_state: dict[str, object]) -> str:
    finger_states = debug_state.get("finger_states")
    navigation_fingers = debug_state.get("navigation_finger_states")
    raw = format_finger_states(finger_states, ("thumb", "index", "middle", "ring", "pinky"))
    nav = format_finger_states(navigation_fingers, ("index", "middle", "ring", "pinky"))
    extended_count = debug_state.get("extended_finger_count")
    non_thumb_count = debug_state.get("non_thumb_extended_count")
    return f"raw TIMRP={raw} ext={extended_count}/{non_thumb_count} nav IMRP={nav}"


def format_finger_states(value: object, order: tuple[str, ...]) -> str:
    if not isinstance(value, dict) or not value:
        return "-----"[: len(order)]
    labels = {
        "thumb": "T",
        "index": "I",
        "middle": "M",
        "ring": "R",
        "pinky": "P",
    }
    return "".join(labels[name] if value.get(name) else "-" for name in order)


def format_left_pinch_debug(debug_state: dict[str, object]) -> str:
    metric = format_metric(debug_state.get("pinch_thumb_index"))
    state = "ON" if debug_state.get("left_pinching") else "OFF"
    threshold = format_metric(debug_state.get("left_pinch_active_threshold"))
    hold = format_seconds(debug_state.get("left_pinch_hold_elapsed"))
    if debug_state.get("left_pinch_drag_ready"):
        phase = "drag"
    elif debug_state.get("left_pinch_click_armed"):
        phase = "armed"
    else:
        phase = "wait"
    return f"{metric} {state} th={threshold} hold={hold} {phase}"


def format_motion_delta(value: object) -> str:
    if not isinstance(value, tuple) or len(value) != 2:
        return "(--, --)"
    dx, dy = value
    if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
        return "(--, --)"
    return f"({dx:+.2f}, {dy:+.2f})"


def format_seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "--"
    return f"{value:.2f}s"


def format_normalized_point(point: tuple[float, float] | None) -> str:
    if point is None:
        return "(--, --)"
    return f"({point[0]:.3f}, {point[1]:.3f})"


def format_point(point: tuple[int, int] | None) -> str:
    if point is None:
        return "(--, --)"
    return f"({point[0]}, {point[1]})"


def format_landmark_status(clamped_count: int | None) -> str:
    if clamped_count is None:
        return "--"
    if clamped_count == 0:
        return "OK"
    return f"CLAMPED({clamped_count})"


def is_valid_frame(frame: Any) -> bool:
    if not isinstance(frame, np.ndarray):
        LOGGER.warning("Skipping non-array frame.")
        return False
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) != 3:
        LOGGER.warning("Skipping frame with invalid shape.")
        return False
    height, width, channels = shape
    if height <= 0 or width <= 0 or channels != 3:
        LOGGER.warning("Skipping frame with invalid dimensions.")
        return False
    return True


def main() -> int:
    config.validate_config()

    state: dict[str, Any] = {
        "cap": None,
        "controller": None,
        "cleaned": False,
    }

    def cleanup() -> None:
        if state["cleaned"]:
            return
        state["cleaned"] = True

        controller = state["controller"]
        if controller is not None:
            try:
                controller.release_if_needed()
            except Exception:
                LOGGER.exception("Controller cleanup failed.")

        cap = state["cap"]
        if cap is not None:
            try:
                cap.release()
            except Exception:
                LOGGER.exception("Webcam release failed.")

        try:
            cv2.destroyAllWindows()
        except Exception:
            LOGGER.exception("OpenCV window cleanup failed.")

    def handle_signal(_signum: int, _frame: FrameType | None) -> None:
        LOGGER.info("Termination signal received. Cleaning up.")
        cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        try:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            state["cap"] = cap
        except Exception:
            LOGGER.exception("Failed to initialize the webcam capture.")
            cleanup()
            return 1

        if cap is None or not cap.isOpened():
            LOGGER.error(
                "Unable to open the webcam. Confirm Camera permission and that no other app is using it."
            )
            cleanup()
            return 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        controller = HandMouseController()
        state["controller"] = controller

        for message in controller.mouse_controller.get_startup_messages():
            LOGGER.info(message)
        for warning in controller.mouse_controller.get_startup_warnings():
            LOGGER.warning(warning)

        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.error("Failed to read a frame from the webcam.")
                return 1

            if config.FRAME_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            if not is_valid_frame(frame):
                continue

            controller.update(frame)

            draw_landmarks(frame, controller.latest_hand_state)
            draw_active_area(frame, controller)
            draw_status_banner(frame, controller)
            draw_debug_panel(frame, controller)

            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (config.QUIT_KEY, config.ESC_KEY):
                LOGGER.info("Shutdown requested by user input.")
                return 0
    except MouseControlSafetyStop:
        LOGGER.warning("PyAutoGUI fail-safe triggered. Exiting gracefully.")
        return 0
    except MouseControlError:
        LOGGER.exception("Mouse control failed. Exiting.")
        return 1
    except NetworkAccessBlockedError:
        LOGGER.exception("Blocked a forbidden network access attempt. Exiting.")
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Fatal runtime error.")
        return 1
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
