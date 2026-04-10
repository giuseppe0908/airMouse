"""Hardened mouse control helpers for macOS and PyAutoGUI."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from math import hypot
import sys
from typing import Any, Callable

from security_utils import install_network_guards

install_network_guards()

import pyautogui

import config

if sys.platform == "darwin":
    import Quartz
else:  # pragma: no cover - macOS is the supported production target
    Quartz = None


LOGGER = logging.getLogger("airmouse.mouse_controller")

CONTROL_MODE_ACTIVE = "ACTIVE"
CONTROL_MODE_TRACKING_ONLY = "TRACKING_ONLY"


class MouseControlSafetyStop(RuntimeError):
    """Raised when FailSafe or a safety-critical mouse condition occurs."""


class MouseControlError(RuntimeError):
    """Raised when the mouse control backend fails unexpectedly."""


class MouseController:
    """Maps webcam coordinates to screen coordinates and controls the mouse."""

    def __init__(self) -> None:
        pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
        pyautogui.PAUSE = config.PYAUTOGUI_PAUSE_SECONDS

        self._screen_width, self._screen_height = pyautogui.size()
        current_x, current_y = self._clamp_screen_coordinates(*self._get_system_position())
        self._smoothed_position = (float(current_x), float(current_y))

        self._accessibility_trusted = self._detect_accessibility_trust()
        self._control_mode = self._resolve_control_mode()
        self._shutdown_requested = False
        self._shutdown_reason: str | None = None
        self._startup_messages = self._build_startup_messages()
        self._startup_warnings = self._build_startup_warnings()
        self._last_move_debug = self._build_default_debug_state()

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen_width, self._screen_height

    @property
    def accessibility_trusted(self) -> bool:
        return self._accessibility_trusted

    @property
    def control_mode(self) -> str:
        return self._control_mode

    @property
    def mouse_control_enabled(self) -> bool:
        return self._control_mode == CONTROL_MODE_ACTIVE

    def get_position(self) -> tuple[int, int]:
        x, y = self._smoothed_position
        return self._clamp_screen_coordinates(int(round(x)), int(round(y)))

    def get_startup_messages(self) -> list[str]:
        return list(self._startup_messages)

    def get_startup_warnings(self) -> list[str]:
        return list(self._startup_warnings)

    def get_debug_state(self) -> dict[str, Any]:
        return dict(self._last_move_debug)

    def get_active_area_rect(
        self,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x1 = int(round(config.ACTIVE_AREA_MARGIN_X * frame_width))
        y1 = int(round(config.ACTIVE_AREA_MARGIN_Y * frame_height))
        x2 = int(round((1.0 - config.ACTIVE_AREA_MARGIN_X) * frame_width))
        y2 = int(round((1.0 - config.ACTIVE_AREA_MARGIN_Y) * frame_height))
        return x1, y1, x2, y2

    def map_camera_to_screen(
        self,
        normalized_x: float,
        normalized_y: float,
    ) -> tuple[float, float]:
        screen_x, screen_y, _, _, _ = self._map_camera_to_screen_detailed(
            normalized_x,
            normalized_y,
        )
        return screen_x, screen_y

    def _map_camera_to_screen_detailed(
        self,
        normalized_x: float,
        normalized_y: float,
    ) -> tuple[float, float, float, float, bool]:
        raw_in_bounds = 0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0
        x = self._normalize_in_active_area(normalized_x, config.ACTIVE_AREA_MARGIN_X)
        y = self._normalize_in_active_area(normalized_y, config.ACTIVE_AREA_MARGIN_Y)
        edge_margin = config.SCREEN_EDGE_SAFETY_MARGIN
        usable_width = max(1, self._screen_width - 1 - edge_margin * 2)
        usable_height = max(1, self._screen_height - 1 - edge_margin * 2)
        screen_x = edge_margin + x * usable_width
        screen_y = edge_margin + y * usable_height
        return screen_x, screen_y, x, y, raw_in_bounds

    def move(self, normalized_x: float, normalized_y: float) -> tuple[int, int]:
        (
            target_x,
            target_y,
            active_area_x,
            active_area_y,
            input_in_bounds,
        ) = self._map_camera_to_screen_detailed(normalized_x, normalized_y)
        current_x, current_y = self._smoothed_position
        alpha = self._adaptive_alpha(target_x, target_y, current_x, current_y)

        smoothed_x = current_x + alpha * (target_x - current_x)
        smoothed_y = current_y + alpha * (target_y - current_y)
        final_x, final_y = self._clamp_screen_coordinates(
            int(round(smoothed_x)),
            int(round(smoothed_y)),
        )
        self._smoothed_position = (float(final_x), float(final_y))
        self._dispatch_move(
            final_x,
            final_y,
            target_position=(target_x, target_y),
            input_position=(normalized_x, normalized_y),
            active_area_position=(active_area_x, active_area_y),
            input_in_bounds=input_in_bounds,
        )
        return final_x, final_y

    def left_click(self) -> None:
        self._dispatch_click("left")

    def right_click(self) -> None:
        self._dispatch_click("right")

    def mouse_down(self) -> None:
        self._dispatch_mouse_down("left")

    def mouse_up(self) -> None:
        self._dispatch_mouse_up("left")

    def keyboard_shortcut(self, keys: tuple[str, ...]) -> bool:
        return self._dispatch_keyboard_shortcut(keys)

    def emergency_release(self) -> None:
        """Best-effort mouse button release for shutdown and fail-safe paths."""

        x, y = self._clamp_screen_coordinates(*self._get_system_position())
        try:
            if sys.platform == "darwin":
                pyautogui.platformModule._mouseUp(x, y, "left")
            else:  # pragma: no cover - macOS is the supported production target
                pyautogui.mouseUp(x=x, y=y, button="left")
        except Exception:
            return

    @staticmethod
    def _normalize_in_active_area(value: float, margin: float) -> float:
        start = margin
        end = 1.0 - margin
        clamped = max(start, min(end, value))
        return (clamped - start) / (end - start)

    @staticmethod
    def _interpolate(start: float, end: float, t: float) -> float:
        return start + (end - start) * t

    def _adaptive_alpha(
        self,
        target_x: float,
        target_y: float,
        current_x: float,
        current_y: float,
    ) -> float:
        movement_speed = hypot(target_x - current_x, target_y - current_y)
        slow = config.SMOOTHING_SPEED_SLOW
        fast = config.SMOOTHING_SPEED_FAST

        if movement_speed <= slow:
            return config.SMOOTHING_ALPHA_SLOW
        if movement_speed >= fast:
            return config.SMOOTHING_ALPHA_FAST

        ratio = (movement_speed - slow) / (fast - slow)
        return self._interpolate(
            config.SMOOTHING_ALPHA_SLOW,
            config.SMOOTHING_ALPHA_FAST,
            ratio,
        )

    def _dispatch_move(
        self,
        x: int,
        y: int,
        *,
        target_position: tuple[float, float],
        input_position: tuple[float, float],
        active_area_position: tuple[float, float],
        input_in_bounds: bool,
    ) -> None:
        system_before = self._clamp_screen_coordinates(*self._get_system_position())
        target_in_bounds = (
            0 <= int(round(target_position[0])) < self._screen_width
            and 0 <= int(round(target_position[1])) < self._screen_height
        )

        if not self.mouse_control_enabled:
            self._last_move_debug = self._build_move_debug_state(
                input_position=input_position,
                input_in_bounds=input_in_bounds,
                active_area_position=active_area_position,
                target_position=target_position,
                target_in_bounds=target_in_bounds,
                command_position=(x, y),
                system_position_before=system_before,
                system_position_after=system_before,
                command_sent=False,
                movement_applied=False,
                dispatch_backend="blocked-no-accessibility",
                dispatch_error=None,
            )
            return

        dispatch_backend = "pyautogui"
        dispatch_error = None
        command_sent = False
        movement_applied = False
        system_after = system_before

        def _action() -> None:
            nonlocal dispatch_backend
            if sys.platform == "darwin":
                pyautogui.platformModule._moveTo(x, y)
                dispatch_backend = "pyautogui-osx-direct"
            else:  # pragma: no cover - macOS is the supported production target
                pyautogui.moveTo(x, y, duration=0)

        try:
            self._execute_control_action("move", x, y, _action)
            command_sent = True
            system_after = self._clamp_screen_coordinates(*self._get_system_position())
            movement_applied = system_after != system_before or system_after == (x, y)

            if (
                sys.platform == "darwin"
                and Quartz is not None
                and not movement_applied
            ):
                Quartz.CGWarpMouseCursorPosition((x, y))
                Quartz.CGAssociateMouseAndMouseCursorPosition(True)
                dispatch_backend = "quartz-warp-fallback"
                system_after = self._clamp_screen_coordinates(*self._get_system_position())
                movement_applied = system_after != system_before or system_after == (x, y)
        except MouseControlSafetyStop as exc:
            dispatch_error = str(exc)
            raise
        except MouseControlError as exc:
            dispatch_error = str(exc)
            raise
        finally:
            self._last_move_debug = self._build_move_debug_state(
                input_position=input_position,
                input_in_bounds=input_in_bounds,
                active_area_position=active_area_position,
                target_position=target_position,
                target_in_bounds=target_in_bounds,
                command_position=(x, y),
                system_position_before=system_before,
                system_position_after=system_after,
                command_sent=command_sent,
                movement_applied=movement_applied,
                dispatch_backend=dispatch_backend,
                dispatch_error=dispatch_error,
            )

    def _dispatch_click(self, button: str) -> None:
        if not self.mouse_control_enabled:
            LOGGER.warning(
                "Mouse click blocked: button=%s mode=%s",
                button,
                self._control_mode,
            )
            return

        x, y = self._clamp_screen_coordinates(*self.get_position())
        LOGGER.info(
            "Dispatching mouse click: button=%s pos=(%d, %d) mode=%s",
            button,
            x,
            y,
            self._control_mode,
        )

        def _action() -> None:
            if sys.platform == "darwin":
                pyautogui.platformModule._click(x, y, button)
            else:  # pragma: no cover - macOS is the supported production target
                pyautogui.click(x=x, y=y, button=button)

        self._execute_control_action("click", x, y, _action)
        LOGGER.info("Mouse click dispatched successfully: button=%s pos=(%d, %d)", button, x, y)

    def _dispatch_mouse_down(self, button: str) -> None:
        if not self.mouse_control_enabled:
            LOGGER.warning(
                "Mouse down blocked: button=%s mode=%s",
                button,
                self._control_mode,
            )
            return

        x, y = self._clamp_screen_coordinates(*self.get_position())
        LOGGER.info(
            "Dispatching mouse down: button=%s pos=(%d, %d) mode=%s",
            button,
            x,
            y,
            self._control_mode,
        )

        def _action() -> None:
            if sys.platform == "darwin":
                pyautogui.platformModule._mouseDown(x, y, button)
            else:  # pragma: no cover - macOS is the supported production target
                pyautogui.mouseDown(x=x, y=y, button=button)

        self._execute_control_action("mouse_down", x, y, _action)
        LOGGER.info("Mouse down dispatched successfully: button=%s pos=(%d, %d)", button, x, y)

    def _dispatch_mouse_up(self, button: str) -> None:
        if not self.mouse_control_enabled:
            LOGGER.warning(
                "Mouse up blocked: button=%s mode=%s",
                button,
                self._control_mode,
            )
            return

        x, y = self._clamp_screen_coordinates(*self.get_position())
        LOGGER.info(
            "Dispatching mouse up: button=%s pos=(%d, %d) mode=%s",
            button,
            x,
            y,
            self._control_mode,
        )

        def _action() -> None:
            if sys.platform == "darwin":
                pyautogui.platformModule._mouseUp(x, y, button)
            else:  # pragma: no cover - macOS is the supported production target
                pyautogui.mouseUp(x=x, y=y, button=button)

        self._execute_control_action("mouse_up", x, y, _action)
        LOGGER.info("Mouse up dispatched successfully: button=%s pos=(%d, %d)", button, x, y)

    def _dispatch_keyboard_shortcut(self, keys: tuple[str, ...]) -> bool:
        if not self.mouse_control_enabled:
            LOGGER.warning(
                "Keyboard shortcut blocked: keys=%s mode=%s",
                "+".join(keys),
                self._control_mode,
            )
            return False

        x, y = self._clamp_screen_coordinates(*self.get_position())
        LOGGER.info(
            "Dispatching keyboard shortcut: keys=%s pos=(%d, %d) mode=%s",
            "+".join(keys),
            x,
            y,
            self._control_mode,
        )

        def _action() -> None:
            pyautogui.hotkey(*keys)

        self._execute_control_action("keyboard_shortcut", x, y, _action)
        LOGGER.info(
            "Keyboard shortcut dispatched successfully: keys=%s pos=(%d, %d)",
            "+".join(keys),
            x,
            y,
        )
        return True

    def _execute_control_action(
        self,
        action_name: str,
        x: int,
        y: int,
        action: Callable[[], None],
    ) -> None:
        try:
            self._check_failsafe()
            action()
            self._check_failsafe()
        except pyautogui.FailSafeException as exc:
            self._shutdown_requested = True
            self._shutdown_reason = f"{action_name} aborted by PyAutoGUI fail-safe"
            self.emergency_release()
            LOGGER.warning("PyAutoGUI fail-safe triggered; shutting down safely.")
            raise MouseControlSafetyStop(self._shutdown_reason) from exc
        except Exception as exc:
            self.emergency_release()
            LOGGER.exception("Mouse control backend failed during %s.", action_name)
            raise MouseControlError("Mouse control backend failure.") from exc

    def _check_failsafe(self) -> None:
        if not pyautogui.FAILSAFE:
            return
        current_position = self._clamp_screen_coordinates(*self._get_system_position())
        failsafe_points = {
            (0, 0),
            (self._screen_width - 1, 0),
            (0, self._screen_height - 1),
            (self._screen_width - 1, self._screen_height - 1),
        }
        if current_position in failsafe_points:
            raise pyautogui.FailSafeException(
                "Mouse moved to a fail-safe screen corner."
            )

    def _build_default_debug_state(self) -> dict[str, Any]:
        position = self.get_position()
        return {
            "input_position": None,
            "input_in_bounds": True,
            "active_area_position": None,
            "target_position": position,
            "target_in_bounds": True,
            "command_position": position,
            "system_position_before": position,
            "system_position_after": position,
            "screen_size": self.screen_size,
            "command_sent": False,
            "movement_applied": False,
            "dispatch_backend": "none",
            "dispatch_error": None,
            "accessibility_trusted": self._accessibility_trusted,
            "control_mode": self._control_mode,
            "mouse_events_blocked": not self.mouse_control_enabled,
            "shutdown_requested": self._shutdown_requested,
            "shutdown_reason": self._shutdown_reason,
        }

    def _build_move_debug_state(
        self,
        *,
        input_position: tuple[float, float],
        input_in_bounds: bool,
        active_area_position: tuple[float, float],
        target_position: tuple[float, float],
        target_in_bounds: bool,
        command_position: tuple[int, int],
        system_position_before: tuple[int, int],
        system_position_after: tuple[int, int],
        command_sent: bool,
        movement_applied: bool,
        dispatch_backend: str,
        dispatch_error: str | None,
    ) -> dict[str, Any]:
        return {
            "input_position": input_position,
            "input_in_bounds": input_in_bounds,
            "active_area_position": active_area_position,
            "target_position": (
                int(round(target_position[0])),
                int(round(target_position[1])),
            ),
            "target_in_bounds": target_in_bounds,
            "command_position": command_position,
            "system_position_before": system_position_before,
            "system_position_after": system_position_after,
            "screen_size": self.screen_size,
            "command_sent": command_sent,
            "movement_applied": movement_applied,
            "dispatch_backend": dispatch_backend,
            "dispatch_error": dispatch_error,
            "accessibility_trusted": self._accessibility_trusted,
            "control_mode": self._control_mode,
            "mouse_events_blocked": not self.mouse_control_enabled,
            "shutdown_requested": self._shutdown_requested,
            "shutdown_reason": self._shutdown_reason,
        }

    def _build_startup_messages(self) -> list[str]:
        screen_width, screen_height = self.screen_size
        messages = [
            f"AirMouse startup: screen={screen_width}x{screen_height}",
            f"Mouse control mode: {self._control_mode}",
        ]

        if sys.platform != "darwin":
            return messages

        messages.extend(
            [
                (
                    "Accessibility trusted: "
                    f"{'YES' if self._accessibility_trusted else 'NO'}"
                ),
            ]
        )

        if not self._accessibility_trusted:
            messages.extend(
                [
                    "TRACKING ONLY: hand tracking and mapped screen coordinates will "
                    "still update, but macOS is blocking real cursor movement and clicks.",
                    "Grant Accessibility to the host app that launched Python "
                    "(Terminal, iTerm, VS Code, PyCharm, etc.), then fully quit and "
                    "relaunch AirMouse.",
                    "Open: System Settings > Privacy & Security > Accessibility",
                ]
            )

        return messages

    def _build_startup_warnings(self) -> list[str]:
        warnings: list[str] = []
        if sys.platform == "darwin" and not self._accessibility_trusted:
            warnings.append(
                "Accessibility permission is missing for this app/process. "
                "Hand tracking can still work, but macOS will block real cursor movement and clicks. "
                "Enable it in System Settings > Privacy & Security > Accessibility."
            )
        return warnings

    def _resolve_control_mode(self) -> str:
        if sys.platform == "darwin" and not self._accessibility_trusted:
            return CONTROL_MODE_TRACKING_ONLY
        return CONTROL_MODE_ACTIVE

    def _detect_accessibility_trust(self) -> bool:
        if sys.platform != "darwin":
            return True

        framework_path = ctypes.util.find_library("ApplicationServices")
        if not framework_path:
            return False

        try:
            application_services = ctypes.cdll.LoadLibrary(framework_path)
            application_services.AXIsProcessTrusted.restype = ctypes.c_bool
            return bool(application_services.AXIsProcessTrusted())
        except Exception:
            return False

    def _get_system_position(self) -> tuple[int, int]:
        if sys.platform == "darwin" and Quartz is not None:
            event = Quartz.CGEventCreate(None)
            location = Quartz.CGEventGetLocation(event)
            return int(round(location.x)), int(round(location.y))

        position = pyautogui.position()
        return int(position.x), int(position.y)

    def _clamp_screen_coordinates(self, x: int, y: int) -> tuple[int, int]:
        clamped_x = max(0, min(int(x), self._screen_width - 1))
        clamped_y = max(0, min(int(y), self._screen_height - 1))
        return clamped_x, clamped_y
