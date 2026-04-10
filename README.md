# AirMouse

AirMouse is a macOS hand gesture mouse controller that uses a webcam, MediaPipe hand landmarks, OpenCV, and PyAutoGUI. The index fingertip drives the cursor, thumb-index pinch handles left click and drag, thumb-middle pinch handles right click, and a fist pauses tracking so the cursor stays frozen and safe.

## File Structure

```text
airMouse/
├── README.md
├── config.py
├── controller.py
├── hand_tracker.py
├── main.py
├── models/
│   └── hand_landmarker.task
├── mouse_controller.py
├── requirements.txt
└── security_utils.py
```

## Prerequisites

- macOS
- Python 3.10 or newer
- A working webcam
- Camera permission for webcam access
- Accessibility permission for mouse control

## Virtual Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

## Install

```bash
python3 -m pip install -r requirements.txt
```

The project expects the local MediaPipe model bundle at `models/hand_landmarker.task`.

## Run

```bash
python3 main.py
```

Press `q` or `Esc` to quit cleanly.

## Important macOS Note

If the preview window works, landmarks draw correctly, and finger distances update, but the real macOS cursor still does not move, the most likely blocker is missing Accessibility permission for the app that launched Python.

AirMouse now makes that state explicit:

- `ACTIVE` mode means hand tracking is live and AirMouse is allowed to send real mouse events.
- `TRACKING_ONLY` mode means hand tracking is live and mapped cursor coordinates still update in the HUD, but macOS is blocking real cursor movement and clicks.

## macOS Permissions Required

AirMouse uses only two macOS permissions:

1. Camera
   Required to read webcam frames for hand landmark detection.
2. Accessibility
   Required to move the macOS cursor and emit mouse clicks through PyAutoGUI.

No other system permissions are requested or used.

Important: on macOS, Accessibility permission usually belongs to the host app that launched Python, not only the Python binary itself. If you start AirMouse from Terminal, iTerm, VS Code, PyCharm, or another IDE, grant Accessibility to that app and then fully quit and relaunch it.

## Gesture Logic

- Cursor movement
  After tracking is active, the index fingertip (`landmark 8`) controls the cursor by itself. You do not need to keep all five fingers open while pointing. Movement is restricted to a configurable active area inside the webcam frame so edge jitter does not fling the cursor around the screen.
- Adaptive smoothing
  Small movements are smoothed more heavily for stability. Larger and faster motions get less smoothing so the cursor remains responsive.
- Left click and drag
  Thumb and index finger pinch starts a pending gesture. If the pinch is released after at least `150 ms` but before `200 ms`, AirMouse performs a single left click. If the pinch remains held for `200 ms` or longer, AirMouse presses and holds the left mouse button for dragging, then releases it when the pinch opens.
- Right click
  Thumb and middle finger pinch must survive a `150 ms` debounce window. After that, AirMouse emits one right click and waits for the pinch to open before allowing another one.
- Pause and resume
  AirMouse starts paused. A stable open hand activates or resumes tracking. A stable closed fist pauses tracking and freezes the cursor. After resuming, you can immediately point with only the index finger. If tracking pauses while a drag is active, AirMouse automatically releases the mouse button.
- Hand lost safety
  If the hand leaves the frame during a drag, AirMouse automatically releases the mouse button.

## Tunable Parameters

All runtime tuning lives in [config.py](/Users/giuseppeplacida/Desktop/airMouse/config.py).

- Camera and frame settings
  `FRAME_WIDTH`, `FRAME_HEIGHT`, `FRAME_FLIP_HORIZONTAL`
- MediaPipe confidence
  `MIN_DETECTION_CONFIDENCE`, `MIN_HAND_PRESENCE_CONFIDENCE`, `MIN_TRACKING_CONFIDENCE`
- Active area
  `ACTIVE_AREA_MARGIN_X`, `ACTIVE_AREA_MARGIN_Y`
- Pinch behavior
  `LEFT_PINCH_THRESHOLD`, `RIGHT_PINCH_THRESHOLD`, `PINCH_DEBOUNCE_SECONDS`, `LEFT_PINCH_CLICK_MAX_SECONDS`
- Hand posture recognition
  `FINGER_UP_Y_MARGIN`, `THUMB_OPEN_X_MARGIN`, `FIST_MAX_AVERAGE_TIP_DISTANCE`, `FIST_MAX_THUMB_TO_INDEX_MCP_DISTANCE`
- Cursor smoothing
  `SMOOTHING_ALPHA_SLOW`, `SMOOTHING_ALPHA_FAST`, `SMOOTHING_SPEED_SLOW`, `SMOOTHING_SPEED_FAST`
- Production and logging
  `PRODUCTION_MODE`, `ENABLE_FILE_LOGGING`, `LOG_LEVEL`

## Troubleshooting

### Webcam window is black or does not open

- Confirm no other app is monopolizing the camera.
- Check Camera permission for the terminal or IDE launching Python.
- Verify the webcam works in another app like FaceTime or Photo Booth.
- Confirm the local MediaPipe model file exists at `models/hand_landmarker.task`.

### Cursor is frozen

- Show an open hand with all five fingers extended for a brief moment to resume tracking after a fist pause.
- The fist pause now requires a short hold, so brief hand closures during a click should not pause tracking immediately.
- If the HUD says `Mode: TRACKING_ONLY`, hand tracking is working but macOS is blocking real mouse events because Accessibility permission is missing.
- Check Accessibility permission for the process running Python.
- If the pointer is in a screen corner, PyAutoGUI fail-safe may have stopped the app intentionally.
- If no hand is detected, adjust lighting and keep the hand fully inside the frame.

### MediaPipe prints `Using NORM_RECT without IMAGE_DIMENSIONS`

- This warning comes from MediaPipe's internal landmark projection path.
- It can affect coordinate accuracy slightly when the internal region of interest is not square.
- AirMouse now clamps small out-of-range landmark overshoots instead of dropping the whole frame.
- It is not the main reason the macOS cursor fails to move.
- If AirMouse shows changing `Index tip`, `Target`, and `Cursor` values while the HUD says `Mode: TRACKING_ONLY`, the practical blocker is macOS Accessibility permission, not this warning.

## macOS Verification Checklist

1. Launch `python3 main.py`.
2. Read the startup diagnostics in the terminal and confirm `Mouse control mode: ACTIVE`.
3. Move your hand and verify the HUD updates `Index tip` and `Target`.
4. Confirm the HUD shows `Move cmd: SENT` and `Move applied: YES`.
5. If the HUD shows `Mode: TRACKING_ONLY`, open `System Settings > Privacy & Security > Accessibility`, enable the terminal or IDE that launched Python, then fully quit and relaunch that app before testing again.

### Cursor is jittery

- Increase `SMOOTHING_ALPHA_SLOW` only slightly if the cursor feels sticky.
- Lower `SMOOTHING_ALPHA_FAST` if fast motions overshoot.
- Increase `ACTIVE_AREA_MARGIN_X` and `ACTIVE_AREA_MARGIN_Y` to reduce edge noise.
- Improve lighting and keep the hand closer to the center of the frame.

### False pinch detections

- Decrease `LEFT_PINCH_THRESHOLD` or `RIGHT_PINCH_THRESHOLD` to require a tighter pinch.
- Increase `PINCH_DEBOUNCE_SECONDS` if brief accidental contacts trigger clicks.
- Keep the thumb clearly separated from nearby fingers when not clicking.

## Security Model

AirMouse is intentionally constrained:

- It reads webcam frames only from the local default camera.
- It reads only project-local files that it needs to operate, notably `models/hand_landmarker.task`.
- It does not open sockets, make HTTP requests, or use remote URLs during runtime.
- It does not spawn subprocesses and does not use `eval`, `exec`, `pickle`, `os.system`, or `os.popen`.
- It writes no runtime files outside the project directory. Optional file logging and any unavoidable third-party runtime cache artifacts are confined to `./logs/`, and each file is capped at 1 MB by startup enforcement.
- It hard-clamps all mouse coordinates to the visible screen bounds before dispatch.
- It keeps PyAutoGUI fail-safe enabled at all times.
- It validates every frame shape and every hand landmark coordinate before use.
- It shuts down cleanly on `SIGINT`, `SIGTERM`, fail-safe activation, webcam failure, or unexpected runtime exceptions.

## Security Verification

You can verify the runtime network lockdown on macOS with:

```bash
pgrep -f "python.*main.py" | head -n 1 | xargs -I{} lsof -nP -a -p {} -i
```

If the hardening is active, that command should print no network sockets for the AirMouse process.

You can also force a blocked network call in-process:

```bash
.venv/bin/python -c "import main, socket; socket.create_connection(('example.com', 80))"
```

That command should fail with `security_utils.NetworkAccessBlockedError`.

## Future Improvements

- Add multi-monitor aware mapping and monitor selection.
- Add optional gesture calibration on startup.
- Support scroll, zoom, and mission control gestures.
- Add a small settings UI for live tuning without editing code.
- Persist named profiles for different users and webcams.
