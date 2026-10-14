"""
main.py
=======
Real-Time Face Emotion Recognition — Camera Inference Script.

Usage
─────
    python main.py                          # live webcam (default cam 0)
    python main.py --camera 1               # use a different camera index
    python main.py --weights emotion_cnn.pth  # load custom trained weights

How this script uses Apple Silicon MPS
───────────────────────────────────────
1.  `get_device()` from model.py detects `torch.backends.mps.is_available()`.
2.  The EmotionCNN model is moved to the MPS device via `.to(device)`.
3.  Each face crop is converted to a PyTorch tensor and pushed to the MPS
    device before the forward pass — all matrix operations run on the
    Apple GPU through Metal Performance Shaders.
4.  `torch.no_grad()` is used during inference to skip gradient tracking,
    halving memory usage and significantly speeding up the forward pass.
5.  Results are pulled back to CPU only for NumPy/OpenCV display operations
    (MPS tensors cannot be directly converted to NumPy).

Pipeline per frame
──────────────────
  Webcam frame → Grayscale → Haar face detection → Crop + Resize (48×48)
  → Normalise → To MPS Tensor → EmotionCNN forward pass → Softmax
  → Top-1 label + confidence → Draw bounding box + text on frame → Display
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import EMOTION_LABELS, EmotionCNN, get_device, load_model


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
FACE_SIZE = 48          # Model expects 48×48 input
MIN_FACE_SIZE = 60      # Ignore detections smaller than this (pixels)

# Haar cascade XML bundled with OpenCV (no separate download needed)
HAAR_CASCADE_PATH = (
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)

# Normalisation stats (mean & std from grayscale FER-2013 dataset)
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD  = (0.5,)

# Colour palette per emotion (BGR for OpenCV)
EMOTION_COLORS = {
    "Angry":    (0,   0,   220),   # red
    "Disgust":  (0,   140, 0  ),   # dark green
    "Fear":     (128, 0,   128),   # purple
    "Happy":    (0,   200, 255),   # gold-ish yellow
    "Sad":      (200, 100, 0  ),   # steel blue
    "Surprise": (0,   165, 255),   # orange
    "Neutral":  (160, 160, 160),   # grey
}

# ──────────────────────────────────────────────────────────────────
# Pre-processing transform (applied to every face crop)
# ──────────────────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),                         # → [0, 1]  (C, H, W)
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),  # → [-1, 1]
])


# ──────────────────────────────────────────────────────────────────
# Face Detector
# ──────────────────────────────────────────────────────────────────
def build_face_detector() -> cv2.CascadeClassifier:
    """
    Load OpenCV's Haar Cascade face detector.

    The XML file ships with opencv-python — no extra download required.
    """
    if not HAAR_CASCADE_PATH.exists():
        sys.exit(
            f"[Error] Haar cascade not found at {HAAR_CASCADE_PATH}.\n"
            "        Please reinstall opencv-python."
        )
    detector = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    print(f"[Detector] Haar cascade loaded from:\n           {HAAR_CASCADE_PATH}")
    return detector


# ──────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────
def preprocess_face(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Crop a BGR face region → normalised tensor on *device*.

    Args:
        face_bgr: NumPy array of shape (H, W, 3), dtype uint8.
        device:   Target device (mps / cuda / cpu).

    Returns:
        Tensor of shape (1, 1, 48, 48) ready for model forward pass.

    Note on MPS:
        We call `.to(device)` here so the GPU receives the tensor
        before the expensive CNN operations begin.
    """
    tensor = PREPROCESS(face_bgr)          # (1, 48, 48)
    tensor = tensor.unsqueeze(0)           # (1, 1, 48, 48) — add batch dim
    tensor = tensor.to(device)            # ← move to MPS / CUDA / CPU
    return tensor


@torch.no_grad()   # Disables autograd — speeds up inference on MPS
def predict_emotion(
    model: EmotionCNN,
    face_tensor: torch.Tensor,
) -> tuple[str, float]:
    """
    Run a single forward pass and return (label, confidence).

    `torch.no_grad()` is critical for real-time performance:
        • Skips gradient tracking → ~2× less memory on MPS
        • Removes gradient computation overhead from every op

    Args:
        model:       EmotionCNN in eval() mode on device.
        face_tensor: Preprocessed tensor of shape (1, 1, 48, 48) on device.

    Returns:
        label:      Predicted emotion string (e.g. "Happy").
        confidence: Softmax probability 0.0 – 1.0 (e.g. 0.94).
    """
    logits = model(face_tensor)                       # (1, 7) on device
    probs  = F.softmax(logits, dim=1)                 # (1, 7) probabilities

    # Pull back to CPU for Python-level operations (MPS → CPU is fast)
    probs_cpu  = probs.cpu().squeeze(0)               # (7,)
    class_idx  = int(probs_cpu.argmax().item())
    confidence = float(probs_cpu[class_idx].item())
    label      = EMOTION_LABELS[class_idx]

    return label, confidence


# ──────────────────────────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────────────────────────
def draw_overlay(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str,
    confidence: float,
) -> None:
    """
    Draw a bounding box and emotion label+confidence on *frame* in-place.

    Visual design:
        • Coloured rectangle around the face (colour per emotion).
        • Filled pill-shaped label background above the box.
        • White emotion text + confidence percentage.
        • Semi-transparent confidence progress bar below the label.

    Args:
        frame:      BGR frame from OpenCV (modified in-place).
        x, y, w, h: Face bounding box coordinates.
        label:      Predicted emotion string.
        confidence: Softmax probability [0, 1].
    """
    color = EMOTION_COLORS.get(label, (255, 255, 255))

    # ── Bounding box ──────────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)

    # ── Label background ──────────────────────────────────────────
    label_text  = f"{label}  {confidence * 100:.1f}%"
    font        = cv2.FONT_HERSHEY_DUPLEX
    font_scale  = 0.65
    thickness   = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    pad       = 6
    bg_x1     = x
    bg_y1     = max(0, y - text_h - 2 * pad)
    bg_x2     = x + text_w + 2 * pad
    bg_y2     = y

    # Filled rectangle for label background (semi-transparent overlay)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Label text ────────────────────────────────────────────────
    cv2.putText(
        frame,
        label_text,
        (bg_x1 + pad, bg_y2 - pad),
        font,
        font_scale,
        (255, 255, 255),  # white text
        thickness,
        lineType=cv2.LINE_AA,
    )

    # ── Confidence bar (thin strip below text) ────────────────────
    bar_y   = y + 2
    bar_x1  = x
    bar_x2  = x + int(w * confidence)
    cv2.rectangle(frame, (bar_x1, bar_y), (x + w, bar_y + 4), (50, 50, 50), cv2.FILLED)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2,  bar_y + 4), color,       cv2.FILLED)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render the current FPS counter in the top-right corner."""
    text  = f"FPS: {fps:.1f}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = frame.shape[1] - tw - 10
    y = th + 10
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)


def draw_device_info(frame: np.ndarray, device: torch.device) -> None:
    """Show which device is being used for inference (bottom-left)."""
    label_map = {"mps": "⚡ Apple MPS (Metal GPU)", "cuda": "⚡ CUDA GPU", "cpu": "CPU"}
    text  = label_map.get(device.type, device.type)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    h     = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10), font, scale, (0, 200, 255), thick, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────────
def run_inference(model: EmotionCNN, device: torch.device, camera_index: int) -> None:
    """
    Open webcam and run continuous emotion recognition.

    Exits cleanly on:
        • Pressing 'q' or Escape
        • Camera read failure

    Args:
        model:        Trained EmotionCNN in eval() mode on *device*.
        device:       Compute device (mps / cuda / cpu).
        camera_index: OpenCV camera index (0 = built-in webcam).
    """
    detector = build_face_detector()

    # ── Open camera ───────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[Error] Cannot open camera at index {camera_index}.")

    # Prefer higher resolution when available
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[Camera] Feed started. Press  q  or  Esc  to quit.\n")

    # FPS tracking
    prev_time  = time.perf_counter()
    fps        = 0.0
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Frame capture failed — exiting.")
            break

        frame_idx += 1

        # ── Face detection (runs on CPU — OpenCV is CPU-based) ────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # ── Per-face inference on MPS ─────────────────────────────
        for (x, y, w, h) in faces:
            # Clamp coordinates to frame boundaries (safety)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Preprocess → MPS tensor → forward pass
            face_tensor          = preprocess_face(face_crop, device)
            label, confidence    = predict_emotion(model, face_tensor)

            # Draw results on frame
            draw_overlay(frame, x1, y1, x2 - x1, y2 - y1, label, confidence)

        # ── HUD overlays ──────────────────────────────────────────
        # FPS: computed as a rolling average (update every frame)
        curr_time = time.perf_counter()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-9))
        prev_time = curr_time

        draw_fps(frame, fps)
        draw_device_info(frame, device)

        # ── Display ───────────────────────────────────────────────
        cv2.imshow("Real-Time Face Emotion Recognition  |  Press Q to quit", frame)

        # Exit on 'q' or Escape
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\n[Info] User requested exit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Camera released. Goodbye!")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Face Emotion Recognition (Apple MPS optimised)"
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=None,
        help="Path to trained model weights (.pth). If omitted, random weights are used.",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0 = built-in webcam).",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = get_device()

    # ── Load model ────────────────────────────────────────────────
    if args.weights and Path(args.weights).exists():
        print(f"[Model] Loading weights from: {args.weights}")
        model = load_model(args.weights, device)
    else:
        if args.weights:
            print(f"[Warning] Weights file '{args.weights}' not found.")
        print(
            "[Model] No weights loaded — using RANDOM weights.\n"
            "        Train the model on FER-2013 and pass --weights <path>."
        )
        model = EmotionCNN().to(device)
        model.eval()

    print(f"[Model] EmotionCNN ready on device: {device}\n")

    # ── Start real-time inference ─────────────────────────────────
    run_inference(model, device, camera_index=args.camera)


if __name__ == "__main__":
    main()
