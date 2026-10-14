"""
model.py
========
CNN Architecture for Real-Time Face Emotion Recognition.

Architecture is inspired by VGG-style blocks and tuned for the
FER-2013 dataset (48×48 grayscale images, 7 emotion classes).

Emotions (class indices):
    0 - Angry
    1 - Disgust
    2 - Fear
    3 - Happy
    4 - Sad
    5 - Surprise
    6 - Neutral

Device support:
    - Apple Silicon GPU  →  'mps'  (Metal Performance Shaders)
    - NVIDIA GPU         →  'cuda'
    - CPU fallback       →  'cpu'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Emotion label mapping (matches FER-2013 class ordering)
# ------------------------------------------------------------------
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

NUM_CLASSES = len(EMOTION_LABELS)  # 7


# ------------------------------------------------------------------
# Helper: Convolutional Block
# ------------------------------------------------------------------
class ConvBlock(nn.Module):
    """
    A reusable Conv → BatchNorm → ReLU block.

    Using BatchNorm after each convolution stabilises training and
    acts as a regulariser, reducing the need for Dropout in early layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # bias is redundant when BatchNorm follows
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------------------
# Main Model: EmotionCNN
# ------------------------------------------------------------------
class EmotionCNN(nn.Module):
    """
    A VGG-style CNN designed for 48×48 grayscale face images.

    Network layout
    ──────────────
    Stage 1 │ 2× ConvBlock(1→64)   │ MaxPool  →  24×24
    Stage 2 │ 2× ConvBlock(64→128) │ MaxPool  →  12×12
    Stage 3 │ 3× ConvBlock(128→256)│ MaxPool  →   6×6
    Stage 4 │ 3× ConvBlock(256→512)│ MaxPool  →   3×3
    Classifier │ FC(512*3*3→1024) → FC(1024→256) → FC(256→7)

    Total parameters: ~5.8M  (fast enough for real-time on MPS)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # ── Feature Extractor ──────────────────────────────────────
        self.features = nn.Sequential(
            # Stage 1: 48×48 → 24×24
            ConvBlock(1, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Stage 2: 24×24 → 12×12
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Stage 3: 12×12 → 6×6
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Stage 4: 6×6 → 3×3
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        # ── Classifier Head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation for faster convergence
        self._initialise_weights()

    # --------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """Kaiming (He) init for Conv layers; Xavier for Linear."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 48, 48) — grayscale face crops.

        Returns:
            logits: Tensor of shape (B, 7) — raw (un-softmaxed) scores.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------------
# Utility: Device selector (MPS → CUDA → CPU)
# ------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority:
        1. MPS  – Apple Silicon GPU (Metal Performance Shaders)
        2. CUDA – NVIDIA GPU
        3. CPU  – Universal fallback

    MPS is available on macOS 12.3+ with an M-series chip and
    requires PyTorch ≥ 1.12.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] ✅ Apple Silicon MPS backend selected (Metal GPU).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] ✅ CUDA GPU selected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] ⚠️  No GPU found — falling back to CPU.")
    return device


# ------------------------------------------------------------------
# Utility: Load model weights safely
# ------------------------------------------------------------------
def load_model(weights_path: str, device: torch.device) -> EmotionCNN:
    """
    Instantiate EmotionCNN and load weights from *weights_path*.

    Args:
        weights_path: Path to a saved ``state_dict`` (.pth / .pt file).
        device:       Target device (mps / cuda / cpu).

    Returns:
        model: EmotionCNN in eval() mode on the requested device.
    """
    model = EmotionCNN(num_classes=NUM_CLASSES)

    # weights_only=False needed to load full checkpoint dicts (epoch, val_acc, etc.)
    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )
    # train.py saves a full checkpoint dict — extract just the model weights
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Disable BatchNorm running stats update & Dropout
    return model


# ------------------------------------------------------------------
# Quick sanity-check (run: python model.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()
    model = EmotionCNN().to(device)

    # Simulate a batch of 4 grayscale 48×48 face images
    dummy_input = torch.randn(4, 1, 48, 48, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"[Model] Input  shape : {dummy_input.shape}")
    print(f"[Model] Output shape : {output.shape}  ← (batch, {NUM_CLASSES} emotions)")
    print(f"[Model] Emotion labels: {EMOTION_LABELS}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params    : {total_params:,}")
    print(f"[Model] Trainable params: {trainable:,}")
