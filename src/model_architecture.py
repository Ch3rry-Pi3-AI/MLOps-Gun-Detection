"""
model_architecture.py
---------------------
Defines the `FasterRCNNModel` class, responsible for building,
compiling, and training a Faster R-CNN model using a ResNet-50 FPN backbone.

This module integrates:
- Model creation and configuration
- Optimiser setup and compilation
- Training loop with epoch-wise logging
- Exception handling for robust error management
"""

from __future__ import annotations

# -------------------------------------------------------------------
# Temporary Import Path Hack (Option D)
# -------------------------------------------------------------------
# Ensure the project root (parent of this file's directory) is on sys.path
# so that `import src.*` and `import config.*` work when running this file
# as a script: `python src/data_ingestion.py`.
#
# ⚠️ Note:
# - This is a pragmatic, script-friendly workaround.
# - Prefer installing the package in editable mode (`pip install -e .`)
#   or running as a module (`python -m src.data_ingestion`) in the long run.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

# -------------------------------------------------------------------
# 📦 Standard Library & Third-Party Imports
# -------------------------------------------------------------------
import torch
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# -------------------------------------------------------------------
# 🧰 Internal Imports
# -------------------------------------------------------------------
from src.logger import get_logger
from src.custom_exception import CustomException

# -------------------------------------------------------------------
# 🧾 Logger Setup
# -------------------------------------------------------------------
logger = get_logger(__name__)


# -------------------------------------------------------------------
# 🧠 FasterRCNNModel Class Definition
# -------------------------------------------------------------------
class FasterRCNNModel:
    """
    Wrapper class for Faster R-CNN (ResNet-50 FPN) model.

    Handles model creation, compilation, and training loop execution
    with built-in logging and error handling.

    Parameters
    ----------
    num_classes : int
        Number of output classes (including background).
    device : str
        Compute device to use ("cpu" or "cuda").

    Attributes
    ----------
    model : torch.nn.Module
        The Faster R-CNN model instance.
    optimizer : torch.optim.Optimizer
        Optimiser used for gradient updates.
    """

    def __init__(self, num_classes: int, device: str) -> None:
        # Store parameters
        self.num_classes = num_classes
        self.device = device

        # Initialise model and optimiser placeholder
        self.optimizer = None
        self.model = self.create_model().to(self.device)

        logger.info("✅ Model architecture initialised successfully.")

    def create_model(self) -> torch.nn.Module:
        """
        Creates a Faster R-CNN model with a ResNet-50 FPN backbone.

        Returns
        -------
        torch.nn.Module
            Configured Faster R-CNN model with a modified classifier head.
        """
        try:
            # Load pretrained Faster R-CNN with ResNet-50 backbone
            model = fasterrcnn_resnet50_fpn(pretrained=True)

            # Get number of input features for the classifier head
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            # Replace the head with a new one matching the target classes
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

            logger.info("Model created successfully with ResNet-50 FPN backbone.")
            return model

        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            raise CustomException("Failed to create model", e)

    def compile(self, lr: float = 1e-4) -> None:
        """
        Configures the optimiser for model training.

        Parameters
        ----------
        lr : float, default=1e-4
            Learning rate for the optimiser.
        """
        try:
            # Initialise Adam optimiser
            self.optimizer = Adam(self.model.parameters(), lr=lr)
            logger.info(f"⚙️ Model compiled successfully with learning rate {lr}")

        except Exception as e:
            logger.error(f"❌ Failed to compile model: {e}")
            raise CustomException("Failed to compile model", e)

    def train(self, train_loader: torch.utils.data.DataLoader, num_epochs: int = 10) -> None:
        """
        Trains the Faster R-CNN model using the provided DataLoader.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing batches of images and targets.
        num_epochs : int, default=10
            Number of training epochs.
        """
        try:
            # Set model to training mode
            self.model.train()

            # Iterate over epochs
            for epoch in range(1, num_epochs + 1):
                total_loss = 0.0
                logger.info(f"🚀 Epoch {epoch} started...")

                # Iterate through batches
                for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    # Move images and labels to target device
                    images = [img.to(self.device) for img in images]
                    targets = [{key: val.to(self.device) for key, val in target.items()} for target in targets]

                    # Forward pass
                    loss_dict = self.model(images, targets)

                    # Sum all loss components
                    loss = sum(loss for loss in loss_dict.values())

                    # Zero gradients, backpropagate, and update weights
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate loss
                    total_loss += loss.item()

                logger.info(f"✅ Epoch {epoch} completed | Total Loss: {total_loss:.4f}")

        except Exception as e:
            logger.error(f"❌ Failed to train model: {e}")
            raise CustomException("Failed to train model", e)