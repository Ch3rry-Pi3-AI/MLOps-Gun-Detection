"""
model_training.py
-----------------
Defines the `ModelTraining` class responsible for orchestrating
the full training workflow of the Faster R-CNN model.

This module integrates:
- Data splitting into training/validation sets
- Model and optimiser initialisation
- Epoch-based training and validation loop
- TensorBoard logging for visualisation
- Automated model checkpoint saving
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
import time
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------------------------
# 🧰 Internal Imports
# -------------------------------------------------------------------
from src.model_architecture import FasterRCNNModel
from src.data_processing import GunDataset
from src.logger import get_logger
from src.custom_exception import CustomException

# -------------------------------------------------------------------
# 🧾 Logger Setup
# -------------------------------------------------------------------
logger = get_logger(__name__)

# -------------------------------------------------------------------
# 💾 Model Save Directory
# -------------------------------------------------------------------
MODEL_SAVE_PATH = "artifacts/models/"
_os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


# -------------------------------------------------------------------
# 🧠 ModelTraining Class Definition
# -------------------------------------------------------------------
class ModelTraining:
    """
    Handles model setup, training, and validation using the
    Faster R-CNN architecture.

    Parameters
    ----------
    model_class : type
        Model class used to instantiate the Faster R-CNN.
    num_classes : int
        Number of output classes (including background).
    learning_rate : float
        Learning rate for the optimiser.
    epochs : int
        Number of epochs for training.
    dataset_path : str
        Path to the dataset directory (containing Images and Labels).
    device : str
        Compute device to use ("cpu" or "cuda").
    """

    def __init__(self, model_class, num_classes, learning_rate, epochs, dataset_path, device):
        # Store configuration parameters
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        # -------------------------------------------------------------------
        # 📊 TensorBoard Setup
        # -------------------------------------------------------------------
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"tensorboard_logs/{timestamp}"
        _os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        try:
            # -------------------------------------------------------------------
            # 🧩 Initialise Model and Optimiser
            # -------------------------------------------------------------------
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.to(self.device)
            logger.info("✅ Model moved to target device.")

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info(f"⚙️ Optimiser initialised with learning rate {self.learning_rate}")

        except Exception as e:
            logger.error(f"❌ Failed to initialise model training: {e}")
            raise CustomException("Failed to initialise model training", e)

    def collate_fn(self, batch):
        """
        Custom collate function for variable-sized target lists.

        Parameters
        ----------
        batch : list
            List of tuples (image_tensor, target_dict).

        Returns
        -------
        Tuple
            Tuple containing separate lists of images and targets.
        """
        # Unpack batch items into images and targets
        return tuple(zip(*batch))

    def split_dataset(self):
        """
        Splits the dataset into training and validation sets,
        then creates DataLoaders for both.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            Train and validation DataLoaders.
        """
        try:
            # Load dataset and limit to a subset (optional)
            dataset = GunDataset(self.dataset_path, self.device)
            dataset = torch.utils.data.Subset(dataset, range(300))  # limit for faster testing

            # Compute train/validation split sizes
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            # Split dataset
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn=self.collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=self.collate_fn
            )

            logger.info("✅ Dataset successfully split into training and validation sets.")
            return train_loader, val_loader

        except Exception as e:
            logger.error(f"❌ Failed to split dataset: {e}")
            raise CustomException("Failed to split dataset", e)

    def train(self):
        """
        Executes the training loop for the Faster R-CNN model.

        Includes:
        - Forward/backward passes
        - Loss calculation
        - Validation after each epoch
        - TensorBoard logging
        - Model checkpoint saving
        """
        try:
            # Prepare DataLoaders
            train_loader, val_loader = self.split_dataset()

            # -------------------------------------------------------------------
            # 🔁 Training Loop
            # -------------------------------------------------------------------
            for epoch in range(self.epochs):
                logger.info(f"🚀 Starting Epoch {epoch + 1}/{self.epochs}")
                self.model.train()

                # Track total loss
                for i, (images, targets) in enumerate(train_loader):
                    # Zero out gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    losses = self.model(images, targets)

                    # Handle loss dictionary returned by Faster R-CNN
                    if isinstance(losses, dict):
                        total_loss = 0
                        for key, value in losses.items():
                            if isinstance(value, torch.Tensor):
                                total_loss += value

                        if total_loss == 0:
                            logger.error("❌ Loss computation failed — total loss is zero.")
                            raise ValueError("Total loss value is zero.")

                        # Log training loss to TensorBoard
                        self.writer.add_scalar("Loss/train", total_loss.item(), epoch * len(train_loader) + i)

                    else:
                        total_loss = losses[0]
                        self.writer.add_scalar("Loss/train", total_loss.item(), epoch * len(train_loader) + i)

                    # Backpropagation
                    total_loss.backward()
                    self.optimizer.step()

                # Flush TensorBoard logs
                self.writer.flush()

                # -------------------------------------------------------------------
                # 🧪 Validation Loop
                # -------------------------------------------------------------------
                self.model.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        val_losses = self.model(images, targets)
                        logger.info(f"📉 Validation Loss Type: {type(val_losses)}")
                        logger.info(f"VAL_LOSS: {val_losses}")

                # -------------------------------------------------------------------
                # 💾 Save Model Checkpoint
                # -------------------------------------------------------------------
                model_path = _os.path.join(MODEL_SAVE_PATH, "fasterrcnn.pth")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"✅ Model saved successfully at {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to train model: {e}")
            raise CustomException("Failed to train model", e)


# -------------------------------------------------------------------
# 🧪 Standalone Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate training pipeline
    training = ModelTraining(
        model_class=FasterRCNNModel,
        num_classes=2,
        learning_rate=0.0001,
        dataset_path="artifacts/raw/",
        device=device,
        epochs=1,
    )

    # Start training
    training.train()