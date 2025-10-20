"""
data_processing.py
------------------
Defines the `GunDataset` class used for loading and preprocessing
the Guns Object Detection dataset.

This module handles:
- Reading images and corresponding bounding box labels
- Converting data to PyTorch tensors
- Normalising pixel values
- Moving tensors to the specified device (CPU/GPU)
- Logging progress and raising custom exceptions where necessary
"""
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

_sys.path.insert(0,_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

# -------------------------------------------------------------------
# 📦 Standard Library Imports
# -------------------------------------------------------------------
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

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
# 📂 Custom Dataset Class: GunDataset
# -------------------------------------------------------------------
class GunDataset(Dataset):
    """
    Custom PyTorch Dataset for gun detection.

    Loads images and corresponding bounding box annotations
    from the dataset directory for use with object detection models.

    Parameters
    ----------
    root : str
        Root directory containing 'Images/' and 'Labels/' subfolders.
    device : str, default="cpu"
        Compute device to which tensors are moved ("cpu" or "cuda").

    Attributes
    ----------
    image_path : str
        Path to the folder containing all image files.
    labels_path : str
        Path to the folder containing all label text files.
    img_name : list[str]
        Sorted list of image filenames.
    label_name : list[str]
        Sorted list of label filenames.
    """

    def __init__(self, root: str, device: str = "cpu") -> None:
        # Define image and label directory paths
        self.image_path =_os.path.join(root, "Images")
        self.labels_path =_os.path.join(root, "Labels")
        self.device = device

        # Retrieve and sort image and label filenames
        self.img_name = sorted(_os.listdir(self.image_path))
        self.label_name = sorted(_os.listdir(self.labels_path))

        # Log dataset initialisation
        logger.info("✅ Data Processing Initialised...")

    def __getitem__(self, idx: int):
        """
        Retrieve and preprocess a single image-label pair.

        Parameters
        ----------
        idx : int
            Index of the dataset item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Image tensor (C x H x W)
            - Target dictionary containing:
                - boxes (Tensor): Bounding box coordinates
                - area (Tensor): Area for each bounding box
                - image_id (Tensor): Image index
                - labels (Tensor): Object class labels
        """
        try:
            # Log which index is being loaded
            logger.info(f"📸 Loading data for index {idx}")

            # Construct image path and read image
            image_path =_os.path.join(self.image_path, str(self.img_name[idx]))
            logger.info(f"Image Path: {image_path}")
            image = cv2.imread(image_path)

            # Convert from BGR ➜ RGB and cast to float32
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            # Normalise pixel values to [0,1]
            img_res = img_rgb / 255.0

            # Convert to tensor and reorder dimensions (C, H, W)
            img_res = torch.as_tensor(img_res).permute(2, 0, 1)

            # Construct corresponding label path
            label_name = self.img_name[idx].rsplit('.', 1)[0] + ".txt"
            label_path =_os.path.join(self.labels_path, str(label_name))

            # Check if label file exists
            if not _os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")

            # Initialise target dictionary with empty tensors
            target = {
                "boxes": torch.tensor([]),
                "area": torch.tensor([]),
                "image_id": torch.tensor([idx]),
                "labels": torch.tensor([], dtype=torch.int64)
            }

            # Read label file and extract bounding box coordinates
            with open(label_path, "r") as label_file:
                l_count = int(label_file.readline())  # number of boxes
                box = [list(map(int, label_file.readline().split())) for _ in range(l_count)]

            # If boxes exist, calculate area and labels
            if box:
                area = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
                labels = [1] * len(box)  # class 1 = gun

                # Update target dictionary with tensors
                target["boxes"] = torch.tensor(box, dtype=torch.float32)
                target["area"] = torch.tensor(area, dtype=torch.float32)
                target["labels"] = torch.tensor(labels, dtype=torch.int64)

            # Move image and target tensors to chosen device
            img_res = img_res.to(self.device)
            for key in target:
                target[key] = target[key].to(self.device)

            # Return processed image and labels
            return img_res, target

        except Exception as e:
            # Log and raise a custom exception with details
            logger.error(f"❌ Error while loading data: {e}")
            raise CustomException("Failed to load data", e)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of available images.
        """
        return len(self.img_name)


# -------------------------------------------------------------------
# 🧪 Test Block (Standalone Execution)
# -------------------------------------------------------------------
if __name__ == "__main__":
    root_path = "artifacts/raw"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate dataset for testing
    dataset = GunDataset(root=root_path, device=device)

    # Fetch one sample to verify integrity
    image, target = dataset[0]

    print("🖼️ Image Shape:", image.shape)
    print("📦 Target Keys:", target.keys())
    print("🔲 Bounding Boxes:", target["boxes"])