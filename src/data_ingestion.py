"""
data_ingestion.py
-----------------
Implements the DataIngestion class responsible for downloading and extracting
the gun detection dataset from Kaggle via KaggleHub into the local `artifacts/raw/` directory.

This module integrates with:
- Centralised logging (`src.logger`)
- Custom exception handling (`src.custom_exception`)
- Configuration management (`config.data_ingestion_config`)

Usage
-----
Example:
    python src/data_ingestion.py

Notes
-----
- Requires KaggleHub access to the dataset defined in `config.data_ingestion_config`.
- Automatically creates directories under `artifacts/raw/` if they do not exist.
- Moves extracted image and label folders into the `artifacts/raw/` structure.
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
# Standard Library Imports
# -------------------------------------------------------------------
import shutil
import zipfile

# -------------------------------------------------------------------
# Third-Party Imports
# -------------------------------------------------------------------
import kagglehub

# -------------------------------------------------------------------
# Internal Imports
# -------------------------------------------------------------------
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

# -------------------------------------------------------------------
# Logger Setup
# -------------------------------------------------------------------
logger = get_logger(__name__)


# -------------------------------------------------------------------
# Class: DataIngestion
# -------------------------------------------------------------------
class DataIngestion:
    """
    Handles the ingestion of the gun detection dataset from Kaggle.

    This includes:
    - Downloading the dataset using KaggleHub.
    - Creating the raw data directory if it does not exist.
    - Extracting image and label files from a ZIP archive.
    - Moving extracted folders into the structured artifacts/raw directory.

    Parameters
    ----------
    dataset_name : str
        The Kaggle dataset identifier (e.g., "issaisasank/guns-object-detection").
    target_dir : str
        The root directory where artifacts will be stored.
    """

    def __init__(self, dataset_name: str, target_dir: str) -> None:
        self.dataset_name = dataset_name
        self.target_dir = target_dir
        logger.info(f"DataIngestion initialised with dataset: {self.dataset_name}")

    # -------------------------------------------------------------------
    # Method: create_raw_dir
    # -------------------------------------------------------------------
    def create_raw_dir(self) -> str:
        """
        Creates the raw data directory under the target artifacts folder.

        Returns
        -------
        str
            Path to the created (or existing) raw data directory.

        Raises
        ------
        CustomException
            If directory creation fails.
        """
        raw_dir = _os.path.join(self.target_dir, "raw")

        if not _os.path.exists(raw_dir):
            try:
                _os.makedirs(raw_dir)
                logger.info(f"Created directory: {raw_dir}")
            except Exception as e:
                logger.error("Error while creating raw directory.")
                raise CustomException("Failed to create raw directory.", e)

        return raw_dir

    # -------------------------------------------------------------------
    # Method: extract_images_and_labels
    # -------------------------------------------------------------------
    def extract_images_and_labels(self, path: str, raw_dir: str) -> None:
        """
        Extracts image and label folders from the downloaded dataset archive.

        Parameters
        ----------
        path : str
            Path to the downloaded dataset or ZIP file.
        raw_dir : str
            Target directory where extracted folders will be moved.

        Raises
        ------
        CustomException
            If extraction or folder movement fails.
        """
        try:
            # Extract ZIP file contents if applicable
            if path.endswith(".zip"):
                logger.info("Extracting ZIP archive...")
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(path)

            # Define expected folders
            images_folder = _os.path.join(path, "Images")
            labels_folder = _os.path.join(path, "Labels")

            # Move extracted image files
            if _os.path.exists(images_folder):
                shutil.move(images_folder, _os.path.join(raw_dir, "Images"))
                logger.info("Images moved successfully.")
            else:
                logger.warning("Images folder does not exist in the extracted archive.")

            # Move extracted label files
            if _os.path.exists(labels_folder):
                shutil.move(labels_folder, _os.path.join(raw_dir, "Labels"))
                logger.info("Labels moved successfully.")
            else:
                logger.warning("Labels folder does not exist in the extracted archive.")

        except Exception as e:
            logger.error("Error while extracting dataset contents.")
            raise CustomException("Error during dataset extraction.", e)

    # -------------------------------------------------------------------
    # Method: download_dataset
    # -------------------------------------------------------------------
    def download_dataset(self, raw_dir: str) -> None:
        """
        Downloads the dataset from Kaggle using KaggleHub.

        Parameters
        ----------
        raw_dir : str
            Directory where extracted data should be placed after download.

        Raises
        ------
        CustomException
            If download or extraction fails.
        """
        try:
            logger.info(f"Starting dataset download from Kaggle: {self.dataset_name}")
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded successfully to temporary path: {path}")

            # Extract and move images/labels to raw directory
            self.extract_images_and_labels(path, raw_dir)

        except Exception as e:
            logger.error("Error while downloading dataset.")
            raise CustomException("Error while downloading dataset.", e)

    # -------------------------------------------------------------------
    # Method: run
    # -------------------------------------------------------------------
    def run(self) -> None:
        """
        Executes the full data ingestion workflow.

        Steps
        -----
        1. Create the raw data directory if missing.
        2. Download the dataset from Kaggle via KaggleHub.
        3. Extract and organise the dataset into artifacts/raw.

        Raises
        ------
        CustomException
            If any step in the ingestion pipeline fails.
        """
        try:
            logger.info("🚀 Starting Data Ingestion Pipeline...")
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("✅ Data Ingestion Pipeline completed successfully.")
        except Exception as e:
            logger.error("❌ Error during Data Ingestion Pipeline.")
            raise CustomException("Error during data ingestion pipeline.", e)


# -------------------------------------------------------------------
# Script Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    data_ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    data_ingestion.run()