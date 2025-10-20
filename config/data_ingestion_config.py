"""
data_ingestion_config.py
------------------------
Centralised configuration for dataset and target directory settings
used in the MLOps Gun Detection project.

This module defines the dataset source and output directory
for storing downloaded artifacts. Keeping these paths in one place
ensures consistency and maintainability across the ingestion pipeline.

Usage
-----
Example:
    from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

Notes
-----
- All directories are defined relative to the project root.
- The target directory will be created dynamically if it does not exist.
"""

# -------------------------------------------------------------------
# 📦 DATA INGESTION CONFIGURATION
# -------------------------------------------------------------------
DATASET_NAME = "issaisasank/guns-object-detection"
TARGET_DIR = "artifacts"