# 🧩 **`src/` README — Core Modules (Utilities, Ingestion & Processing)**

This folder contains the **core source modules** that power the **MLOps Gun Detection** project.
Together, they provide the essential functionality for **dataset ingestion**, **image preprocessing**, **logging**, and **exception handling** — ensuring reproducibility, traceability, and maintainability across all stages of the pipeline.

## 📁 **Folder Overview**

```text
src/
├─ custom_exception.py   # Unified and detailed exception handling
├─ logger.py             # Centralised logging configuration
├─ data_ingestion.py     # Handles Kaggle dataset download and extraction
└─ data_processing.py    # Custom PyTorch dataset for image + label preprocessing
```

## ⚠️ **`custom_exception.py` — Unified Error Handling**

### 🧠 Purpose

Defines a `CustomException` class that provides detailed context for debugging runtime errors across all modules — including during **OpenCV operations**, **data ingestion**, and **model training**.

### 🔑 Key Features

* Reports both the **file name** and **line number** where the error occurred.
* Includes a readable **traceback** for fast debugging.
* Works whether you pass:

  * the `sys` module,
  * an exception instance, or
  * nothing (defaults to `sys.exc_info()` automatically).

### 💻 Example Usage

```python
from src.custom_exception import CustomException
import sys
import cv2

try:
    image = cv2.imread("nonexistent_image.jpg")
    if image is None:
        raise FileNotFoundError("Image could not be loaded.")
except Exception as e:
    raise CustomException("Error during image loading", sys) from e
```

### 🧾 Output Example

```
Error in /mlops-gun-detection/src/data_processing.py, line 64: Error during image loading
Traceback (most recent call last):
  File "/mlops-gun-detection/src/data_processing.py", line 64, in <module>
    image = cv2.imread("nonexistent_image.jpg")
FileNotFoundError: Image could not be loaded.
```

This ensures that every pipeline failure is clearly logged, context-rich, and easy to trace.

## 🪵 **`logger.py` — Centralised Logging**

### 🧠 Purpose

Provides a **standardised logging system** used across all modules in the MLOps Gun Detection project.
Each log entry includes a timestamp and is written to a daily log file under the `logs/` directory — creating a structured record of all ingestion, preprocessing, and training activity.

### 🗂️ Log File Format

* Directory: `logs/`
* File name: `log_YYYY-MM-DD.log`
* Example: `logs/log_2025-10-20.log`

### ⚙️ Default Configuration

* Logging level: `INFO`
* Format:

  ```
  %(asctime)s - %(levelname)s - %(message)s
  ```

### 💻 Example Usage

```python
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("Starting gun detection data ingestion.")
logger.warning("Low confidence detected in bounding box.")
logger.error("Failed to extract image folder.")
```

### 🧾 Output Example

```
2025-10-20 18:45:21,112 - INFO - Starting gun detection data ingestion.
2025-10-20 18:45:22,509 - WARNING - Low confidence detected in bounding box.
2025-10-20 18:45:23,021 - ERROR - Failed to extract image folder.
```

## 📦 **`data_ingestion.py` — Dataset Download & Extraction**

### 🧠 Purpose

Implements the `DataIngestion` class responsible for retrieving and preparing the **Guns Object Detection dataset** from **Kaggle via KaggleHub**.
It automates the creation of the `artifacts/raw/` directory, handles ZIP extraction, and organises the dataset structure.

### 🔄 Workflow Overview

| Step | Description                                                                 |
| :--- | :-------------------------------------------------------------------------- |
| 1️⃣  | Create a `raw/` folder under `artifacts/` if it does not exist.             |
| 2️⃣  | Download the Kaggle dataset specified in `config/data_ingestion_config.py`. |
| 3️⃣  | Extract the ZIP archive if one is provided.                                 |
| 4️⃣  | Move `Images/` and `Labels/` into `artifacts/raw/`.                         |

### 💻 Example Usage

```python
from src.data_ingestion import DataIngestion
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

if __name__ == "__main__":
    ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    ingestion.run()
```

### 🧾 Output Example

```
2025-10-20 18:45:21,100 - INFO - 🚀 Starting Data Ingestion Pipeline...
2025-10-20 18:45:22,211 - INFO - Downloaded dataset to /tmp/kagglehub_cache/guns-object-detection
2025-10-20 18:45:23,532 - INFO - Extracting ZIP archive...
2025-10-20 18:45:25,114 - INFO - Images moved successfully.
2025-10-20 18:45:26,221 - INFO - Labels moved successfully.
2025-10-20 18:45:27,002 - INFO - ✅ Data Ingestion Pipeline completed successfully.
```

This ensures that dataset ingestion is automated, reproducible, and fully traceable via unified logging.

## 🧹 **`data_processing.py` — Image & Label Preprocessing**

### 🧠 Purpose

Defines the `GunDataset` class used for loading, normalising, and preparing images and bounding boxes for training.
This module transforms raw images and text-based label files into PyTorch tensors that can be directly consumed by the Faster R-CNN model.

### 🔑 Key Features

* Reads and normalises image data using **OpenCV**.
* Loads label files containing bounding box coordinates.
* Computes bounding box areas and assigns class labels.
* Moves tensors automatically to the specified **device (CPU/GPU)**.
* Logs each operation and raises a `CustomException` when a file is missing or corrupted.

### 💻 Example Usage

```python
from src.data_processing import GunDataset
import torch

root_path = "artifacts/raw"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = GunDataset(root=root_path, device=device)
image, target = dataset[0]

print("Image shape:", image.shape)
print("Bounding boxes:", target["boxes"])
```

### 🧾 Output Example

```
2025-10-20 19:05:42,018 - INFO - ✅ Data Processing Initialised...
2025-10-20 19:05:42,152 - INFO - 📸 Loading data for index 0
2025-10-20 19:05:42,384 - INFO - Image Path: artifacts/raw/Images/001.jpeg
Image shape: torch.Size([3, 480, 640])
Bounding boxes: tensor([[ 54., 129., 212., 320.]])
```

This module bridges the gap between **raw dataset ingestion** and **model training**, ensuring consistent preprocessing and robust error handling.

## 🧩 **Integration Guidelines**

| Module Type        | Use `CustomException` for…                        | Use `get_logger` for…                                   |
| :----------------- | :------------------------------------------------ | :------------------------------------------------------ |
| Data Ingestion     | File I/O, missing directories, or download errors | Tracking dataset downloads and extraction progress      |
| Data Processing    | OpenCV read or label parsing errors               | Logging preprocessing, normalisation, and label loading |
| Training Pipeline  | Model setup, GPU allocation, or optimiser errors  | Logging epoch progress and training losses              |
| Inference Pipeline | Detection or rendering issues                     | Logging predictions and confidence scores               |

By combining **centralised logging** with **structured error handling**, every stage of the Gun Detection pipeline becomes fully auditable, reliable, and easy to maintain.

✅ **In summary:**

* `custom_exception.py` delivers **context-rich error messages**.
* `logger.py` ensures **consistent timestamped logging**.
* `data_ingestion.py` automates **dataset download and preparation**.
* `data_processing.py` provides **robust data loading and preprocessing** for model training.

Together, these modules form the **operational backbone** of the **MLOps Gun Detection pipeline**, ensuring a smooth transition from data ingestion to preprocessing and model experimentation.