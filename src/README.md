# `src/` README — Core Modules (Utilities & Data Ingestion)

This folder contains the **core modules** that power the **Gun Detection** MLOps pipeline.
These modules provide the essential building blocks for **logging**, **error handling**, and **data ingestion**, ensuring consistency, traceability, and maintainability across all stages — from dataset download to image preprocessing and bounding box detection.

## 📁 Folder Overview

```text
src/
├─ custom_exception.py   # Unified and detailed exception handling
├─ logger.py             # Centralised logging configuration
└─ data_ingestion.py     # Handles Kaggle dataset download and extraction
```

## ⚠️ `custom_exception.py` — Unified Error Handling

### Purpose

Defines a **CustomException** class that provides detailed context for debugging errors that occur anywhere in the pipeline — such as during **OpenCV image loading**, **file extraction**, or **model inference**.

### Key Features

* Displays the **file name** and **line number** where the error occurred.
* Includes a formatted **traceback**, improving readability and consistency.
* Works whether you pass:

  * the `sys` module,
  * an exception instance, or
  * nothing (defaults to current `sys.exc_info()`).

### Example Usage

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

### Output Example

```
Error in /mlops-gun-detection/src/image_preprocessor.py, line 25: Error during image loading
Traceback (most recent call last):
  File "/mlops-gun-detection/src/image_preprocessor.py", line 25, in <module>
    image = cv2.imread("nonexistent_image.jpg")
FileNotFoundError: Image could not be loaded.
```

This ensures all pipeline-related exceptions are **clearly logged and traceable** during ingestion, preprocessing, or inference.

## 🪵 `logger.py` — Centralised Logging

### Purpose

Provides a **standardised logging system** used across all modules in the MLOps Gun Detection project.
Each log entry includes a timestamp and is written to a daily log file under the `logs/` directory — creating a structured record of all operations, including downloads, extractions, and detections.

### Log File Format

* Directory: `logs/`
* File name: `log_YYYY-MM-DD.log`
* Example: `logs/log_2025-10-20.log`

### Default Configuration

* Logging level: `INFO`
* Format:

  ```
  %(asctime)s - %(levelname)s - %(message)s
  ```

### Example Usage

```python
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("Starting gun detection data ingestion.")
logger.warning("Low confidence detected in bounding box.")
logger.error("Failed to extract image folder.")
```

### Output Example

```
2025-10-20 18:45:21,112 - INFO - Starting gun detection data ingestion.
2025-10-20 18:45:22,509 - WARNING - Low confidence detected in bounding box.
2025-10-20 18:45:23,021 - ERROR - Failed to extract image folder.
```

## 📦 `data_ingestion.py` — Dataset Download & Extraction

### Purpose

Implements the **DataIngestion** class responsible for retrieving and preparing the **gun detection dataset** from **Kaggle via KaggleHub**.
It handles directory creation, ZIP extraction, and organisation of image and label folders into the standard `artifacts/raw/` structure.

### Workflow Overview

| Step | Description                                                                   |
| ---- | ----------------------------------------------------------------------------- |
| 1️⃣  | Create a `raw/` folder under the `artifacts/` directory if it does not exist. |
| 2️⃣  | Download the Kaggle dataset specified in `config/data_ingestion_config.py`.   |
| 3️⃣  | Extract the downloaded ZIP archive, if present.                               |
| 4️⃣  | Move `Images/` and `Labels/` folders into `artifacts/raw/`.                   |

### Example Usage

```python
from src.data_ingestion import DataIngestion
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

if __name__ == "__main__":
    ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    ingestion.run()
```

### Output Example

```
2025-10-20 18:45:21,100 - INFO - 🚀 Starting Data Ingestion Pipeline...
2025-10-20 18:45:22,211 - INFO - Downloaded dataset to /tmp/kagglehub_cache/guns-object-detection
2025-10-20 18:45:23,532 - INFO - Extracting ZIP archive...
2025-10-20 18:45:25,114 - INFO - Images moved successfully.
2025-10-20 18:45:26,221 - INFO - Labels moved successfully.
2025-10-20 18:45:27,002 - INFO - ✅ Data Ingestion Pipeline completed successfully.
```

This process guarantees that the **dataset ingestion stage is automated, reproducible, and fully traceable** through unified logging and exception handling.

## 🧩 Integration Guidelines

| Module Type        | Use `CustomException` for…                        | Use `get_logger` for…                               |
| ------------------ | ------------------------------------------------- | --------------------------------------------------- |
| Data Ingestion     | File I/O, missing directories, or download errors | Tracking dataset downloads and extraction progress  |
| Preprocessing      | OpenCV read/resize errors                         | Logging preprocessing, augmentation, or filtering   |
| Detection Pipeline | Model inference or bounding box rendering errors  | Logging detection metrics and confidence thresholds |
| Evaluation Scripts | File saving or metric computation errors          | Logging accuracy, precision, and recall statistics  |

**Tip:** Use both `get_logger` and `CustomException` together to make every stage of the pipeline auditable and failure-tolerant.

✅ **In summary:**

* `custom_exception.py` provides **consistent, context-rich error messages**.
* `logger.py` ensures **structured, timestamped log tracking**.
* `data_ingestion.py` automates **dataset download and preparation**.

Together, these modules form the **core operational backbone** of the **MLOps Gun Detection** pipeline — enabling a reliable, debuggable, and maintainable workflow from the start.
