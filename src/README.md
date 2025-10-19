Got it ✅ Here’s your adapted version for the **Gun Detection** MLOps project — tailored for the **initial setup stage** of the `src/` folder (no Airflow references, and rewritten around OpenCV-based image processing and detection pipelines):



# `src/` README — Core Utilities (Custom Exception & Logger)

This folder contains **foundational utilities** for the **Gun Detection** MLOps pipeline.
These modules establish **consistent logging** and **error handling** practices across all components — from image ingestion and preprocessing to bounding box detection and evaluation.



## 📁 Folder Overview

```text
src/
├─ custom_exception.py   # Unified and detailed exception handling
└─ logger.py             # Centralised logging configuration
```



## ⚠️ `custom_exception.py` — Unified Error Handling

### Purpose

Defines a **CustomException** class that provides detailed context for debugging errors that occur anywhere in the pipeline — for example, during **OpenCV image loading**, **model inference**, or **bounding box rendering**.

### Key Features

* Displays the **file name** and **line number** where the error occurred.
* Includes a formatted **traceback**, making debugging simple and consistent.
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

This ensures all exceptions are reported in a **uniform and traceable** format during model training, inference, or evaluation.



## 🪵 `logger.py` — Centralised Logging

### Purpose

Provides a **standardised logging setup** for the project.
Each log entry is timestamped and saved to a dated file in a `logs/` directory, ensuring a clear audit trail for **model runs**, **OpenCV operations**, and **pipeline events**.

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

logger.info("Starting gun detection pipeline.")
logger.warning("Low confidence detected for bounding box.")
logger.error("Failed to load model weights.")
```

### Output Example

```
2025-10-20 18:45:21,112 - INFO - Starting gun detection pipeline.
2025-10-20 18:45:22,509 - WARNING - Low confidence detected for bounding box.
2025-10-20 18:45:23,021 - ERROR - Failed to load model weights.
```



## 🧩 Integration Guidelines

| Module Type        | Use `CustomException` for…                       | Use `get_logger` for…                                |
| ------------------ | ------------------------------------------------ | ---------------------------------------------------- |
| Data Ingestion     | File I/O or invalid image path errors            | Tracking number of images loaded                     |
| Preprocessing      | OpenCV operations or transformation failures     | Logging resizing, augmentation, or normalisation     |
| Detection Pipeline | Model inference or bounding box rendering errors | Confidence scores, detections per image, performance |
| Evaluation Scripts | Metric computation or file output issues         | Summary statistics and timing logs                   |

**Tip:** Use both together for best traceability:

```python
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def detect_guns(image_path):
    try:
        logger.info(f"Processing {image_path}...")
        # Simulate detection failure
        raise ValueError("Bounding box coordinates invalid.")
    except Exception as e:
        logger.error("Detection failed.")
        raise CustomException("Gun detection error", sys) from e
```



✅ **In summary:**

* `custom_exception.py` ensures **consistent and informative error messages**.
* `logger.py` provides **structured, timestamped logging**.

Together they form the **core reliability layer** of the **MLOps Gun Detection** pipeline — supporting smooth debugging, traceability, and maintainability across all modules.



Would you like me to include a short “Next Steps” section at the bottom (e.g., mentioning that model, preprocessing, and inference modules will be added next under `src/`)? It can help tie this README into the overall project scaffolding.