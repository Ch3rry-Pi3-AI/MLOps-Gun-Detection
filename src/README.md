# 🧩 **`src/` README — Core Modules (Utilities, Ingestion, Processing & Architecture)**

This folder contains the **core source modules** that power the **MLOps Gun Detection** project.
Together, they provide the foundational functionality for **data ingestion**, **image preprocessing**, **model architecture**, **logging**, and **exception handling** — ensuring a fully traceable, reproducible, and maintainable workflow across all stages of the pipeline.

## 📁 **Folder Overview**

```text
src/
├─ custom_exception.py     # Unified and detailed exception handling
├─ logger.py               # Centralised logging configuration
├─ data_ingestion.py       # Handles Kaggle dataset download and extraction
├─ data_processing.py      # Custom PyTorch dataset for image + label preprocessing
└─ model_architecture.py   # Defines Faster R-CNN model and training loop
```

## ⚠️ **`custom_exception.py` — Unified Error Handling**

### 🧠 Purpose

Defines a `CustomException` class that provides detailed context for debugging runtime errors across all modules — including **OpenCV operations**, **dataset ingestion**, and **model training**.

### 🔑 Key Features

* Displays the **file name** and **line number** where the error occurred.
* Includes a formatted **traceback** for improved readability.
* Works whether you pass:

  * the `sys` module,
  * an exception instance, or
  * nothing (defaults to `sys.exc_info()` automatically).

### 💻 Example Usage

```python
from src.custom_exception import CustomException
import sys, cv2

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

Ensures that all exceptions raised in the pipeline are clear, traceable, and consistently formatted.

## 🪵 **`logger.py` — Centralised Logging**

### 🧠 Purpose

Provides a **standardised logging system** used across all modules in the Gun Detection project.
Each log entry includes a timestamp and is written to a daily log file under the `logs/` directory, ensuring all ingestion, preprocessing, and training activities are traceable.

### 🗂️ Log File Format

* Directory: `logs/`
* File name: `log_YYYY-MM-DD.log`
* Example: `logs/log_2025-10-21.log`

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
2025-10-21 10:32:45,308 - INFO - Starting gun detection data ingestion.
2025-10-21 10:32:46,512 - WARNING - Low confidence detected in bounding box.
2025-10-21 10:32:47,024 - ERROR - Failed to extract image folder.
```

## 📦 **`data_ingestion.py` — Dataset Download & Extraction**

### 🧠 Purpose

Implements the `DataIngestion` class, which retrieves and prepares the **Guns Object Detection dataset** from **Kaggle via KaggleHub**.
Handles directory creation, ZIP extraction, and dataset organisation under the `artifacts/raw/` directory.

### 🔄 Workflow Overview

| Step | Description                                                               |
| :--- | :------------------------------------------------------------------------ |
| 1️⃣  | Create a `raw/` folder under `artifacts/` if it does not exist.           |
| 2️⃣  | Download the Kaggle dataset defined in `config/data_ingestion_config.py`. |
| 3️⃣  | Extract the ZIP archive if one exists.                                    |
| 4️⃣  | Move `Images/` and `Labels/` folders into `artifacts/raw/`.               |

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
2025-10-21 10:34:21,100 - INFO - 🚀 Starting Data Ingestion Pipeline...
2025-10-21 10:34:22,201 - INFO - Downloaded dataset to /tmp/kagglehub_cache/guns-object-detection
2025-10-21 10:34:23,524 - INFO - Extracting ZIP archive...
2025-10-21 10:34:25,111 - INFO - Images moved successfully.
2025-10-21 10:34:26,215 - INFO - Labels moved successfully.
2025-10-21 10:34:27,009 - INFO - ✅ Data Ingestion completed successfully.
```

## 🧹 **`data_processing.py` — Image & Label Preprocessing**

### 🧠 Purpose

Defines the `GunDataset` class for loading, normalising, and preparing images and bounding boxes for training.
Transforms raw images and text-based labels into PyTorch tensors ready for object detection models such as Faster R-CNN.

### 🔑 Key Features

* Reads and normalises images using **OpenCV**.
* Loads bounding box coordinates from text labels.
* Computes box areas and assigns class labels.
* Moves tensors automatically to **CPU or GPU**.
* Logs every operation and raises `CustomException` on file or I/O errors.

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
2025-10-21 10:35:11,018 - INFO - ✅ Data Processing Initialised...
2025-10-21 10:35:11,146 - INFO - 📸 Loading data for index 0
2025-10-21 10:35:11,394 - INFO - Image Path: artifacts/raw/Images/001.jpeg
Image shape: torch.Size([3, 480, 640])
Bounding boxes: tensor([[ 54., 129., 212., 320.]])
```

This module bridges the gap between **raw ingestion** and **model architecture**, ensuring reliable preprocessing for every image–label pair.

## 🧠 **`model_architecture.py` — Faster R-CNN Definition**

### 🧠 Purpose

Implements the `FasterRCNNModel` class that builds, compiles, and trains a **Faster R-CNN (ResNet-50 FPN)** model.
This is the backbone of the detection pipeline, transforming processed datasets into trained weights ready for inference.

### 🔑 Key Features

* Loads **pretrained Faster R-CNN (ResNet-50 FPN)** from TorchVision.
* Customises the classifier head for target classes (guns vs background).
* Supports model compilation with **Adam optimiser** and configurable learning rate.
* Includes a structured **training loop** with per-epoch logging.
* Integrates with `logger` and `CustomException` for robust tracking and error handling.

### 💻 Example Usage

```python
from src.model_architecture import FasterRCNNModel
from torch.utils.data import DataLoader
from src.data_processing import GunDataset
import torch

# Prepare dataset and dataloader
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = GunDataset(root="artifacts/raw", device=device)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Initialise and train model
model = FasterRCNNModel(num_classes=2, device=device)
model.compile(lr=1e-4)
model.train(train_loader, num_epochs=5)
```

### 🧾 Output Example

```
2025-10-21 10:36:41,112 - INFO - ✅ Model architecture initialised successfully.
2025-10-21 10:36:41,998 - INFO - ⚙️ Model compiled successfully with learning rate 0.0001
2025-10-21 10:36:42,110 - INFO - 🚀 Epoch 1 started...
2025-10-21 10:37:11,322 - INFO - ✅ Epoch 1 completed | Total Loss: 1.3478
```

This module introduces the **model layer** of the Gun Detection pipeline — preparing the architecture for structured training and evaluation in the next stage.

## 🧩 **Integration Guidelines**

| Module Type        | Use `CustomException` for…                        | Use `get_logger` for…                                      |
| :----------------- | :------------------------------------------------ | :--------------------------------------------------------- |
| Data Ingestion     | File I/O, missing directories, or download errors | Tracking dataset downloads and extraction progress         |
| Data Processing    | OpenCV read or label parsing errors               | Logging preprocessing, normalisation, and label loading    |
| Model Architecture | Optimiser setup or training errors                | Logging model creation, compilation, and training progress |
| Training Pipeline  | Runtime issues or device mismatches               | Logging loss metrics and validation performance            |
| Inference Pipeline | Detection or rendering issues                     | Logging predictions and confidence thresholds              |

By combining **centralised logging**, **structured exception handling**, and **modular architecture**, every component of the Gun Detection pipeline remains transparent, reliable, and easy to debug.

✅ **In summary:**

* `custom_exception.py` provides **context-rich error messages**.
* `logger.py` maintains **consistent timestamped logs**.
* `data_ingestion.py` automates **dataset download and extraction**.
* `data_processing.py` delivers **robust data loading and preprocessing**.
* `model_architecture.py` defines the **Faster R-CNN backbone and training routine**.

Together, these modules form the **core operational backbone** of the **MLOps Gun Detection** system, enabling a seamless transition from **data preparation** to **model training** in the next stage.
