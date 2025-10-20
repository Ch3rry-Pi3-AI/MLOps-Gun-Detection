# 🏗️ **Stage 01 — Data Ingestion (MLOps Gun Detection)**

This stage implements the **data ingestion pipeline** for the **MLOps Gun Detection** project.
It automates the retrieval and preparation of the **gun detection dataset** from **Kaggle via KaggleHub**, structuring the results into a standardised artifact directory for subsequent analysis and model development.

The ingestion process ensures **reproducibility**, **traceability**, and **data integrity** by combining consistent configuration management, unified logging, and robust exception handling.

## 🗂️ **Project Structure**

```text
mlops-gun-detection/
├── artifacts/                     # 📦 Dataset storage for raw and processed data
│   └── raw/                       # Unprocessed dataset components
│       ├── Images/                # Gun detection images
│       └── Labels/                # Bounding box coordinate files for detected guns
├── config/                        # ⚙️ Configuration files and environment settings
│   ├── __init__.py
│   └── data_ingestion_config.py   # Dataset source and target directory configuration
└── src/                           # 🧠 Core utilities and ingestion module
    ├── __init__.py
    └── data_ingestion.py          # Handles dataset download and extraction from KaggleHub
```

> 💡 **Note:** The `artifacts/` directory is automatically created during data ingestion if it does not already exist.

## ⚙️ **Overview**

The **Data Ingestion** stage performs the following operations:

1. **Creates the required directory structure** under `artifacts/raw/`.
2. **Downloads the dataset** from Kaggle using `kagglehub`.
3. **Extracts and organises** the dataset contents into `Images/` and `Labels/` folders.
4. **Logs every step** of the process with timestamps for full traceability.

This modular design ensures that future stages — such as preprocessing and model training — can seamlessly build upon a clean, well-organised data foundation.

## ⚡ **Key Components**

### 1️⃣ `config/data_ingestion_config.py`

Defines dataset and target directory configuration for ingestion.

```python
DATASET_NAME = "issaisasank/guns-object-detection"
TARGET_DIR = "artifacts"
```

### 2️⃣ `src/data_ingestion.py`

Implements the **DataIngestion** class, which automates dataset download and extraction using **KaggleHub**.

**Core functionality includes:**

* Directory creation under `artifacts/raw/`
* Downloading the specified Kaggle dataset
* Moving image and label files to structured subfolders
* Unified logging and exception handling for full traceability

**Example Usage:**

```python
from src.data_ingestion import DataIngestion
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

if __name__ == "__main__":
    ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    ingestion.run()
```

## 🧾 **Expected Output**

```
2025-10-20 00:56:17,580 - INFO - DataIngestion initialised with dataset: issaisasank/guns-object-detection
2025-10-20 00:56:17,580 - INFO - 🚀 Starting Data Ingestion Pipeline...
2025-10-20 00:56:17,581 - INFO - Starting dataset download from Kaggle: issaisasank/guns-object-detection
2025-10-20 00:56:18,325 - INFO - Dataset downloaded successfully to temporary path: C:\Users\HP\.cache\kagglehub\datasets\issaisasank\guns-object-detection\versions\1
2025-10-20 00:56:18,326 - WARNING - Images folder does not exist in the extracted archive.
2025-10-20 00:56:18,360 - INFO - Labels moved successfully.
2025-10-20 00:56:18,360 - INFO - ✅ Data Ingestion Pipeline completed successfully.
```

## 🚀 **Next Stage**

The next workflow stage will focus on **Experimentation using Kaggle Notebooks**.
This phase will include:

* Exploring the ingested dataset and verifying data integrity
* Inspecting sample images and label coordinates
* Performing early experiments on bounding box visualisation and annotation consistency
* Documenting findings for subsequent preprocessing and model training stages

✅ **In summary:**
The **Data Ingestion stage** establishes a complete, automated process for downloading, extracting, and structuring the dataset, forming a reliable foundation for all future stages in the **MLOps Gun Detection** pipeline.