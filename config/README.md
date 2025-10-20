# ⚙️ **Configuration Directory — `config/`**

The `config/` directory defines all **centralised configuration files** used by the **MLOps Gun Detection** project.
It provides a **single source of truth** for dataset references, target directories, and other key parameters used during the pipeline lifecycle.

At this stage, configuration focuses on **data ingestion**, defining the dataset source and output structure for storing downloaded artifacts.
Additional configuration modules (for preprocessing, model training, and monitoring) will be introduced as the project evolves.

## 🗂️ **Folder Structure**

```
mlops-gun-detection/
└── config/
    ├── __init__.py
    └── data_ingestion_config.py     # Dataset source and artifact directory configuration
```

## 🎯 **Purpose**

These configuration files ensure **consistent and maintainable management of dataset and storage settings** throughout the project — particularly during data ingestion and validation stages.

| File                       | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| `data_ingestion_config.py` | Defines the dataset name and artifact directory used for data ingestion. |

## 🧱 **Overview of `data_ingestion_config.py`**

This file centralises the dataset source and output directory paths used during data ingestion.
It keeps all configuration in one place, ensuring that ingestion scripts remain modular and free from hardcoded values.

### 🧩 **Highlights**

```python
DATASET_NAME = "issaisasank/guns-object-detection"
TARGET_DIR = "artifacts"
```

### 🧠 **Notes**

* The dataset name corresponds to the **Kaggle repository identifier** used for download via **KaggleHub**.
* `TARGET_DIR` defines where the downloaded dataset artifacts will be stored.
* Both values are defined relative to the project root, ensuring portability across environments.
* The configuration can be imported directly into ingestion or preprocessing scripts:

```python
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR
```

## ✅ **Best Practices**

* Avoid hardcoding dataset names or directory paths in scripts — always import them from `config/`.
* Use `os.makedirs(..., exist_ok=True)` when writing ingestion logic to ensure the target directory exists.
* Treat this directory as the **central reference point** for all configurable parameters across the MLOps Gun Detection pipeline.

By maintaining these settings centrally, the Gun Detection project remains **reproducible, maintainable, and scalable** across data ingestion and subsequent development stages.